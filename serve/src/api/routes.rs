use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::body::{Body, Bytes};
use axum::extract::{Query, State};
use axum::http::StatusCode;
use axum::response::sse::{Event, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures::StreamExt;
use serde_json::json;
use tokio_stream::wrappers::ReceiverStream;

use crate::api::anthropic::{
    AnthropicRequest, AnthropicResponse, AnthropicUsage, ContentBlock, count_input_tokens,
};
use crate::api::types::{EmbedRequest, EvalBestQuery, EvaluateRequest, GenerateRequest, RoutingInfo};
use crate::error::ServeError;
use crate::eval::EvalRecord;
use crate::router::RoutingDecision;

use super::AppState;

pub async fn root_head() -> impl IntoResponse {
    StatusCode::OK
}

pub async fn root_get() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

pub async fn health() -> impl IntoResponse {
    Json(json!({"status": "ok"}))
}

pub async fn metrics(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    match &state.metrics_handle {
        Some(handle) => {
            let body = handle.render();
            (
                axum::http::StatusCode::OK,
                [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
                body,
            )
        }
        None => (
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            [("content-type", "text/plain; version=0.0.4; charset=utf-8")],
            "metrics not available".to_string(),
        ),
    }
}

#[tracing::instrument(skip(state, req), fields(task = req.task.as_deref().unwrap_or("default")))]
pub async fn generate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<impl IntoResponse, ServeError> {
    let stream = req.stream.unwrap_or(false);

    // Resolve routing: explicit provider override or router decision.
    let (decision, routing_info) = if let Some(ref provider_name) = req.provider {
        // Explicit provider — create a simple decision with no fallbacks.
        let decision = RoutingDecision {
            provider_name: provider_name.clone(),
            reason: "explicit provider override".to_string(),
            fallbacks: vec![],
        };
        (decision, None)
    } else {
        let decision = state.router.route(&req).await?;
        let info = RoutingInfo {
            matched_rule: decision.reason.clone(),
            provider_name: decision.provider_name.clone(),
        };
        (decision, Some(info))
    };

    if stream {
        let chunk_stream = state.executor.execute_stream(&decision, &req).await?;

        let sse_stream = chunk_stream.map(|result| match result {
            Ok(chunk) => Ok::<_, Infallible>(
                Event::default()
                    .json_data(&chunk)
                    .unwrap_or_else(|_| Event::default().data("")),
            ),
            Err(e) => Ok(Event::default()
                .event("error")
                .data(e.to_string())),
        });

        Ok(Sse::new(sse_stream)
            .keep_alive(
                axum::response::sse::KeepAlive::new()
                    .interval(Duration::from_secs(15))
                    .text("keep-alive"),
            )
            .into_response())
    } else {
        let mut resp = state.executor.execute(&decision, &req).await?;
        resp.routing = routing_info;

        // Record eval if eval store is available.
        if let Some(ref eval_store) = state.eval_store {
            let record = EvalRecord {
                id: resp.id.clone(),
                request: req.clone(),
                response: resp.clone(),
                provider: resp.provider.clone(),
                task: req.task.clone(),
                latency_ms: resp.latency_ms,
                score: None,
                created_at: chrono::Utc::now(),
            };
            eval_store.record(record).await;
        }

        Ok(Json(resp).into_response())
    }
}

pub async fn evaluate(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EvaluateRequest>,
) -> Result<impl IntoResponse, ServeError> {
    let eval_store = state.eval_store.as_ref().ok_or_else(|| {
        ServeError::Internal("evaluation system is not enabled".to_string())
    })?;

    if !(0.0..=1.0).contains(&req.score) {
        return Err(ServeError::Routing(format!(
            "score must be between 0.0 and 1.0, got {}",
            req.score
        )));
    }

    eval_store.set_score(&req.id, req.score).await?;

    Ok(Json(json!({"status": "ok"})))
}

pub async fn eval_stats(
    State(state): State<Arc<AppState>>,
) -> Result<impl IntoResponse, ServeError> {
    let eval_store = state.eval_store.as_ref().ok_or_else(|| {
        ServeError::Internal("evaluation system is not enabled".to_string())
    })?;

    let stats = eval_store.stats().await;
    Ok(Json(json!({"stats": stats})))
}

pub async fn eval_best(
    State(state): State<Arc<AppState>>,
    Query(query): Query<EvalBestQuery>,
) -> Result<impl IntoResponse, ServeError> {
    let eval_store = state.eval_store.as_ref().ok_or_else(|| {
        ServeError::Internal("evaluation system is not enabled".to_string())
    })?;

    let best = eval_store.best_provider_for_task(&query.task).await;
    Ok(Json(json!({"task": query.task, "best_provider": best})))
}

pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<impl IntoResponse, ServeError> {
    let provider = state
        .embedding_provider
        .as_ref()
        .ok_or_else(|| ServeError::NotSupported("embedding endpoint not configured".into()))?;

    let resp = provider.embed(&req).await?;
    Ok(Json(resp).into_response())
}

/// Anthropic Messages API compatible endpoint (`POST /v1/messages`).
///
/// Accepts requests in the Anthropic Messages format, translates them to the
/// internal GenerateRequest, and returns responses in the Anthropic format.
/// Supports both streaming (SSE) and non-streaming modes.
pub async fn messages(
    State(state): State<Arc<AppState>>,
    Json(req): Json<AnthropicRequest>,
) -> Result<Response, ServeError> {
    let stream = req.stream.unwrap_or(false);
    let (gen_req, chat_messages) = req.into_generate_request();
    let input_tokens = count_input_tokens(&chat_messages);

    // Route the request through the same routing logic as /v1/generate.
    let decision = state.router.route(&gen_req).await?;

    if stream {
        let chunk_stream = state.executor.execute_stream(&decision, &gen_req).await?;

        // Build the SSE byte stream matching proxy.py format exactly:
        //   event: <name>\n
        //   <json>\n
        //   \n
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<Bytes, Infallible>>(32);

        tokio::spawn(async move {
            // Helper to send an SSE event in proxy.py format.
            macro_rules! send_event {
                ($event:expr, $data:expr) => {
                    let mut buf = Vec::new();
                    buf.extend_from_slice(b"event: ");
                    buf.extend_from_slice($event.as_bytes());
                    buf.push(b'\n');
                    buf.extend_from_slice(
                        serde_json::to_string(&$data)
                            .unwrap_or_default()
                            .as_bytes(),
                    );
                    buf.extend_from_slice(b"\n\n");
                    if tx.send(Ok(Bytes::from(buf))).await.is_err() {
                        return;
                    }
                };
            }

            // 1. message_start
            send_event!(
                "message_start",
                json!({
                    "type": "message_start",
                    "message": {
                        "id": "msg_local",
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                        "model": "claude-local",
                        "usage": {
                            "input_tokens": input_tokens,
                            "output_tokens": 0
                        }
                    }
                })
            );

            // 2. content_block_start
            send_event!(
                "content_block_start",
                json!({
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": "text", "text": ""}
                })
            );

            // 3. Stream content_block_delta events from the provider.
            let mut output_tokens: u32 = 0;
            let mut chunk_stream = std::pin::pin!(chunk_stream);
            while let Some(result) = chunk_stream.next().await {
                match result {
                    Ok(chunk) => {
                        if !chunk.delta.is_empty() {
                            output_tokens += 1;
                            send_event!(
                                "content_block_delta",
                                json!({
                                    "type": "content_block_delta",
                                    "index": 0,
                                    "delta": {
                                        "type": "text_delta",
                                        "text": chunk.delta
                                    }
                                })
                            );
                        }
                    }
                    Err(_) => break,
                }
            }

            // 4. content_block_stop
            send_event!(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": 0})
            );

            // 5. message_delta
            send_event!(
                "message_delta",
                json!({
                    "type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": output_tokens}
                })
            );

            // 6. message_stop
            send_event!("message_stop", json!({"type": "message_stop"}));
        });

        let body = Body::from_stream(ReceiverStream::new(rx));

        Ok(Response::builder()
            .status(StatusCode::OK)
            .header("content-type", "text/event-stream")
            .header("cache-control", "no-cache")
            .body(body)
            .unwrap())
    } else {
        // Non-streaming path.
        let resp = state.executor.execute(&decision, &gen_req).await?;
        let output_tokens = resp.output.split_whitespace().count() as u32;

        let anthropic_resp = AnthropicResponse {
            id: "msg_local".to_string(),
            type_: "message".to_string(),
            role: "assistant".to_string(),
            content: vec![ContentBlock {
                type_: "text".to_string(),
                text: resp.output,
            }],
            model: "claude-local".to_string(),
            stop_reason: "end_turn".to_string(),
            usage: AnthropicUsage {
                input_tokens,
                output_tokens,
            },
        };

        Ok(Json(anthropic_resp).into_response())
    }
}
