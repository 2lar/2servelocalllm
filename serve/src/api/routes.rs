use std::convert::Infallible;
use std::sync::Arc;
use std::time::Duration;

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::response::IntoResponse;
use axum::Json;
use futures::StreamExt;
use serde_json::json;

use crate::api::types::{GenerateRequest, RoutingInfo};
use crate::error::ServeError;
use crate::router::RoutingDecision;

use super::AppState;

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
        Ok(Json(resp).into_response())
    }
}
