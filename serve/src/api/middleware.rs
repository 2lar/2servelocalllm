use axum::extract::Request;
use axum::http::HeaderValue;
use axum::middleware::Next;
use axum::response::Response;
use uuid::Uuid;

/// Middleware that assigns a request ID, creates a tracing span, and logs
/// request start/completion with latency.
pub async fn request_tracing(mut req: Request, next: Next) -> Response {
    let request_id = req
        .headers()
        .get("x-request-id")
        .and_then(|v| v.to_str().ok())
        .map(String::from)
        .unwrap_or_else(|| Uuid::new_v4().to_string());

    let method = req.method().clone();
    let path = req.uri().path().to_string();

    // Attach the request ID to the incoming request headers.
    if let Ok(val) = HeaderValue::from_str(&request_id) {
        req.headers_mut().insert("x-request-id", val);
    }

    let span = tracing::info_span!(
        "request",
        request_id = %request_id,
        method = %method,
        path = %path,
    );

    tracing::info!(parent: &span, "request started");

    let start = std::time::Instant::now();
    let response = {
        let fut = next.run(req);
        tracing::Instrument::instrument(fut, span.clone()).await
    };

    let latency = start.elapsed();
    let status = response.status().as_u16();

    tracing::info!(parent: &span, status, latency_ms = latency.as_millis() as u64, "request completed");

    // Attach request ID to response headers.
    let mut response = response;
    if let Ok(val) = HeaderValue::from_str(&request_id) {
        response.headers_mut().insert("x-request-id", val);
    }

    response
}
