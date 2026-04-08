use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use serde_json::json;

#[derive(Debug, thiserror::Error)]
pub enum ServeError {
    #[error("provider '{provider}': {source}")]
    Provider {
        provider: String,
        source: reqwest::Error,
    },

    #[error("config error: {0}")]
    Config(String),

    #[error("process manager error: {0}")]
    ProcessManager(String),

    #[error("routing error: {0}")]
    Routing(String),

    #[error("provider not found: {0}")]
    ProviderNotFound(String),

    #[error("request timed out after {timeout_secs}s")]
    Timeout { timeout_secs: u64 },

    #[error("not supported: {0}")]
    NotSupported(String),

    #[error("internal error: {0}")]
    Internal(String),
}

impl ServeError {
    /// Returns true if this error represents a transient failure that may
    /// succeed on retry (provider network errors, timeouts, 5xx responses).
    /// Returns false for client errors (bad request, not found) that would
    /// fail again with the same input.
    pub fn is_retryable(&self) -> bool {
        match self {
            ServeError::Provider { source, .. } => {
                // Retry on connection/timeout errors and 5xx server errors.
                // Don't retry on 4xx client errors.
                if source.is_connect() || source.is_timeout() {
                    return true;
                }
                if let Some(status) = source.status() {
                    return status.is_server_error();
                }
                // Other reqwest errors (decode, redirect, etc.) — not retryable.
                false
            }
            ServeError::Timeout { .. } => true,
            ServeError::Config(_)
            | ServeError::ProcessManager(_)
            | ServeError::Routing(_)
            | ServeError::ProviderNotFound(_)
            | ServeError::NotSupported(_)
            | ServeError::Internal(_) => false,
        }
    }
}

impl IntoResponse for ServeError {
    fn into_response(self) -> Response {
        let (status, message) = match &self {
            ServeError::Provider { .. } => (StatusCode::BAD_GATEWAY, self.to_string()),
            ServeError::Config(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
            ServeError::Routing(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ServeError::ProviderNotFound(_) => (StatusCode::BAD_REQUEST, self.to_string()),
            ServeError::ProcessManager(_) => (StatusCode::SERVICE_UNAVAILABLE, self.to_string()),
            ServeError::Timeout { .. } => (StatusCode::GATEWAY_TIMEOUT, self.to_string()),
            ServeError::NotSupported(_) => (StatusCode::NOT_IMPLEMENTED, self.to_string()),
            ServeError::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, self.to_string()),
        };

        let body = json!({
            "error": {
                "type": status.canonical_reason().unwrap_or("Unknown"),
                "message": message,
            }
        });

        (status, axum::Json(body)).into_response()
    }
}
