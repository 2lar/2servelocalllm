pub mod middleware;
pub mod routes;
pub mod types;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use metrics_exporter_prometheus::PrometheusHandle;
use tower_http::cors::CorsLayer;

use crate::cache::Cache;
use crate::config::AppConfig;
use crate::executor::Executor;
use crate::provider::registry::ProviderRegistry;
use crate::router::Router as LlmRouter;

pub struct AppState {
    pub config: Arc<AppConfig>,
    pub provider_registry: Arc<ProviderRegistry>,
    pub router: Arc<dyn LlmRouter>,
    pub executor: Arc<Executor>,
    pub metrics_handle: Option<PrometheusHandle>,
    pub cache: Option<Arc<dyn Cache>>,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/generate", post(routes::generate))
        .route("/metrics", get(routes::metrics))
        .layer(axum::middleware::from_fn(middleware::request_tracing))
        .layer(CorsLayer::permissive())
        .with_state(state)
}
