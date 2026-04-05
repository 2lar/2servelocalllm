pub mod routes;
pub mod types;

use std::sync::Arc;

use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;

use crate::config::AppConfig;
use crate::executor::Executor;
use crate::provider::registry::ProviderRegistry;
use crate::router::Router as LlmRouter;

pub struct AppState {
    pub config: Arc<AppConfig>,
    pub provider_registry: Arc<ProviderRegistry>,
    pub router: Arc<dyn LlmRouter>,
    pub executor: Arc<Executor>,
}

pub fn build_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/health", get(routes::health))
        .route("/v1/generate", post(routes::generate))
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive())
        .with_state(state)
}
