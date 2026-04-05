pub mod rule_based;

use async_trait::async_trait;

use crate::api::types::GenerateRequest;
use crate::error::ServeError;

#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub provider_name: String,
    pub reason: String,
    pub fallbacks: Vec<String>,
}

#[async_trait]
pub trait Router: Send + Sync {
    async fn route(&self, req: &GenerateRequest) -> Result<RoutingDecision, ServeError>;
}
