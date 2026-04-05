pub mod store;

use serde::{Deserialize, Serialize};

use crate::api::types::{GenerateRequest, GenerateResponse};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalRecord {
    pub id: String,
    pub request: GenerateRequest,
    pub response: GenerateResponse,
    pub provider: String,
    pub task: Option<String>,
    pub latency_ms: u64,
    pub score: Option<f64>,
    pub created_at: chrono::DateTime<chrono::Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalStats {
    pub provider: String,
    pub task: Option<String>,
    pub avg_score: Option<f64>,
    pub avg_latency_ms: f64,
    pub request_count: u64,
}
