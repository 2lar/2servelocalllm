use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateRequest {
    pub prompt: Option<String>,
    pub messages: Option<Vec<ChatMessage>>,
    pub task: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
    pub provider: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateResponse {
    pub id: String,
    pub output: String,
    pub model: String,
    pub provider: String,
    pub latency_ms: u64,
    pub usage: Usage,
    pub cached: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingInfo {
    pub matched_rule: String,
    pub provider_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Usage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamChunk {
    pub delta: String,
    pub done: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvaluateRequest {
    pub id: String,
    pub score: f64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct EvalBestQuery {
    pub task: String,
}

/// OpenAI-compatible embedding request.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedRequest {
    pub input: EmbedInput,
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub encoding_format: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum EmbedInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbedInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbedInput::Single(s) => vec![s],
            EmbedInput::Batch(v) => v,
        }
    }
}

/// OpenAI-compatible embedding response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedResponse {
    pub object: String,
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbedUsage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub latency_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingData {
    pub object: String,
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}
