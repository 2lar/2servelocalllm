use std::time::{Duration, Instant};

use crate::api::types::{EmbedRequest, EmbedResponse};
use crate::config::EmbeddingConfig;
use crate::error::ServeError;

pub struct EmbeddingProvider {
    url: String,
    model_name: String,
    client: reqwest::Client,
}

impl EmbeddingProvider {
    pub fn new(config: &EmbeddingConfig) -> Result<Self, ServeError> {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(30))
            .build()
            .map_err(|e| ServeError::Internal(format!("failed to build embedding client: {e}")))?;

        // Extract model filename (without path/extension) for the model field in responses.
        let model_name = std::path::Path::new(&config.model)
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| config.model.clone());

        Ok(Self {
            url: format!("http://{}:{}", config.host, config.port),
            model_name,
            client,
        })
    }

    pub async fn embed(&self, req: &EmbedRequest) -> Result<EmbedResponse, ServeError> {
        let start = Instant::now();

        let texts = req.input.clone().into_vec();

        let body = serde_json::json!({
            "input": texts,
            "model": req.model.as_deref().unwrap_or(&self.model_name),
        });

        let resp = self
            .client
            .post(format!("{}/v1/embeddings", self.url))
            .json(&body)
            .send()
            .await
            .map_err(|e| ServeError::Internal(format!("embedding request failed: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ServeError::Internal(format!(
                "embedding server returned {status}: {text}"
            )));
        }

        let mut embed_resp: EmbedResponse = resp.json().await.map_err(|e| {
            ServeError::Internal(format!("failed to parse embedding response: {e}"))
        })?;

        embed_resp.latency_ms = Some(start.elapsed().as_millis() as u64);
        embed_resp.provider = Some("local-embedding".to_string());

        metrics::counter!("embedding_requests_total").increment(1);
        metrics::histogram!("embedding_latency_ms").record(start.elapsed().as_millis() as f64);

        Ok(embed_resp)
    }
}
