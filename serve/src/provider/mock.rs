use std::time::Duration;

use async_trait::async_trait;
use futures::stream;
use uuid::Uuid;

use crate::api::types::{GenerateRequest, GenerateResponse, StreamChunk, Usage};
use crate::error::ServeError;

use super::{ChunkStream, Provider};

pub struct MockProvider {
    delay: Duration,
}

impl MockProvider {
    pub fn new(delay: Duration) -> Self {
        Self { delay }
    }
}

#[async_trait]
impl Provider for MockProvider {
    fn name(&self) -> &str {
        "mock"
    }

    async fn generate(&self, _req: &GenerateRequest) -> Result<GenerateResponse, ServeError> {
        tokio::time::sleep(self.delay).await;

        Ok(GenerateResponse {
            id: Uuid::new_v4().to_string(),
            output: "This is a mock response.".to_string(),
            model: "mock-model".to_string(),
            provider: "mock".to_string(),
            latency_ms: self.delay.as_millis() as u64,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 6,
            },
            cached: false,
            routing: None,
        })
    }

    async fn generate_stream(&self, _req: &GenerateRequest) -> Result<ChunkStream, ServeError> {
        let delay = self.delay;
        let chunks = vec!["This ", "is ", "a ", "mock ", "response."];

        let stream = stream::iter(chunks.into_iter().enumerate().map(move |(i, text)| {
            Ok(StreamChunk {
                delta: text.to_string(),
                done: i == 4,
            })
        }));

        // Apply delay once upfront, then stream chunks immediately
        tokio::time::sleep(delay).await;

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn mock_generate_returns_canned_response() {
        let provider = MockProvider::new(Duration::from_millis(0));
        let req = GenerateRequest {
            prompt: Some("Hello".to_string()),
            messages: None,
            task: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            provider: None,
        };

        let resp = provider.generate(&req).await.unwrap();
        assert_eq!(resp.output, "This is a mock response.");
        assert_eq!(resp.provider, "mock");
        assert_eq!(resp.model, "mock-model");
        assert!(!resp.cached);
    }

    #[tokio::test]
    async fn mock_generate_stream_yields_chunks() {
        use futures::StreamExt;

        let provider = MockProvider::new(Duration::from_millis(0));
        let req = GenerateRequest {
            prompt: Some("Hello".to_string()),
            messages: None,
            task: None,
            max_tokens: None,
            temperature: None,
            stream: Some(true),
            provider: None,
        };

        let mut stream = provider.generate_stream(&req).await.unwrap();
        let mut collected = String::new();
        let mut last_done = false;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            collected.push_str(&chunk.delta);
            last_done = chunk.done;
        }

        assert_eq!(collected, "This is a mock response.");
        assert!(last_done);
    }

    #[tokio::test]
    async fn mock_name_returns_mock() {
        let provider = MockProvider::new(Duration::from_millis(0));
        assert_eq!(provider.name(), "mock");
    }
}
