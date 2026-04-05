use std::time::{Duration, Instant};

use async_trait::async_trait;
use futures::stream;
use futures::StreamExt;
use reqwest::Client;
use serde_json::json;
use uuid::Uuid;

use crate::api::types::{GenerateRequest, GenerateResponse, StreamChunk, Usage};
use crate::config::ProviderConfig;
use crate::error::ServeError;

use super::{ChunkStream, Provider};

pub struct LocalProvider {
    config: ProviderConfig,
    client: Client,
}

impl LocalProvider {
    pub fn new(config: ProviderConfig) -> Result<Self, ServeError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(config.timeout_secs))
            .build()
            .map_err(|e| ServeError::Internal(format!("failed to build HTTP client: {e}")))?;

        Ok(Self { config, client })
    }

    fn build_messages(req: &GenerateRequest) -> Vec<serde_json::Value> {
        if let Some(messages) = &req.messages {
            messages
                .iter()
                .map(|m| json!({"role": m.role, "content": m.content}))
                .collect()
        } else if let Some(prompt) = &req.prompt {
            vec![json!({"role": "user", "content": prompt})]
        } else {
            vec![]
        }
    }

    fn build_openai_body(req: &GenerateRequest, model: &str, stream: bool) -> serde_json::Value {
        let messages = Self::build_messages(req);
        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": stream,
        });

        if let Some(max_tokens) = req.max_tokens {
            body["max_tokens"] = json!(max_tokens);
        }
        if let Some(temperature) = req.temperature {
            body["temperature"] = json!(temperature);
        }

        body
    }
}

#[async_trait]
impl Provider for LocalProvider {
    fn name(&self) -> &str {
        &self.config.name
    }

    async fn generate(&self, req: &GenerateRequest) -> Result<GenerateResponse, ServeError> {
        let start = Instant::now();
        let url = format!("{}/v1/chat/completions", self.config.url);
        let body = Self::build_openai_body(req, &self.config.model, false);

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| ServeError::Provider {
                provider: self.config.name.clone(),
                source: e,
            })?;

        let data: serde_json::Value =
            resp.json().await.map_err(|e| ServeError::Provider {
                provider: self.config.name.clone(),
                source: e,
            })?;

        let output = data["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("")
            .to_string();

        let usage = &data["usage"];
        let input_tokens = usage["prompt_tokens"].as_u64().unwrap_or(0) as u32;
        let output_tokens = usage["completion_tokens"].as_u64().unwrap_or(0) as u32;

        Ok(GenerateResponse {
            id: Uuid::new_v4().to_string(),
            output,
            model: self.config.model.clone(),
            provider: self.config.name.clone(),
            latency_ms: start.elapsed().as_millis() as u64,
            usage: Usage {
                input_tokens,
                output_tokens,
            },
            cached: false,
            routing: None,
        })
    }

    async fn generate_stream(&self, req: &GenerateRequest) -> Result<ChunkStream, ServeError> {
        let url = format!("{}/v1/chat/completions", self.config.url);
        let body = Self::build_openai_body(req, &self.config.model, true);

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .await
            .map_err(|e| ServeError::Provider {
                provider: self.config.name.clone(),
                source: e,
            })?;

        let provider_name = self.config.name.clone();
        let byte_stream = resp.bytes_stream();

        // Buffer SSE lines from the byte stream, parse OpenAI streaming format.
        // Each SSE line looks like: "data: {json}\n\n" or "data: [DONE]\n\n"
        let stream = byte_stream
            .map(move |chunk_result| {
                let provider_name = provider_name.clone();
                match chunk_result {
                    Ok(bytes) => {
                        let text = String::from_utf8_lossy(&bytes);
                        let mut chunks = Vec::new();

                        for line in text.lines() {
                            let line = line.trim();
                            if !line.starts_with("data: ") {
                                continue;
                            }
                            let data = &line[6..];
                            if data == "[DONE]" {
                                chunks.push(Ok(StreamChunk {
                                    delta: String::new(),
                                    done: true,
                                }));
                                break;
                            }
                            if let Ok(parsed) = serde_json::from_str::<serde_json::Value>(data) {
                                let delta_content = parsed["choices"][0]["delta"]["content"]
                                    .as_str()
                                    .unwrap_or("");
                                if !delta_content.is_empty() {
                                    chunks.push(Ok(StreamChunk {
                                        delta: delta_content.to_string(),
                                        done: false,
                                    }));
                                }
                            }
                        }

                        stream::iter(chunks)
                    }
                    Err(e) => stream::iter(vec![Err(ServeError::Provider {
                        provider: provider_name,
                        source: e,
                    })]),
                }
            })
            .flatten();

        Ok(Box::pin(stream))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::ChatMessage;

    fn test_config(url: &str) -> ProviderConfig {
        ProviderConfig {
            name: "test-local".to_string(),
            url: url.to_string(),
            model: "test-model".to_string(),
            timeout_secs: 10,
        }
    }

    fn test_request() -> GenerateRequest {
        GenerateRequest {
            prompt: Some("Hello".to_string()),
            messages: None,
            task: None,
            max_tokens: Some(100),
            temperature: Some(0.7),
            stream: None,
            provider: None,
        }
    }

    #[test]
    fn build_messages_from_prompt() {
        let req = test_request();
        let messages = LocalProvider::build_messages(&req);
        assert_eq!(messages.len(), 1);
        assert_eq!(messages[0]["role"], "user");
        assert_eq!(messages[0]["content"], "Hello");
    }

    #[test]
    fn build_messages_from_chat_messages() {
        let req = GenerateRequest {
            prompt: None,
            messages: Some(vec![
                ChatMessage {
                    role: "system".to_string(),
                    content: "You are helpful.".to_string(),
                },
                ChatMessage {
                    role: "user".to_string(),
                    content: "Hi".to_string(),
                },
            ]),
            task: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            provider: None,
        };
        let messages = LocalProvider::build_messages(&req);
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[1]["content"], "Hi");
    }

    #[test]
    fn build_openai_body_includes_params() {
        let req = test_request();
        let body = LocalProvider::build_openai_body(&req, "test-model", false);
        assert_eq!(body["model"], "test-model");
        assert_eq!(body["stream"], false);
        assert_eq!(body["max_tokens"], 100);
        let temp = body["temperature"].as_f64().unwrap();
        assert!((temp - 0.7).abs() < 0.001, "temperature was {temp}");
    }

    #[test]
    fn build_openai_body_omits_none_params() {
        let req = GenerateRequest {
            prompt: Some("Hello".to_string()),
            messages: None,
            task: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            provider: None,
        };
        let body = LocalProvider::build_openai_body(&req, "test-model", false);
        assert!(body.get("max_tokens").is_none());
        assert!(body.get("temperature").is_none());
    }

    #[tokio::test]
    async fn generate_against_wiremock() {
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        let response_body = json!({
            "id": "chatcmpl-123",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from mock!"},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": 5,
                "completion_tokens": 3,
                "total_tokens": 8
            }
        });

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(&response_body))
            .mount(&mock_server)
            .await;

        let config = test_config(&mock_server.uri());
        let provider = LocalProvider::new(config).unwrap();
        let req = test_request();

        let resp = provider.generate(&req).await.unwrap();
        assert_eq!(resp.output, "Hello from mock!");
        assert_eq!(resp.usage.input_tokens, 5);
        assert_eq!(resp.usage.output_tokens, 3);
        assert_eq!(resp.provider, "test-local");
        assert!(!resp.cached);
    }

    #[tokio::test]
    async fn generate_stream_against_wiremock() {
        use futures::StreamExt;
        use wiremock::matchers::{method, path};
        use wiremock::{Mock, MockServer, ResponseTemplate};

        let mock_server = MockServer::start().await;

        // Simulate OpenAI SSE streaming format
        let sse_body = [
            "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"},\"index\":0}]}\n\n",
            "data: {\"choices\":[{\"delta\":{\"content\":\" world\"},\"index\":0}]}\n\n",
            "data: [DONE]\n\n",
        ]
        .join("");

        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(
                ResponseTemplate::new(200)
                    .set_body_string(&sse_body)
                    .append_header("content-type", "text/event-stream"),
            )
            .mount(&mock_server)
            .await;

        let config = test_config(&mock_server.uri());
        let provider = LocalProvider::new(config).unwrap();
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
        let mut saw_done = false;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.unwrap();
            if chunk.done {
                saw_done = true;
            } else {
                collected.push_str(&chunk.delta);
            }
        }

        assert_eq!(collected, "Hello world");
        assert!(saw_done);
    }
}
