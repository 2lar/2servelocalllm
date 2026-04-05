pub mod memory;
pub mod metrics;

use async_trait::async_trait;
use sha2::{Digest, Sha256};

use crate::api::types::{GenerateRequest, GenerateResponse};

#[async_trait]
pub trait Cache: Send + Sync {
    async fn get(&self, req: &GenerateRequest) -> Option<GenerateResponse>;
    async fn put(&self, req: &GenerateRequest, resp: &GenerateResponse);
}

/// Compute a deterministic cache key from a request.
///
/// Hashes prompt + messages + temperature (rounded to 2 decimals) + max_tokens
/// using SHA-256 to produce a stable hex string.
pub fn cache_key(req: &GenerateRequest) -> String {
    let mut hasher = Sha256::new();

    if let Some(ref prompt) = req.prompt {
        hasher.update(b"prompt:");
        hasher.update(prompt.as_bytes());
    }

    if let Some(ref messages) = req.messages {
        hasher.update(b"messages:");
        for msg in messages {
            hasher.update(msg.role.as_bytes());
            hasher.update(b":");
            hasher.update(msg.content.as_bytes());
            hasher.update(b"|");
        }
    }

    if let Some(temp) = req.temperature {
        // Round to 2 decimal places to avoid float inconsistencies.
        let rounded = (temp * 100.0).round() / 100.0;
        hasher.update(b"temp:");
        hasher.update(rounded.to_bits().to_le_bytes());
    }

    if let Some(max_tokens) = req.max_tokens {
        hasher.update(b"max_tokens:");
        hasher.update(max_tokens.to_le_bytes());
    }

    let result = hasher.finalize();
    hex::encode(result)
}

// We use hex encoding inline since sha2 already depends on it.
// Provide a minimal hex encoder to avoid adding the `hex` crate.
mod hex {
    pub fn encode(bytes: impl AsRef<[u8]>) -> String {
        bytes
            .as_ref()
            .iter()
            .fold(String::new(), |mut acc, b| {
                use std::fmt::Write;
                let _ = write!(acc, "{b:02x}");
                acc
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{ChatMessage, GenerateRequest};

    fn base_request() -> GenerateRequest {
        GenerateRequest {
            prompt: Some("Hello world".to_string()),
            messages: None,
            task: None,
            max_tokens: Some(100),
            temperature: Some(0.7),
            stream: None,
            provider: None,
        }
    }

    #[test]
    fn same_request_produces_same_key() {
        let req1 = base_request();
        let req2 = base_request();
        assert_eq!(cache_key(&req1), cache_key(&req2));
    }

    #[test]
    fn different_prompt_produces_different_key() {
        let req1 = base_request();
        let mut req2 = base_request();
        req2.prompt = Some("Different prompt".to_string());
        assert_ne!(cache_key(&req1), cache_key(&req2));
    }

    #[test]
    fn different_temperature_produces_different_key() {
        let req1 = base_request();
        let mut req2 = base_request();
        req2.temperature = Some(0.9);
        assert_ne!(cache_key(&req1), cache_key(&req2));
    }

    #[test]
    fn different_max_tokens_produces_different_key() {
        let req1 = base_request();
        let mut req2 = base_request();
        req2.max_tokens = Some(200);
        assert_ne!(cache_key(&req1), cache_key(&req2));
    }

    #[test]
    fn temperature_rounding_produces_same_key() {
        let mut req1 = base_request();
        req1.temperature = Some(0.7000001);
        let mut req2 = base_request();
        req2.temperature = Some(0.6999999);
        assert_eq!(cache_key(&req1), cache_key(&req2));
    }

    #[test]
    fn messages_included_in_key() {
        let mut req1 = base_request();
        req1.messages = Some(vec![ChatMessage {
            role: "user".to_string(),
            content: "Hello".to_string(),
        }]);

        let mut req2 = base_request();
        req2.messages = Some(vec![ChatMessage {
            role: "user".to_string(),
            content: "Different".to_string(),
        }]);

        assert_ne!(cache_key(&req1), cache_key(&req2));
    }

    #[test]
    fn key_is_hex_sha256_length() {
        let req = base_request();
        let key = cache_key(&req);
        // SHA-256 produces 32 bytes = 64 hex chars.
        assert_eq!(key.len(), 64);
    }

    #[test]
    fn task_and_stream_and_provider_do_not_affect_key() {
        let req1 = base_request();
        let mut req2 = base_request();
        req2.task = Some("code".to_string());
        req2.stream = Some(true);
        req2.provider = Some("other".to_string());
        // These fields are not part of the cache key.
        assert_eq!(cache_key(&req1), cache_key(&req2));
    }
}
