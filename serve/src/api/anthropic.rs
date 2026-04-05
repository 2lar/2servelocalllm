use serde::{Deserialize, Serialize};

use crate::api::types::{ChatMessage, GenerateRequest};

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicRequest {
    pub model: Option<String>,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub stream: Option<bool>,
    /// Can be a plain string or a list of `{"text": "..."}` blocks.
    pub system: Option<serde_json::Value>,
    #[serde(default)]
    pub messages: Vec<AnthropicMessage>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnthropicMessage {
    pub role: String,
    /// Can be a plain string or a list of content blocks.
    pub content: serde_json::Value,
}

// ---------------------------------------------------------------------------
// Response types (non-streaming)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicResponse {
    pub id: String,
    #[serde(rename = "type")]
    pub type_: String,
    pub role: String,
    pub content: Vec<ContentBlock>,
    pub model: String,
    pub stop_reason: String,
    pub usage: AnthropicUsage,
}

#[derive(Debug, Clone, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub type_: String,
    pub text: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnthropicUsage {
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// ---------------------------------------------------------------------------
// Conversion: AnthropicRequest -> GenerateRequest
// ---------------------------------------------------------------------------

/// Extract the system prompt text from the polymorphic `system` field.
/// - If it's a string, return it directly.
/// - If it's an array of `{"text": "..."}` blocks, join them with spaces.
fn extract_system_text(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|block| block.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Extract the message content text from the polymorphic `content` field.
/// - If it's a string, return it directly.
/// - If it's an array of content blocks, join text blocks with spaces.
fn extract_message_content(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|block| {
                let is_text = block
                    .get("type")
                    .and_then(|t| t.as_str())
                    .map_or(false, |t| t == "text");
                if is_text {
                    block.get("text").and_then(|t| t.as_str()).map(String::from)
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join(" "),
        _ => String::new(),
    }
}

/// Count input tokens using word count (matching proxy.py behavior).
pub fn count_input_tokens(messages: &[ChatMessage]) -> u32 {
    messages
        .iter()
        .map(|m| m.content.split_whitespace().count() as u32)
        .sum()
}

impl AnthropicRequest {
    /// Convert to the internal GenerateRequest, returning both the request
    /// and the flat list of ChatMessages (needed for token counting).
    pub fn into_generate_request(self) -> (GenerateRequest, Vec<ChatMessage>) {
        let mut chat_messages = Vec::new();

        // Prepend system message if present.
        if let Some(ref system) = self.system {
            let text = extract_system_text(system);
            if !text.is_empty() {
                chat_messages.push(ChatMessage {
                    role: "system".to_string(),
                    content: text,
                });
            }
        }

        // Convert each Anthropic message.
        for msg in &self.messages {
            let content = extract_message_content(&msg.content);
            chat_messages.push(ChatMessage {
                role: msg.role.clone(),
                content,
            });
        }

        let generate_req = GenerateRequest {
            prompt: None,
            messages: Some(chat_messages.clone()),
            task: None,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream,
            provider: None,
        };

        (generate_req, chat_messages)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn convert_string_system_and_string_content() {
        let req = AnthropicRequest {
            model: Some("claude-sonnet-4-20250514".to_string()),
            max_tokens: Some(4096),
            temperature: Some(0.6),
            stream: Some(false),
            system: Some(json!("You are a helpful assistant.")),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: json!("Hello world"),
            }],
        };

        let (gen_req, chat_msgs) = req.into_generate_request();

        assert_eq!(chat_msgs.len(), 2);
        assert_eq!(chat_msgs[0].role, "system");
        assert_eq!(chat_msgs[0].content, "You are a helpful assistant.");
        assert_eq!(chat_msgs[1].role, "user");
        assert_eq!(chat_msgs[1].content, "Hello world");

        assert!(gen_req.prompt.is_none());
        assert_eq!(gen_req.messages.as_ref().unwrap().len(), 2);
        assert_eq!(gen_req.max_tokens, Some(4096));
        assert_eq!(gen_req.temperature, Some(0.6));
        assert_eq!(gen_req.stream, Some(false));
    }

    #[test]
    fn convert_array_system_and_array_content() {
        let req = AnthropicRequest {
            model: Some("claude-sonnet-4-20250514".to_string()),
            max_tokens: Some(1024),
            temperature: None,
            stream: Some(true),
            system: Some(json!([
                {"text": "You are a helpful assistant."},
                {"text": "Be concise."}
            ])),
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: json!([
                    {"type": "text", "text": "Hello"},
                    {"type": "text", "text": "world"}
                ]),
            }],
        };

        let (gen_req, chat_msgs) = req.into_generate_request();

        assert_eq!(chat_msgs.len(), 2);
        assert_eq!(chat_msgs[0].role, "system");
        assert_eq!(
            chat_msgs[0].content,
            "You are a helpful assistant. Be concise."
        );
        assert_eq!(chat_msgs[1].role, "user");
        assert_eq!(chat_msgs[1].content, "Hello world");

        assert_eq!(gen_req.stream, Some(true));
        assert_eq!(gen_req.max_tokens, Some(1024));
    }

    #[test]
    fn input_token_count_uses_word_count() {
        let messages = vec![
            ChatMessage {
                role: "system".to_string(),
                content: "You are a helpful assistant.".to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: "Hello world".to_string(),
            },
        ];

        // "You are a helpful assistant." = 5 words, "Hello world" = 2 words
        assert_eq!(count_input_tokens(&messages), 7);
    }

    #[test]
    fn no_system_message_when_system_is_none() {
        let req = AnthropicRequest {
            model: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            system: None,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: json!("Hello"),
            }],
        };

        let (_, chat_msgs) = req.into_generate_request();
        assert_eq!(chat_msgs.len(), 1);
        assert_eq!(chat_msgs[0].role, "user");
    }

    #[test]
    fn non_text_content_blocks_are_skipped() {
        let req = AnthropicRequest {
            model: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            system: None,
            messages: vec![AnthropicMessage {
                role: "user".to_string(),
                content: json!([
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "source": {"data": "base64..."}},
                    {"type": "text", "text": "world"}
                ]),
            }],
        };

        let (_, chat_msgs) = req.into_generate_request();
        assert_eq!(chat_msgs[0].content, "Hello world");
    }
}
