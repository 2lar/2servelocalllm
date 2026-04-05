use async_trait::async_trait;

use crate::api::types::GenerateRequest;
use crate::config::RoutingRule;
use crate::error::ServeError;

use super::{Router, RoutingDecision};

pub struct RuleBasedRouter {
    rules: Vec<RoutingRule>,
}

impl RuleBasedRouter {
    pub fn new(rules: Vec<RoutingRule>) -> Self {
        Self { rules }
    }

    fn prompt_text(req: &GenerateRequest) -> String {
        if let Some(prompt) = &req.prompt {
            return prompt.clone();
        }
        if let Some(messages) = &req.messages {
            return messages
                .iter()
                .map(|m| m.content.as_str())
                .collect::<Vec<_>>()
                .join(" ");
        }
        String::new()
    }

    fn matches(rule: &RoutingRule, req: &GenerateRequest) -> bool {
        // A rule with no conditions is a catch-all (always matches).
        let has_conditions = rule.task.is_some()
            || rule.max_prompt_length.is_some()
            || rule.keywords.is_some();

        if !has_conditions {
            return true;
        }

        // Task: exact match against request's task field.
        if let Some(rule_task) = &rule.task {
            if let Some(req_task) = &req.task {
                if req_task == rule_task {
                    return true;
                }
            }
            // If the rule requires a task but request has no task, this rule doesn't match.
            // Fall through to check other rules.
        }

        // Max prompt length: match if prompt char count is under the threshold.
        if let Some(max_len) = rule.max_prompt_length {
            let prompt = Self::prompt_text(req);
            if prompt.len() < max_len {
                return true;
            }
        }

        // Keywords: match if prompt contains any keyword (case-insensitive).
        if let Some(keywords) = &rule.keywords {
            let prompt = Self::prompt_text(req).to_lowercase();
            if keywords.iter().any(|kw| prompt.contains(&kw.to_lowercase())) {
                return true;
            }
        }

        false
    }
}

#[async_trait]
impl Router for RuleBasedRouter {
    async fn route(&self, req: &GenerateRequest) -> Result<RoutingDecision, ServeError> {
        for rule in &self.rules {
            if Self::matches(rule, req) {
                return Ok(RoutingDecision {
                    provider_name: rule.provider.clone(),
                    reason: format!("matched rule '{}'", rule.name),
                    fallbacks: rule.fallbacks.clone().unwrap_or_default(),
                });
            }
        }

        Err(ServeError::Routing(
            "no routing rule matched the request".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{ChatMessage, GenerateRequest};
    use crate::config::RoutingRule;

    fn base_request() -> GenerateRequest {
        GenerateRequest {
            prompt: Some("Hello world".to_string()),
            messages: None,
            task: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            provider: None,
        }
    }

    fn rule(name: &str, provider: &str) -> RoutingRule {
        RoutingRule {
            name: name.to_string(),
            task: None,
            max_prompt_length: None,
            keywords: None,
            provider: provider.to_string(),
            fallbacks: None,
        }
    }

    #[tokio::test]
    async fn route_matches_task() {
        let rules = vec![
            RoutingRule {
                task: Some("code".to_string()),
                ..rule("code-tasks", "code-provider")
            },
            rule("default", "default-provider"),
        ];
        let router = RuleBasedRouter::new(rules);

        let req = GenerateRequest {
            task: Some("code".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "code-provider");
        assert!(decision.reason.contains("code-tasks"));
    }

    #[tokio::test]
    async fn route_task_mismatch_falls_through() {
        let rules = vec![
            RoutingRule {
                task: Some("code".to_string()),
                ..rule("code-tasks", "code-provider")
            },
            rule("default", "default-provider"),
        ];
        let router = RuleBasedRouter::new(rules);

        let req = GenerateRequest {
            task: Some("chat".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "default-provider");
    }

    #[tokio::test]
    async fn route_matches_prompt_length() {
        let rules = vec![
            RoutingRule {
                max_prompt_length: Some(20),
                ..rule("short-prompts", "fast-provider")
            },
            rule("default", "default-provider"),
        ];
        let router = RuleBasedRouter::new(rules);

        let req = GenerateRequest {
            prompt: Some("Short".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "fast-provider");
    }

    #[tokio::test]
    async fn route_long_prompt_skips_length_rule() {
        let rules = vec![
            RoutingRule {
                max_prompt_length: Some(5),
                ..rule("short-prompts", "fast-provider")
            },
            rule("default", "default-provider"),
        ];
        let router = RuleBasedRouter::new(rules);

        let req = GenerateRequest {
            prompt: Some("This is a very long prompt".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "default-provider");
    }

    #[tokio::test]
    async fn route_matches_keywords_case_insensitive() {
        let rules = vec![
            RoutingRule {
                keywords: Some(vec!["rust".to_string(), "python".to_string()]),
                ..rule("programming", "code-provider")
            },
            rule("default", "default-provider"),
        ];
        let router = RuleBasedRouter::new(rules);

        let req = GenerateRequest {
            prompt: Some("Help me write RUST code".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "code-provider");
    }

    #[tokio::test]
    async fn route_keywords_no_match_falls_through() {
        let rules = vec![
            RoutingRule {
                keywords: Some(vec!["rust".to_string()]),
                ..rule("programming", "code-provider")
            },
            rule("default", "default-provider"),
        ];
        let router = RuleBasedRouter::new(rules);

        let req = GenerateRequest {
            prompt: Some("Tell me about cooking".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "default-provider");
    }

    #[tokio::test]
    async fn route_default_catchall() {
        let rules = vec![rule("default", "fallback-provider")];
        let router = RuleBasedRouter::new(rules);

        let decision = router.route(&base_request()).await.unwrap();
        assert_eq!(decision.provider_name, "fallback-provider");
        assert!(decision.reason.contains("default"));
    }

    #[tokio::test]
    async fn route_no_rules_returns_error() {
        let router = RuleBasedRouter::new(vec![]);
        let result = router.route(&base_request()).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn route_first_match_wins() {
        let rules = vec![
            rule("first", "provider-a"),
            rule("second", "provider-b"),
        ];
        let router = RuleBasedRouter::new(rules);

        let decision = router.route(&base_request()).await.unwrap();
        assert_eq!(decision.provider_name, "provider-a");
    }

    #[tokio::test]
    async fn route_includes_fallbacks() {
        let rules = vec![RoutingRule {
            fallbacks: Some(vec!["backup-1".to_string(), "backup-2".to_string()]),
            ..rule("default", "primary")
        }];
        let router = RuleBasedRouter::new(rules);

        let decision = router.route(&base_request()).await.unwrap();
        assert_eq!(decision.fallbacks, vec!["backup-1", "backup-2"]);
    }

    #[tokio::test]
    async fn route_prompt_length_uses_messages_when_no_prompt() {
        let rules = vec![
            RoutingRule {
                max_prompt_length: Some(50),
                ..rule("short", "fast-provider")
            },
            rule("default", "default-provider"),
        ];
        let router = RuleBasedRouter::new(rules);

        let req = GenerateRequest {
            prompt: None,
            messages: Some(vec![ChatMessage {
                role: "user".to_string(),
                content: "Hi".to_string(),
            }]),
            task: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            provider: None,
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "fast-provider");
    }
}
