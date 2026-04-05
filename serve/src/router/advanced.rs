use std::sync::atomic::{AtomicUsize, Ordering};

use async_trait::async_trait;
use regex::Regex;

use crate::api::types::GenerateRequest;
use crate::config::AdvancedRoutingConfig;
use crate::error::ServeError;

use super::{Router, RoutingDecision};

struct CompiledKeywordRule {
    regex: Regex,
    provider: String,
    fallbacks: Vec<String>,
}

struct CompiledLengthRule {
    max_chars: usize,
    provider: String,
    fallbacks: Vec<String>,
}

pub struct AdvancedRouter {
    default_provider: String,
    length_rules: Vec<CompiledLengthRule>,
    keyword_rules: Vec<CompiledKeywordRule>,
    load_balance_providers: Vec<String>,
    lb_counter: AtomicUsize,
}

impl AdvancedRouter {
    pub fn new(
        config: &AdvancedRoutingConfig,
        default_provider: String,
    ) -> Result<Self, ServeError> {
        let length_rules = config
            .length_rules
            .iter()
            .map(|r| CompiledLengthRule {
                max_chars: r.max_chars,
                provider: r.provider.clone(),
                fallbacks: r.fallbacks.clone(),
            })
            .collect();

        let keyword_rules = config
            .keyword_rules
            .iter()
            .map(|r| {
                let regex = Regex::new(&r.pattern).map_err(|e| {
                    ServeError::Config(format!("invalid keyword regex '{}': {}", r.pattern, e))
                })?;
                Ok(CompiledKeywordRule {
                    regex,
                    provider: r.provider.clone(),
                    fallbacks: r.fallbacks.clone(),
                })
            })
            .collect::<Result<Vec<_>, ServeError>>()?;

        let load_balance_providers = config
            .load_balance
            .as_ref()
            .map(|lb| lb.providers.clone())
            .unwrap_or_default();

        Ok(Self {
            default_provider,
            length_rules,
            keyword_rules,
            load_balance_providers,
            lb_counter: AtomicUsize::new(0),
        })
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
}

#[async_trait]
impl Router for AdvancedRouter {
    async fn route(&self, req: &GenerateRequest) -> Result<RoutingDecision, ServeError> {
        let prompt = Self::prompt_text(req);

        // 1. Check keyword rules first (most specific).
        for rule in &self.keyword_rules {
            if rule.regex.is_match(&prompt) {
                return Ok(RoutingDecision {
                    provider_name: rule.provider.clone(),
                    reason: format!("keyword match '{}'", rule.regex.as_str()),
                    fallbacks: rule.fallbacks.clone(),
                });
            }
        }

        // 2. Check length rules (ordered by max_chars ascending in config).
        for rule in &self.length_rules {
            if prompt.len() <= rule.max_chars {
                return Ok(RoutingDecision {
                    provider_name: rule.provider.clone(),
                    reason: format!(
                        "prompt length {} <= {} chars",
                        prompt.len(),
                        rule.max_chars
                    ),
                    fallbacks: rule.fallbacks.clone(),
                });
            }
        }

        // 3. Load balance if providers are configured.
        if !self.load_balance_providers.is_empty() {
            let idx = self.lb_counter.fetch_add(1, Ordering::Relaxed)
                % self.load_balance_providers.len();
            let provider = self.load_balance_providers[idx].clone();
            let others: Vec<String> = self
                .load_balance_providers
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != idx)
                .map(|(_, p)| p.clone())
                .collect();
            return Ok(RoutingDecision {
                provider_name: provider,
                reason: "load balanced".to_string(),
                fallbacks: others,
            });
        }

        // 4. Default provider.
        Ok(RoutingDecision {
            provider_name: self.default_provider.clone(),
            reason: "default provider (no rule matched)".to_string(),
            fallbacks: vec![],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{AdvancedRoutingConfig, KeywordRule, LengthRule, LoadBalanceConfig};

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

    fn empty_config() -> AdvancedRoutingConfig {
        AdvancedRoutingConfig {
            length_rules: vec![],
            keyword_rules: vec![],
            load_balance: None,
        }
    }

    #[tokio::test]
    async fn short_prompt_routes_via_length_rule() {
        let config = AdvancedRoutingConfig {
            length_rules: vec![
                LengthRule {
                    max_chars: 100,
                    provider: "fast-provider".to_string(),
                    fallbacks: vec![],
                },
                LengthRule {
                    max_chars: 1000,
                    provider: "medium-provider".to_string(),
                    fallbacks: vec![],
                },
            ],
            ..empty_config()
        };
        let router = AdvancedRouter::new(&config, "default".to_string()).unwrap();

        let req = GenerateRequest {
            prompt: Some("Short".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "fast-provider");
        assert!(decision.reason.contains("chars"));
    }

    #[tokio::test]
    async fn long_prompt_falls_through_to_default() {
        let config = AdvancedRoutingConfig {
            length_rules: vec![LengthRule {
                max_chars: 5,
                provider: "fast-provider".to_string(),
                fallbacks: vec![],
            }],
            ..empty_config()
        };
        let router = AdvancedRouter::new(&config, "default-provider".to_string()).unwrap();

        let req = GenerateRequest {
            prompt: Some("This is a much longer prompt that exceeds the threshold".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "default-provider");
        assert!(decision.reason.contains("default"));
    }

    #[tokio::test]
    async fn keyword_match_routes_correctly() {
        let config = AdvancedRoutingConfig {
            keyword_rules: vec![KeywordRule {
                pattern: "(?i)(code|debug|function)".to_string(),
                provider: "code-provider".to_string(),
                fallbacks: vec!["backup".to_string()],
            }],
            ..empty_config()
        };
        let router = AdvancedRouter::new(&config, "default".to_string()).unwrap();

        let req = GenerateRequest {
            prompt: Some("Help me debug this function".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "code-provider");
        assert!(decision.reason.contains("keyword"));
    }

    #[tokio::test]
    async fn keyword_regex_is_case_insensitive() {
        let config = AdvancedRoutingConfig {
            keyword_rules: vec![KeywordRule {
                pattern: "(?i)(code|debug)".to_string(),
                provider: "code-provider".to_string(),
                fallbacks: vec![],
            }],
            ..empty_config()
        };
        let router = AdvancedRouter::new(&config, "default".to_string()).unwrap();

        let req = GenerateRequest {
            prompt: Some("Please DEBUG my app".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "code-provider");
    }

    #[tokio::test]
    async fn keyword_no_match_falls_through() {
        let config = AdvancedRoutingConfig {
            keyword_rules: vec![KeywordRule {
                pattern: "(?i)(code|debug)".to_string(),
                provider: "code-provider".to_string(),
                fallbacks: vec![],
            }],
            ..empty_config()
        };
        let router = AdvancedRouter::new(&config, "default-provider".to_string()).unwrap();

        let req = GenerateRequest {
            prompt: Some("Tell me about cooking".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "default-provider");
    }

    #[tokio::test]
    async fn load_balancing_rotates_across_providers() {
        let config = AdvancedRoutingConfig {
            load_balance: Some(LoadBalanceConfig {
                providers: vec![
                    "provider-a".to_string(),
                    "provider-b".to_string(),
                    "provider-c".to_string(),
                ],
            }),
            ..empty_config()
        };
        let router = AdvancedRouter::new(&config, "default".to_string()).unwrap();

        let mut seen = std::collections::HashMap::new();
        for _ in 0..9 {
            let decision = router.route(&base_request()).await.unwrap();
            *seen.entry(decision.provider_name).or_insert(0) += 1;
        }

        // Each provider should be hit exactly 3 times in 9 calls.
        assert_eq!(seen.get("provider-a"), Some(&3));
        assert_eq!(seen.get("provider-b"), Some(&3));
        assert_eq!(seen.get("provider-c"), Some(&3));
    }

    #[tokio::test]
    async fn fallbacks_included_in_length_rule_decision() {
        let config = AdvancedRoutingConfig {
            length_rules: vec![LengthRule {
                max_chars: 1000,
                provider: "primary".to_string(),
                fallbacks: vec!["backup-1".to_string(), "backup-2".to_string()],
            }],
            ..empty_config()
        };
        let router = AdvancedRouter::new(&config, "default".to_string()).unwrap();

        let decision = router.route(&base_request()).await.unwrap();
        assert_eq!(decision.provider_name, "primary");
        assert_eq!(decision.fallbacks, vec!["backup-1", "backup-2"]);
    }

    #[tokio::test]
    async fn fallbacks_included_in_keyword_rule_decision() {
        let config = AdvancedRoutingConfig {
            keyword_rules: vec![KeywordRule {
                pattern: "(?i)hello".to_string(),
                provider: "primary".to_string(),
                fallbacks: vec!["fallback-1".to_string()],
            }],
            ..empty_config()
        };
        let router = AdvancedRouter::new(&config, "default".to_string()).unwrap();

        let decision = router.route(&base_request()).await.unwrap();
        assert_eq!(decision.provider_name, "primary");
        assert_eq!(decision.fallbacks, vec!["fallback-1"]);
    }

    #[tokio::test]
    async fn no_matching_rule_uses_default_provider() {
        let config = empty_config();
        let router = AdvancedRouter::new(&config, "fallback-default".to_string()).unwrap();

        let decision = router.route(&base_request()).await.unwrap();
        assert_eq!(decision.provider_name, "fallback-default");
        assert!(decision.reason.contains("default"));
    }

    #[tokio::test]
    async fn keyword_rules_take_priority_over_length_rules() {
        let config = AdvancedRoutingConfig {
            keyword_rules: vec![KeywordRule {
                pattern: "(?i)hello".to_string(),
                provider: "keyword-provider".to_string(),
                fallbacks: vec![],
            }],
            length_rules: vec![LengthRule {
                max_chars: 1000,
                provider: "length-provider".to_string(),
                fallbacks: vec![],
            }],
            load_balance: None,
        };
        let router = AdvancedRouter::new(&config, "default".to_string()).unwrap();

        let req = GenerateRequest {
            prompt: Some("Hello world".to_string()),
            ..base_request()
        };
        let decision = router.route(&req).await.unwrap();
        assert_eq!(decision.provider_name, "keyword-provider");
    }

    #[tokio::test]
    async fn invalid_regex_returns_config_error() {
        let config = AdvancedRoutingConfig {
            keyword_rules: vec![KeywordRule {
                pattern: "(?P<invalid".to_string(),
                provider: "provider".to_string(),
                fallbacks: vec![],
            }],
            ..empty_config()
        };
        let result = AdvancedRouter::new(&config, "default".to_string());
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn load_balance_fallbacks_are_other_providers() {
        let config = AdvancedRoutingConfig {
            load_balance: Some(LoadBalanceConfig {
                providers: vec!["a".to_string(), "b".to_string(), "c".to_string()],
            }),
            ..empty_config()
        };
        let router = AdvancedRouter::new(&config, "default".to_string()).unwrap();

        let decision = router.route(&base_request()).await.unwrap();
        // The selected provider should not appear in fallbacks.
        assert!(!decision.fallbacks.contains(&decision.provider_name));
        // Fallbacks should contain the other two.
        assert_eq!(decision.fallbacks.len(), 2);
    }
}
