pub mod retry;

use std::sync::Arc;
use std::time::{Duration, Instant};

use tracing::{info, warn};

use crate::api::types::{GenerateRequest, GenerateResponse};
use crate::config::ExecutorConfig;
use crate::error::ServeError;
use crate::observability::metrics as obs;
use crate::provider::registry::ProviderRegistry;
use crate::provider::ChunkStream;
use crate::router::RoutingDecision;

use retry::RetryPolicy;

pub struct Executor {
    registry: Arc<ProviderRegistry>,
    timeout: Duration,
    retry_policy: RetryPolicy,
}

impl Executor {
    pub fn new(registry: Arc<ProviderRegistry>, config: &ExecutorConfig) -> Self {
        Self {
            registry,
            timeout: Duration::from_secs(config.timeout_secs),
            retry_policy: RetryPolicy::from_config(&config.retry),
        }
    }

    /// Execute a non-streaming request. Tries the primary provider with retries,
    /// then falls back to each fallback provider (also with retries).
    #[tracing::instrument(skip(self, request), fields(provider = %decision.provider_name))]
    pub async fn execute(
        &self,
        decision: &RoutingDecision,
        request: &GenerateRequest,
    ) -> Result<GenerateResponse, ServeError> {
        let task = request.task.as_deref().unwrap_or("default");
        let start = Instant::now();

        // Build the ordered list of providers to try: primary, then fallbacks.
        let mut providers_to_try = vec![&decision.provider_name];
        for fallback in &decision.fallbacks {
            providers_to_try.push(fallback);
        }

        let mut last_error = ServeError::Internal("no providers to try".to_string());

        for provider_name in providers_to_try {
            match self
                .try_provider_with_retries(provider_name, request)
                .await
            {
                Ok(response) => {
                    let duration = start.elapsed();
                    obs::record_request(provider_name, task, "success");
                    obs::record_latency(provider_name, duration);
                    obs::record_tokens(
                        provider_name,
                        response.usage.input_tokens,
                        response.usage.output_tokens,
                    );
                    return Ok(response);
                }
                Err(err) => {
                    warn!(
                        provider = %provider_name,
                        error = %err,
                        "provider failed, trying next"
                    );
                    last_error = err;
                }
            }
        }

        // All providers failed — record the error against the primary provider.
        let duration = start.elapsed();
        obs::record_request(&decision.provider_name, task, "error");
        obs::record_latency(&decision.provider_name, duration);

        Err(last_error)
    }

    /// Execute a streaming request. Tries the primary provider, then fallbacks.
    /// Retries are not applied to streaming requests since partial streams
    /// cannot be transparently restarted.
    pub async fn execute_stream(
        &self,
        decision: &RoutingDecision,
        request: &GenerateRequest,
    ) -> Result<ChunkStream, ServeError> {
        let mut providers_to_try = vec![&decision.provider_name];
        for fallback in &decision.fallbacks {
            providers_to_try.push(fallback);
        }

        let mut last_error = ServeError::Internal("no providers to try".to_string());

        for provider_name in providers_to_try {
            let provider = match self.registry.get(provider_name) {
                Some(p) => p,
                None => {
                    last_error = ServeError::ProviderNotFound(provider_name.clone());
                    continue;
                }
            };

            let result = tokio::time::timeout(self.timeout, provider.generate_stream(request))
                .await;

            match result {
                Ok(Ok(stream)) => return Ok(stream),
                Ok(Err(err)) => {
                    warn!(
                        provider = %provider_name,
                        error = %err,
                        "stream provider failed, trying next"
                    );
                    last_error = err;
                }
                Err(_elapsed) => {
                    warn!(
                        provider = %provider_name,
                        timeout_secs = self.timeout.as_secs(),
                        "stream provider timed out, trying next"
                    );
                    last_error = ServeError::Timeout {
                        timeout_secs: self.timeout.as_secs(),
                    };
                }
            }
        }

        Err(last_error)
    }

    /// Try a single provider with retry policy. Retries only on retryable errors.
    async fn try_provider_with_retries(
        &self,
        provider_name: &str,
        request: &GenerateRequest,
    ) -> Result<GenerateResponse, ServeError> {
        let provider = self
            .registry
            .get(provider_name)
            .ok_or_else(|| ServeError::ProviderNotFound(provider_name.to_string()))?;

        let mut last_error: Option<ServeError> = None;

        // Attempt 0 is the first call, then up to max_retries additional attempts.
        let total_attempts = 1 + self.retry_policy.max_retries;

        for attempt in 0..total_attempts {
            if attempt > 0 {
                let backoff = self.retry_policy.backoff_duration(attempt - 1);
                info!(
                    provider = %provider_name,
                    attempt,
                    backoff_ms = backoff.as_millis() as u64,
                    "retrying provider"
                );
                tokio::time::sleep(backoff).await;
            }

            let result =
                tokio::time::timeout(self.timeout, provider.generate(request)).await;

            match result {
                Ok(Ok(response)) => return Ok(response),
                Ok(Err(err)) => {
                    if !err.is_retryable() {
                        // Non-retryable error — don't retry, propagate immediately.
                        return Err(err);
                    }
                    warn!(
                        provider = %provider_name,
                        attempt,
                        error = %err,
                        "retryable error"
                    );
                    last_error = Some(err);
                }
                Err(_elapsed) => {
                    let err = ServeError::Timeout {
                        timeout_secs: self.timeout.as_secs(),
                    };
                    warn!(
                        provider = %provider_name,
                        attempt,
                        timeout_secs = self.timeout.as_secs(),
                        "request timed out"
                    );
                    last_error = Some(err);
                }
            }
        }

        Err(last_error.unwrap_or_else(|| {
            ServeError::Internal(format!("provider '{provider_name}' failed with no error"))
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{GenerateRequest, GenerateResponse, Usage};
    use crate::config::{ExecutorConfig, RetryConfig};
    use crate::provider::mock::MockProvider;
    use crate::provider::{ChunkStream, Provider};

    use async_trait::async_trait;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn test_request() -> GenerateRequest {
        GenerateRequest {
            prompt: Some("Hello".to_string()),
            messages: None,
            task: None,
            max_tokens: None,
            temperature: None,
            stream: None,
            provider: None,
        }
    }

    fn fast_config() -> ExecutorConfig {
        ExecutorConfig {
            timeout_secs: 5,
            retry: RetryConfig {
                max_retries: 2,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
            },
        }
    }

    fn simple_decision(provider: &str) -> RoutingDecision {
        RoutingDecision {
            provider_name: provider.to_string(),
            reason: "test".to_string(),
            fallbacks: vec![],
        }
    }

    fn decision_with_fallbacks(primary: &str, fallbacks: Vec<&str>) -> RoutingDecision {
        RoutingDecision {
            provider_name: primary.to_string(),
            reason: "test".to_string(),
            fallbacks: fallbacks.into_iter().map(String::from).collect(),
        }
    }

    /// A provider that fails a configurable number of times before succeeding.
    struct FailNTimesProvider {
        name: String,
        fail_count: AtomicU32,
        times_to_fail: u32,
    }

    impl FailNTimesProvider {
        fn new(name: &str, times_to_fail: u32) -> Self {
            Self {
                name: name.to_string(),
                fail_count: AtomicU32::new(0),
                times_to_fail,
            }
        }
    }

    #[async_trait]
    impl Provider for FailNTimesProvider {
        fn name(&self) -> &str {
            &self.name
        }

        async fn generate(&self, _req: &GenerateRequest) -> Result<GenerateResponse, ServeError> {
            let count = self.fail_count.fetch_add(1, Ordering::SeqCst);
            if count < self.times_to_fail {
                // Simulate a retryable provider error by using a connection error.
                // We use reqwest to create a real connect error.
                let err = reqwest::get("http://127.0.0.1:1")
                    .await
                    .expect_err("should fail to connect");
                return Err(ServeError::Provider {
                    provider: self.name.clone(),
                    source: err,
                });
            }

            Ok(GenerateResponse {
                id: "test-id".to_string(),
                output: format!("success from {}", self.name),
                model: "test-model".to_string(),
                provider: self.name.clone(),
                latency_ms: 0,
                usage: Usage {
                    input_tokens: 5,
                    output_tokens: 3,
                },
                cached: false,
                routing: None,
            })
        }

        async fn generate_stream(
            &self,
            _req: &GenerateRequest,
        ) -> Result<ChunkStream, ServeError> {
            Err(ServeError::Internal("not implemented".to_string()))
        }
    }

    /// A provider that always fails with a retryable error.
    struct AlwaysFailProvider {
        name: String,
    }

    impl AlwaysFailProvider {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
            }
        }
    }

    #[async_trait]
    impl Provider for AlwaysFailProvider {
        fn name(&self) -> &str {
            &self.name
        }

        async fn generate(&self, _req: &GenerateRequest) -> Result<GenerateResponse, ServeError> {
            let err = reqwest::get("http://127.0.0.1:1")
                .await
                .expect_err("should fail to connect");
            Err(ServeError::Provider {
                provider: self.name.clone(),
                source: err,
            })
        }

        async fn generate_stream(
            &self,
            _req: &GenerateRequest,
        ) -> Result<ChunkStream, ServeError> {
            let err = reqwest::get("http://127.0.0.1:1")
                .await
                .expect_err("should fail to connect");
            Err(ServeError::Provider {
                provider: self.name.clone(),
                source: err,
            })
        }
    }

    /// A provider that sleeps longer than any reasonable timeout.
    struct SlowProvider {
        name: String,
        delay: Duration,
    }

    impl SlowProvider {
        fn new(name: &str, delay: Duration) -> Self {
            Self {
                name: name.to_string(),
                delay,
            }
        }
    }

    #[async_trait]
    impl Provider for SlowProvider {
        fn name(&self) -> &str {
            &self.name
        }

        async fn generate(&self, _req: &GenerateRequest) -> Result<GenerateResponse, ServeError> {
            tokio::time::sleep(self.delay).await;
            Ok(GenerateResponse {
                id: "slow-id".to_string(),
                output: "slow response".to_string(),
                model: "test-model".to_string(),
                provider: self.name.clone(),
                latency_ms: self.delay.as_millis() as u64,
                usage: Usage {
                    input_tokens: 1,
                    output_tokens: 1,
                },
                cached: false,
                routing: None,
            })
        }

        async fn generate_stream(
            &self,
            _req: &GenerateRequest,
        ) -> Result<ChunkStream, ServeError> {
            tokio::time::sleep(self.delay).await;
            Err(ServeError::Internal("too slow".to_string()))
        }
    }

    #[tokio::test]
    async fn successful_execution_returns_response() {
        let mut registry = ProviderRegistry::new();
        registry.register(
            "mock".to_string(),
            Arc::new(MockProvider::new(Duration::from_millis(0))),
        );

        let executor = Executor::new(Arc::new(registry), &fast_config());
        let decision = simple_decision("mock");

        let resp = executor.execute(&decision, &test_request()).await.unwrap();
        assert_eq!(resp.provider, "mock");
        assert_eq!(resp.output, "This is a mock response.");
    }

    #[tokio::test]
    async fn retry_on_failure_then_succeed() {
        let mut registry = ProviderRegistry::new();
        // Fails twice, then succeeds. With max_retries=2, total attempts = 3.
        registry.register(
            "flaky".to_string(),
            Arc::new(FailNTimesProvider::new("flaky", 2)),
        );

        let executor = Executor::new(Arc::new(registry), &fast_config());
        let decision = simple_decision("flaky");

        let resp = executor.execute(&decision, &test_request()).await.unwrap();
        assert_eq!(resp.output, "success from flaky");
    }

    #[tokio::test]
    async fn fallback_to_second_provider_when_primary_fails() {
        let mut registry = ProviderRegistry::new();
        registry.register(
            "primary".to_string(),
            Arc::new(AlwaysFailProvider::new("primary")),
        );
        registry.register(
            "backup".to_string(),
            Arc::new(MockProvider::new(Duration::from_millis(0))),
        );

        let config = ExecutorConfig {
            timeout_secs: 5,
            retry: RetryConfig {
                max_retries: 0, // No retries, go straight to fallback
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
            },
        };

        let executor = Executor::new(Arc::new(registry), &config);
        let decision = decision_with_fallbacks("primary", vec!["backup"]);

        let resp = executor.execute(&decision, &test_request()).await.unwrap();
        assert_eq!(resp.provider, "mock");
    }

    #[tokio::test]
    async fn all_providers_fail_returns_last_error() {
        let mut registry = ProviderRegistry::new();
        registry.register(
            "fail-a".to_string(),
            Arc::new(AlwaysFailProvider::new("fail-a")),
        );
        registry.register(
            "fail-b".to_string(),
            Arc::new(AlwaysFailProvider::new("fail-b")),
        );

        let config = ExecutorConfig {
            timeout_secs: 5,
            retry: RetryConfig {
                max_retries: 0,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
            },
        };

        let executor = Executor::new(Arc::new(registry), &config);
        let decision = decision_with_fallbacks("fail-a", vec!["fail-b"]);

        let err = executor
            .execute(&decision, &test_request())
            .await
            .unwrap_err();
        // The last error should be from fail-b
        let msg = err.to_string();
        assert!(msg.contains("fail-b"), "expected fail-b in error: {msg}");
    }

    #[tokio::test]
    async fn timeout_returns_timeout_error() {
        let mut registry = ProviderRegistry::new();
        registry.register(
            "slow".to_string(),
            Arc::new(SlowProvider::new("slow", Duration::from_secs(10))),
        );

        let config = ExecutorConfig {
            timeout_secs: 1, // 1 second timeout
            retry: RetryConfig {
                max_retries: 0, // No retries to keep test fast
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
            },
        };

        let executor = Executor::new(Arc::new(registry), &config);
        let decision = simple_decision("slow");

        let err = executor
            .execute(&decision, &test_request())
            .await
            .unwrap_err();
        assert!(
            matches!(err, ServeError::Timeout { .. }),
            "expected Timeout, got: {err}"
        );
    }

    #[tokio::test]
    async fn provider_not_found_returns_error() {
        let registry = ProviderRegistry::new();
        let executor = Executor::new(Arc::new(registry), &fast_config());
        let decision = simple_decision("nonexistent");

        let err = executor
            .execute(&decision, &test_request())
            .await
            .unwrap_err();
        assert!(matches!(err, ServeError::ProviderNotFound(_)));
    }

    #[tokio::test]
    async fn stream_falls_back_on_failure() {
        let mut registry = ProviderRegistry::new();
        registry.register(
            "fail".to_string(),
            Arc::new(AlwaysFailProvider::new("fail")),
        );
        registry.register(
            "mock".to_string(),
            Arc::new(MockProvider::new(Duration::from_millis(0))),
        );

        let config = ExecutorConfig {
            timeout_secs: 5,
            retry: RetryConfig {
                max_retries: 0,
                initial_backoff_ms: 10,
                max_backoff_ms: 50,
            },
        };

        let executor = Executor::new(Arc::new(registry), &config);
        let decision = decision_with_fallbacks("fail", vec!["mock"]);

        let stream = executor
            .execute_stream(&decision, &test_request())
            .await;
        assert!(stream.is_ok());
    }
}
