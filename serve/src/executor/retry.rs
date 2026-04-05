use std::cmp::min;
use std::time::Duration;

use crate::config::RetryConfig;

pub struct RetryPolicy {
    pub max_retries: u32,
    pub initial_backoff_ms: u64,
    pub max_backoff_ms: u64,
}

impl RetryPolicy {
    pub fn from_config(config: &RetryConfig) -> Self {
        Self {
            max_retries: config.max_retries,
            initial_backoff_ms: config.initial_backoff_ms,
            max_backoff_ms: config.max_backoff_ms,
        }
    }

    /// Calculate the backoff duration for a given attempt (0-indexed).
    /// Uses exponential backoff: initial_backoff_ms * 2^attempt, capped at max_backoff_ms.
    pub fn backoff_duration(&self, attempt: u32) -> Duration {
        let multiplier = 1u64.checked_shl(attempt).unwrap_or(u64::MAX);
        let backoff_ms = self.initial_backoff_ms.saturating_mul(multiplier);
        Duration::from_millis(min(backoff_ms, self.max_backoff_ms))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_policy() -> RetryPolicy {
        RetryPolicy {
            max_retries: 3,
            initial_backoff_ms: 500,
            max_backoff_ms: 5000,
        }
    }

    #[test]
    fn backoff_doubles_each_attempt() {
        let policy = test_policy();
        assert_eq!(policy.backoff_duration(0), Duration::from_millis(500));
        assert_eq!(policy.backoff_duration(1), Duration::from_millis(1000));
        assert_eq!(policy.backoff_duration(2), Duration::from_millis(2000));
        assert_eq!(policy.backoff_duration(3), Duration::from_millis(4000));
    }

    #[test]
    fn backoff_caps_at_max() {
        let policy = test_policy();
        // 500 * 2^4 = 8000, but max is 5000
        assert_eq!(policy.backoff_duration(4), Duration::from_millis(5000));
        assert_eq!(policy.backoff_duration(10), Duration::from_millis(5000));
    }

    #[test]
    fn backoff_handles_zero_initial() {
        let policy = RetryPolicy {
            max_retries: 2,
            initial_backoff_ms: 0,
            max_backoff_ms: 1000,
        };
        assert_eq!(policy.backoff_duration(0), Duration::from_millis(0));
        assert_eq!(policy.backoff_duration(5), Duration::from_millis(0));
    }

    #[test]
    fn backoff_handles_large_attempt_without_overflow() {
        let policy = test_policy();
        // Very large attempt number should not panic from overflow
        let duration = policy.backoff_duration(63);
        assert_eq!(duration, Duration::from_millis(5000));
    }

    #[test]
    fn from_config_maps_fields() {
        let config = RetryConfig {
            max_retries: 5,
            initial_backoff_ms: 100,
            max_backoff_ms: 10000,
        };
        let policy = RetryPolicy::from_config(&config);
        assert_eq!(policy.max_retries, 5);
        assert_eq!(policy.initial_backoff_ms, 100);
        assert_eq!(policy.max_backoff_ms, 10000);
    }

    #[test]
    fn retryable_vs_non_retryable_errors() {
        use crate::error::ServeError;

        // Timeout is retryable
        let err = ServeError::Timeout { timeout_secs: 30 };
        assert!(err.is_retryable());

        // ProviderNotFound is not retryable
        let err = ServeError::ProviderNotFound("missing".to_string());
        assert!(!err.is_retryable());

        // Routing error is not retryable
        let err = ServeError::Routing("no match".to_string());
        assert!(!err.is_retryable());

        // Internal error is not retryable
        let err = ServeError::Internal("bug".to_string());
        assert!(!err.is_retryable());
    }
}
