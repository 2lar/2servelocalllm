use std::time::Duration;

use metrics::{counter, histogram};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Install the Prometheus metrics recorder and return the handle
/// used to render the `/metrics` endpoint.
pub fn init_metrics() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder")
}

/// Increment the `llm_requests_total` counter.
pub fn record_request(provider: &str, task: &str, status: &str) {
    counter!("llm_requests_total", "provider" => provider.to_string(), "task" => task.to_string(), "status" => status.to_string())
        .increment(1);
}

/// Record a latency observation in the `llm_request_duration_seconds` histogram.
pub fn record_latency(provider: &str, duration: Duration) {
    histogram!("llm_request_duration_seconds", "provider" => provider.to_string())
        .record(duration.as_secs_f64());
}

/// Increment the `llm_tokens_total` counters for input and output tokens.
pub fn record_tokens(provider: &str, input: u32, output: u32) {
    counter!("llm_tokens_total", "direction" => "input", "provider" => provider.to_string())
        .increment(u64::from(input));
    counter!("llm_tokens_total", "direction" => "output", "provider" => provider.to_string())
        .increment(u64::from(output));
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify that the recording functions don't panic when called.
    /// We can't easily assert counter values without installing a test recorder
    /// (which conflicts with the global recorder), so we verify correctness
    /// via the integration test that hits GET /metrics.
    #[test]
    fn record_functions_do_not_panic() {
        // Install a no-op recorder so metrics macros don't fail.
        // metrics crate silently drops if no recorder is installed, so this
        // just verifies the label construction doesn't panic.
        record_request("test-provider", "code", "success");
        record_request("test-provider", "chat", "error");
        record_latency("test-provider", Duration::from_millis(123));
        record_tokens("test-provider", 10, 20);
    }
}
