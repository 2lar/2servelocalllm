use tracing_subscriber::EnvFilter;

/// Initialize the tracing subscriber with structured output.
///
/// - `format`: "json" for machine-readable output, "pretty" for human-readable.
/// - `level`: default filter level (e.g. "info"), overridden by `RUST_LOG` env var.
pub fn init_tracing(format: &str, level: &str) {
    let env_filter =
        EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new(level));

    match format {
        "pretty" => {
            tracing_subscriber::fmt()
                .with_env_filter(env_filter)
                .init();
        }
        _ => {
            // Default to JSON for production.
            tracing_subscriber::fmt()
                .json()
                .with_env_filter(env_filter)
                .init();
        }
    }
}
