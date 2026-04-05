use std::sync::Arc;
use std::time::Duration;

use tokio::signal;
use tracing_subscriber::EnvFilter;

use llm_serve::api::{build_router, AppState};
use llm_serve::config::load_config;
use llm_serve::executor::Executor;
use llm_serve::process::ProcessManager;
use llm_serve::provider::local::LocalProvider;
use llm_serve::provider::mock::MockProvider;
use llm_serve::provider::registry::ProviderRegistry;
use llm_serve::router::rule_based::RuleBasedRouter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = load_config().expect("failed to load config");
    tracing::info!(
        host = %config.server.host,
        port = config.server.port,
        llama_enabled = config.llama.enabled,
        "starting llm-serve"
    );

    let mut process_manager: Option<ProcessManager> = None;
    let mut registry = ProviderRegistry::new();

    if config.llama.enabled {
        let pm = ProcessManager::start(&config.llama)
            .await
            .expect("failed to start llama-server");
        process_manager = Some(pm);

        let provider_config = config
            .providers
            .get("local")
            .expect("missing providers.local config")
            .clone();

        let provider = LocalProvider::new(provider_config).expect("failed to create local provider");
        registry.register("local-qwen".to_string(), Arc::new(provider));
    } else {
        tracing::info!("llama disabled, using mock provider");
        let mock = MockProvider::new(Duration::from_millis(50));
        registry.register("mock".to_string(), Arc::new(mock));
    }

    let routing_rules = config
        .routing
        .as_ref()
        .map(|r| r.rules.clone())
        .unwrap_or_default();

    let router = Arc::new(RuleBasedRouter::new(routing_rules));
    let registry = Arc::new(registry);
    let executor = Arc::new(Executor::new(
        Arc::clone(&registry),
        &config.executor,
    ));

    let state = Arc::new(AppState {
        config: Arc::new(config.clone()),
        provider_registry: registry,
        router,
        executor,
    });

    let app = build_router(state);
    let bind_addr = format!("{}:{}", config.server.host, config.server.port);
    let listener = tokio::net::TcpListener::bind(&bind_addr)
        .await
        .expect("failed to bind");

    tracing::info!("listening on {bind_addr}");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");

    if let Some(mut pm) = process_manager {
        pm.shutdown().await;
    }

    tracing::info!("shutdown complete");
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("failed to install ctrl+c handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => { tracing::info!("received SIGINT"); }
        _ = terminate => { tracing::info!("received SIGTERM"); }
    }
}
