use std::sync::Arc;
use std::time::Duration;

use tokio::signal;

use llm_serve::api::{build_router, AppState};
use llm_serve::cache::memory::MemoryCache;
use llm_serve::cache::Cache;
use llm_serve::config::load_config;
use llm_serve::embedding::EmbeddingProvider;
use llm_serve::eval::store::EvalStore;
use llm_serve::executor::Executor;
use llm_serve::observability::logging::init_tracing;
use llm_serve::observability::metrics::init_metrics;
use llm_serve::process::ProcessManager;
use llm_serve::provider::local::LocalProvider;
use llm_serve::provider::mock::MockProvider;
use llm_serve::provider::registry::ProviderRegistry;
use llm_serve::router::advanced::AdvancedRouter;
use llm_serve::router::rule_based::RuleBasedRouter;
use llm_serve::router::Router;

#[tokio::main]
async fn main() {
    let config = load_config().expect("failed to load config");

    init_tracing(&config.observability.log_format, &config.observability.log_level);
    let metrics_handle = init_metrics();
    let embedding_enabled = config
        .embedding
        .as_ref()
        .is_some_and(|e| e.enabled);

    tracing::info!(
        host = %config.server.host,
        port = config.server.port,
        llama_enabled = config.llama.enabled,
        embedding_enabled,
        "starting llm-serve"
    );

    let mut process_manager: Option<ProcessManager> = None;
    let mut embed_process_manager: Option<ProcessManager> = None;
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

    // Start embedding llama-server if configured.
    let embedding_provider = if let Some(ref embed_cfg) = config.embedding {
        if embed_cfg.enabled {
            let pm = ProcessManager::start_embedding(embed_cfg)
                .await
                .expect("failed to start embedding llama-server");
            embed_process_manager = Some(pm);

            let provider = EmbeddingProvider::new(embed_cfg)
                .expect("failed to create embedding provider");
            Some(Arc::new(provider))
        } else {
            None
        }
    } else {
        None
    };

    let router: Arc<dyn Router> = match config
        .routing
        .as_ref()
        .map(|r| r.strategy.as_str())
    {
        Some("advanced") => {
            let routing_config = config.routing.as_ref().unwrap();
            let default_provider = routing_config
                .default_provider
                .clone()
                .unwrap_or_else(|| "local-qwen".to_string());
            let advanced_config = routing_config
                .advanced
                .as_ref()
                .expect("routing.advanced config required when strategy = 'advanced'");
            let router = AdvancedRouter::new(advanced_config, default_provider)
                .expect("failed to create advanced router");
            tracing::info!("using advanced router");
            Arc::new(router)
        }
        _ => {
            let routing_rules = config
                .routing
                .as_ref()
                .map(|r| r.rules.clone())
                .unwrap_or_default();
            tracing::info!("using rule-based router");
            Arc::new(RuleBasedRouter::new(routing_rules))
        }
    };
    let registry = Arc::new(registry);

    let cache: Option<Arc<dyn Cache>> = if config.cache.enabled {
        tracing::info!(
            max_entries = config.cache.max_entries,
            ttl_secs = config.cache.ttl_secs,
            "cache enabled"
        );
        Some(Arc::new(MemoryCache::new(&config.cache)))
    } else {
        tracing::info!("cache disabled");
        None
    };

    let eval_store: Option<Arc<EvalStore>> = if config.eval.enabled {
        tracing::info!(
            max_records = config.eval.max_records,
            "eval store enabled"
        );
        Some(Arc::new(EvalStore::new(config.eval.max_records)))
    } else {
        tracing::info!("eval store disabled");
        None
    };

    let mut executor = Executor::new(Arc::clone(&registry), &config.executor);
    if let Some(ref cache) = cache {
        executor = executor.with_cache(Arc::clone(cache));
    }
    let executor = Arc::new(executor);

    let state = Arc::new(AppState {
        config: Arc::new(config.clone()),
        provider_registry: registry,
        router,
        executor,
        metrics_handle: Some(metrics_handle),
        cache,
        eval_store,
        embedding_provider,
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

    if let Some(mut pm) = embed_process_manager {
        pm.shutdown().await;
    }
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
