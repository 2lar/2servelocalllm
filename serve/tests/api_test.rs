use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use llm_serve::api::{build_router, AppState};
use llm_serve::config::{
    AppConfig, CacheConfig, EvalConfig, ExecutorConfig, LlamaConfig, ObservabilityConfig,
    RoutingConfig, RoutingRule, ServerConfig,
};
use llm_serve::eval::store::EvalStore;
use llm_serve::executor::Executor;
use llm_serve::provider::mock::MockProvider;
use llm_serve::provider::registry::ProviderRegistry;
use llm_serve::router::rule_based::RuleBasedRouter;

fn test_config() -> AppConfig {
    AppConfig {
        server: ServerConfig {
            host: "127.0.0.1".to_string(),
            port: 3000,
        },
        llama: LlamaConfig {
            enabled: false,
            binary: String::new(),
            model: String::new(),
            host: "127.0.0.1".to_string(),
            port: 8080,
            gpu_layers: 0,
            ctx_size: 2048,
            health_check_timeout_secs: 5,
            health_check_interval_ms: 500,
        },
        providers: std::collections::HashMap::new(),
        routing: Some(RoutingConfig {
            strategy: "rule_based".to_string(),
            default_provider: None,
            rules: vec![RoutingRule {
                name: "default".to_string(),
                task: None,
                max_prompt_length: None,
                keywords: None,
                provider: "mock".to_string(),
                fallbacks: None,
            }],
            advanced: None,
        }),
        executor: ExecutorConfig::default(),
        observability: ObservabilityConfig::default(),
        cache: CacheConfig::default(),
        eval: EvalConfig::default(),
    }
}

fn test_state() -> Arc<AppState> {
    let mut registry = ProviderRegistry::new();
    registry.register(
        "mock".to_string(),
        Arc::new(MockProvider::new(Duration::from_millis(0))),
    );

    let config = test_config();
    let rules = config.routing.as_ref().unwrap().rules.clone();
    let registry = Arc::new(registry);
    let executor = Arc::new(Executor::new(Arc::clone(&registry), &config.executor));

    Arc::new(AppState {
        config: Arc::new(config),
        provider_registry: registry,
        router: Arc::new(RuleBasedRouter::new(rules)),
        executor,
        metrics_handle: None,
        cache: None,
        eval_store: None,
    })
}

fn test_state_with_eval() -> Arc<AppState> {
    let mut registry = ProviderRegistry::new();
    registry.register(
        "mock".to_string(),
        Arc::new(MockProvider::new(Duration::from_millis(0))),
    );

    let config = test_config();
    let rules = config.routing.as_ref().unwrap().rules.clone();
    let registry = Arc::new(registry);
    let executor = Arc::new(Executor::new(Arc::clone(&registry), &config.executor));
    let eval_store = Arc::new(EvalStore::new(10000));

    Arc::new(AppState {
        config: Arc::new(config),
        provider_registry: registry,
        router: Arc::new(RuleBasedRouter::new(rules)),
        executor,
        metrics_handle: None,
        cache: None,
        eval_store: Some(eval_store),
    })
}

#[tokio::test]
async fn health_returns_ok() {
    let app = build_router(test_state());

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/health")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), 1024).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["status"], "ok");
}

#[tokio::test]
async fn generate_returns_mock_response() {
    let app = build_router(test_state());

    let body = serde_json::json!({
        "prompt": "Hello",
        "max_tokens": 100
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["output"], "This is a mock response.");
    assert_eq!(json["provider"], "mock");
    assert_eq!(json["cached"], false);
}

#[tokio::test]
async fn generate_stream_returns_sse() {
    let app = build_router(test_state());

    let body = serde_json::json!({
        "prompt": "Hello",
        "stream": true
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    assert!(resp
        .headers()
        .get("content-type")
        .unwrap()
        .to_str()
        .unwrap()
        .contains("text/event-stream"));
}

#[tokio::test]
async fn generate_with_messages() {
    let app = build_router(test_state());

    let body = serde_json::json!({
        "messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"}
        ]
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
}

#[tokio::test]
async fn not_found_returns_404() {
    let app = build_router(test_state());

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/nonexistent")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn generate_includes_routing_info() {
    let app = build_router(test_state());

    let body = serde_json::json!({
        "prompt": "Hello"
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["routing"].is_object());
    assert_eq!(json["routing"]["provider_name"], "mock");
    assert!(json["routing"]["matched_rule"]
        .as_str()
        .unwrap()
        .contains("default"));
}

#[tokio::test]
async fn generate_with_explicit_provider_bypasses_routing() {
    let app = build_router(test_state());

    let body = serde_json::json!({
        "prompt": "Hello",
        "provider": "mock"
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert_eq!(json["provider"], "mock");
    // No routing info when provider is explicitly specified
    assert!(json.get("routing").is_none() || json["routing"].is_null());
}

#[tokio::test]
async fn generate_with_unknown_provider_returns_error() {
    let app = build_router(test_state());

    let body = serde_json::json!({
        "prompt": "Hello",
        "provider": "nonexistent-provider"
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn routing_selects_correct_provider_by_task() {
    let mut registry = ProviderRegistry::new();
    registry.register(
        "mock".to_string(),
        Arc::new(MockProvider::new(Duration::from_millis(0))),
    );

    let rules = vec![
        RoutingRule {
            name: "code-tasks".to_string(),
            task: Some("code".to_string()),
            max_prompt_length: None,
            keywords: None,
            provider: "mock".to_string(),
            fallbacks: None,
        },
        RoutingRule {
            name: "default".to_string(),
            task: None,
            max_prompt_length: None,
            keywords: None,
            provider: "mock".to_string(),
            fallbacks: None,
        },
    ];

    let config = test_config();
    let registry = Arc::new(registry);
    let executor = Arc::new(Executor::new(Arc::clone(&registry), &config.executor));

    let state = Arc::new(AppState {
        config: Arc::new(config),
        provider_registry: registry,
        router: Arc::new(RuleBasedRouter::new(rules)),
        executor,
        metrics_handle: None,
        cache: None,
        eval_store: None,
    });

    let app = build_router(state);

    let body = serde_json::json!({
        "prompt": "Write a function",
        "task": "code"
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
    assert!(json["routing"]["matched_rule"]
        .as_str()
        .unwrap()
        .contains("code-tasks"));
}

#[tokio::test]
async fn metrics_endpoint_returns_prometheus_text() {
    // Use a state with metrics_handle = None. The endpoint should still respond
    // (with a 503 in this case, since no recorder is installed in tests).
    let app = build_router(test_state());

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    // Without a PrometheusHandle, we get 503.
    assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
}

#[tokio::test]
async fn metrics_endpoint_with_handle_returns_200() {
    use metrics_exporter_prometheus::PrometheusBuilder;

    let handle = PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install test recorder");

    // Record a metric so the output is non-empty.
    llm_serve::observability::metrics::record_request("test-provider", "code", "success");

    let mut registry = ProviderRegistry::new();
    registry.register(
        "mock".to_string(),
        Arc::new(MockProvider::new(Duration::from_millis(0))),
    );

    let config = test_config();
    let rules = config.routing.as_ref().unwrap().rules.clone();
    let registry = Arc::new(registry);
    let executor = Arc::new(Executor::new(Arc::clone(&registry), &config.executor));

    let state = Arc::new(AppState {
        config: Arc::new(config),
        provider_registry: registry,
        router: Arc::new(RuleBasedRouter::new(rules)),
        executor,
        metrics_handle: Some(handle),
        cache: None,
        eval_store: None,
    });

    let app = build_router(state);

    let resp = app
        .oneshot(
            Request::builder()
                .uri("/metrics")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);

    let body = axum::body::to_bytes(resp.into_body(), 16384).await.unwrap();
    let text = String::from_utf8(body.to_vec()).unwrap();
    assert!(
        text.contains("llm_requests_total"),
        "metrics output should contain llm_requests_total, got: {text}"
    );
}

#[tokio::test]
async fn cached_response_returns_cached_true() {
    use llm_serve::cache::memory::MemoryCache;
    use llm_serve::cache::Cache;

    let mut registry = ProviderRegistry::new();
    registry.register(
        "mock".to_string(),
        Arc::new(MockProvider::new(Duration::from_millis(0))),
    );

    let config = test_config();
    let rules = config.routing.as_ref().unwrap().rules.clone();
    let registry = Arc::new(registry);

    let cache: Arc<dyn Cache> = Arc::new(MemoryCache::new(&config.cache));
    let executor = Arc::new(
        Executor::new(Arc::clone(&registry), &config.executor)
            .with_cache(Arc::clone(&cache)),
    );

    let state = Arc::new(AppState {
        config: Arc::new(config),
        provider_registry: registry,
        router: Arc::new(RuleBasedRouter::new(rules)),
        executor,
        metrics_handle: None,
        cache: Some(cache),
        eval_store: None,
    });

    let body_json = serde_json::json!({
        "prompt": "Hello",
        "max_tokens": 100
    });
    let body_str = serde_json::to_string(&body_json).unwrap();

    // First request: cache miss, provider is called.
    let app = build_router(state.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(body_str.clone()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["cached"], false);

    // Second request: cache hit, should return cached: true.
    let app = build_router(state.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(body_str))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["cached"], true);
}

#[tokio::test]
async fn generate_then_eval_stats_shows_request() {
    let state = test_state_with_eval();

    // Generate a request so it gets recorded in the eval store.
    let app = build_router(state.clone());
    let body = serde_json::json!({
        "prompt": "Hello",
        "task": "code"
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Now check eval stats.
    let app = build_router(state.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/eval/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    let stats = json["stats"].as_array().unwrap();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0]["provider"], "mock");
    assert_eq!(stats[0]["task"], "code");
    assert_eq!(stats[0]["request_count"], 1);
}

#[tokio::test]
async fn evaluate_sets_score_reflected_in_stats() {
    let state = test_state_with_eval();

    // Generate a request.
    let app = build_router(state.clone());
    let body = serde_json::json!({
        "prompt": "Hello",
        "task": "code"
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/generate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let gen_json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    let response_id = gen_json["id"].as_str().unwrap().to_string();

    // Set score via POST /v1/evaluate.
    let app = build_router(state.clone());
    let eval_body = serde_json::json!({
        "id": response_id,
        "score": 0.85
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/evaluate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&eval_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();
    assert_eq!(resp.status(), StatusCode::OK);

    // Check stats reflect the score.
    let app = build_router(state.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/eval/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();

    let stats = json["stats"].as_array().unwrap();
    assert_eq!(stats.len(), 1);
    assert_eq!(stats[0]["avg_score"], 0.85);
}

#[tokio::test]
async fn eval_best_returns_correct_provider() {
    let state = test_state_with_eval();
    let eval_store = state.eval_store.as_ref().unwrap();

    // Manually insert records with scores to control provider names.
    use llm_serve::api::types::{GenerateRequest, GenerateResponse, Usage};
    use llm_serve::eval::EvalRecord;

    let req = GenerateRequest {
        prompt: Some("test".to_string()),
        messages: None,
        task: Some("code".to_string()),
        max_tokens: None,
        temperature: None,
        stream: None,
        provider: None,
    };

    let make_resp = |id: &str, provider: &str| GenerateResponse {
        id: id.to_string(),
        output: "test".to_string(),
        model: "test".to_string(),
        provider: provider.to_string(),
        latency_ms: 100,
        usage: Usage { input_tokens: 5, output_tokens: 3 },
        cached: false,
        routing: None,
    };

    eval_store
        .record(EvalRecord {
            id: "r1".to_string(),
            request: req.clone(),
            response: make_resp("r1", "provider-a"),
            provider: "provider-a".to_string(),
            task: Some("code".to_string()),
            latency_ms: 100,
            score: Some(0.7),
            created_at: chrono::Utc::now(),
        })
        .await;

    eval_store
        .record(EvalRecord {
            id: "r2".to_string(),
            request: req.clone(),
            response: make_resp("r2", "provider-b"),
            provider: "provider-b".to_string(),
            task: Some("code".to_string()),
            latency_ms: 100,
            score: Some(0.95),
            created_at: chrono::Utc::now(),
        })
        .await;

    let app = build_router(state.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/eval/best?task=code")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::OK);
    let bytes = axum::body::to_bytes(resp.into_body(), 4096).await.unwrap();
    let json: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(json["task"], "code");
    assert_eq!(json["best_provider"], "provider-b");
}

#[tokio::test]
async fn evaluate_with_invalid_score_returns_400() {
    let state = test_state_with_eval();

    let app = build_router(state.clone());
    let eval_body = serde_json::json!({
        "id": "some-id",
        "score": 1.5
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/evaluate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&eval_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn evaluate_with_negative_score_returns_400() {
    let state = test_state_with_eval();

    let app = build_router(state.clone());
    let eval_body = serde_json::json!({
        "id": "some-id",
        "score": -0.1
    });

    let resp = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/v1/evaluate")
                .header("content-type", "application/json")
                .body(Body::from(serde_json::to_string(&eval_body).unwrap()))
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn eval_endpoints_return_error_when_eval_disabled() {
    let state = test_state(); // eval_store is None

    let app = build_router(state.clone());
    let resp = app
        .oneshot(
            Request::builder()
                .uri("/v1/eval/stats")
                .body(Body::empty())
                .unwrap(),
        )
        .await
        .unwrap();

    assert_eq!(resp.status(), StatusCode::INTERNAL_SERVER_ERROR);
}
