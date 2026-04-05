use std::sync::Arc;
use std::time::Duration;

use axum::body::Body;
use axum::http::{Request, StatusCode};
use tower::ServiceExt;

use llm_serve::api::{build_router, AppState};
use llm_serve::config::{
    AppConfig, ExecutorConfig, LlamaConfig, RoutingConfig, RoutingRule, ServerConfig,
};
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
            rules: vec![RoutingRule {
                name: "default".to_string(),
                task: None,
                max_prompt_length: None,
                keywords: None,
                provider: "mock".to_string(),
                fallbacks: None,
            }],
        }),
        executor: ExecutorConfig::default(),
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
