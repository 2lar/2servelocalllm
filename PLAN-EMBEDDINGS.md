# Plan: Add Embeddings Endpoint to llm-serve

## Goal

Add an OpenAI-compatible `/v1/embeddings` endpoint to llm-serve so B2 (and any other app) can generate text embeddings locally via llama.cpp without external API dependencies.

## Context

- llama.cpp already serves `/v1/embeddings` natively when loaded with an embedding model
- llm-serve currently proxies only chat/completion endpoints
- B2's embedding service expects an OpenAI-compatible embeddings API
- The embedding model (`nomic-embed-text-v1.5` GGUF) is separate from the chat model

## Architecture Decision: Dual Model Support

llm-serve currently manages one llama-server process for chat. Embedding requires a different model. Two options:

**Option A — Single process, model swap:** Load embedding model in the same llama-server. Not viable — can't serve chat and embeddings simultaneously with different models.

**Option B — Dual process (recommended):** Spawn a second llama-server instance for the embedding model on a separate port. llm-serve manages both processes and routes requests to the appropriate one.

```
llm-serve :11434
├── /v1/messages      → llama-server :8080 (chat model, e.g. qwen-27b)
├── /v1/generate      → llama-server :8080
└── /v1/embeddings    → llama-server :8081 (embedding model, e.g. nomic-embed-text)
```

---

## Implementation Plan

### Step 1: Config — Add embedding model support

**File:** `serve/src/config.rs`

Add a new `EmbeddingConfig` section to `AppConfig`:

```rust
pub struct AppConfig {
    pub server: ServerConfig,
    pub llama: LlamaConfig,
    pub embedding: Option<EmbeddingConfig>,  // NEW
    pub providers: HashMap<String, ProviderConfig>,
    // ... rest unchanged
}
```

```rust
#[derive(Debug, Deserialize, Clone)]
pub struct EmbeddingConfig {
    pub enabled: bool,
    pub binary: String,            // path to llama-server binary (can share with llama.binary)
    pub model: String,             // path to embedding GGUF model
    pub host: String,              // default: "127.0.0.1"
    pub port: u16,                 // default: 8081 (different from chat port)
    pub gpu_layers: u32,           // embedding models are small, can run on CPU (0) or GPU
    pub ctx_size: u32,             // context window for embedding (default: 2048)
    pub health_check_timeout_secs: u64,
    pub health_check_interval_ms: u64,
}
```

**Config file addition** (`config.toml`):

```toml
[embedding]
enabled = true
binary = "../llama.cpp/build/bin/llama-server"
model = "../models/nomic-embed-text-v1.5.Q8_0.gguf"
host = "127.0.0.1"
port = 8081
gpu_layers = 0          # embedding model is small, CPU is fine
ctx_size = 2048
health_check_timeout_secs = 30
health_check_interval_ms = 1000
```

Environment override: `APP__EMBEDDING__MODEL=/path/to/model.gguf`

**Defaults:** Add `impl Default for EmbeddingConfig` with sensible values.

---

### Step 2: Process Management — Spawn second llama-server

**File:** `serve/src/process/mod.rs`

The existing `ProcessManager` manages a single `Child`. Extend to support multiple named processes.

**Option A (minimal):** Add a second `ProcessManager` instance for embedding. Keep the existing struct unchanged.

In `main.rs`, create two ProcessManager instances:

```rust
let mut chat_process = ProcessManager::new();
let mut embed_process = ProcessManager::new();

if config.llama.enabled {
    chat_process.start(&config.llama).await?;
}
if let Some(ref embed_cfg) = config.embedding {
    if embed_cfg.enabled {
        // Convert EmbeddingConfig to LlamaConfig-compatible args
        embed_process.start_embedding(embed_cfg).await?;
    }
}
```

**New method on ProcessManager** (`start_embedding`):

```rust
pub async fn start_embedding(&mut self, config: &EmbeddingConfig) -> Result<(), ServeError> {
    let mut command = Command::new(&config.binary);
    // Set library paths (same as chat)
    command
        .arg("--model").arg(&config.model)
        .arg("--n-gpu-layers").arg(config.gpu_layers.to_string())
        .arg("--port").arg(config.port.to_string())
        .arg("--host").arg(&config.host)
        .arg("--ctx-size").arg(config.ctx_size.to_string())
        .arg("--embedding")  // KEY FLAG: tells llama-server to enable embedding endpoint
        .kill_on_drop(true);

    self.child = Some(command.spawn()?);
    self.wait_for_health(&config.host, config.port, config.health_check_timeout_secs, config.health_check_interval_ms).await?;
    Ok(())
}
```

The `--embedding` flag is critical — it tells llama-server to expose the `/v1/embeddings` endpoint.

**Shutdown:** Both processes must be shut down in the graceful shutdown handler.

---

### Step 3: API Types — Add embedding request/response

**File:** `serve/src/api/types.rs`

```rust
/// OpenAI-compatible embedding request
#[derive(Debug, Deserialize, Serialize)]
pub struct EmbedRequest {
    /// Text(s) to embed — string or array of strings
    pub input: EmbedInput,
    /// Model name (optional, uses configured default)
    pub model: Option<String>,
    /// Encoding format (optional, default: "float")
    pub encoding_format: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(untagged)]
pub enum EmbedInput {
    Single(String),
    Batch(Vec<String>),
}

impl EmbedInput {
    pub fn into_vec(self) -> Vec<String> {
        match self {
            EmbedInput::Single(s) => vec![s],
            EmbedInput::Batch(v) => v,
        }
    }
}

/// OpenAI-compatible embedding response
#[derive(Debug, Deserialize, Serialize)]
pub struct EmbedResponse {
    pub object: String,             // "list"
    pub data: Vec<EmbeddingData>,
    pub model: String,
    pub usage: EmbedUsage,
    // llm-serve additions
    pub provider: Option<String>,
    pub latency_ms: Option<u64>,
    pub cached: Option<bool>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbeddingData {
    pub object: String,             // "embedding"
    pub embedding: Vec<f32>,
    pub index: usize,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct EmbedUsage {
    pub prompt_tokens: u32,
    pub total_tokens: u32,
}
```

These match the OpenAI embeddings API format exactly, so B2's existing client works unchanged.

---

### Step 4: Provider — Add embed method

**File:** `serve/src/provider/mod.rs`

Extend the Provider trait:

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn generate(&self, req: &GenerateRequest) -> Result<GenerateResponse, ServeError>;
    async fn generate_stream(&self, req: &GenerateRequest) -> Result<ChunkStream, ServeError>;
    async fn embed(&self, req: &EmbedRequest) -> Result<EmbedResponse, ServeError> {
        // Default implementation returns "not supported"
        Err(ServeError::NotSupported("embeddings not supported by this provider".into()))
    }
}
```

Default implementation means existing providers (mock, etc.) don't break.

**File:** `serve/src/provider/local.rs`

Add embedding implementation. This calls the EMBEDDING llama-server (port 8081), not the chat one (port 8080).

```rust
pub struct LocalEmbeddingProvider {
    url: String,        // "http://127.0.0.1:8081"
    model: String,
    client: Client,
}

impl LocalEmbeddingProvider {
    pub fn new(config: &EmbeddingConfig) -> Result<Self, ServeError> {
        let client = Client::builder()
            .timeout(Duration::from_secs(30))
            .build()?;
        Ok(Self {
            url: format!("http://{}:{}", config.host, config.port),
            model: config.model.clone(),
            client,
        })
    }

    pub async fn embed(&self, req: &EmbedRequest) -> Result<EmbedResponse, ServeError> {
        let start = Instant::now();

        let texts = req.input.clone().into_vec();

        let body = serde_json::json!({
            "input": texts,
            "model": req.model.as_deref().unwrap_or(&self.model),
        });

        let resp = self.client
            .post(format!("{}/v1/embeddings", self.url))
            .json(&body)
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(ServeError::Provider(format!("embedding request failed ({}): {}", status, body)));
        }

        let mut embed_resp: EmbedResponse = resp.json().await?;
        embed_resp.latency_ms = Some(start.elapsed().as_millis() as u64);
        embed_resp.provider = Some("local-embedding".to_string());

        Ok(embed_resp)
    }
}
```

**Design note:** Embedding uses a separate struct (not the Provider trait) because it routes to a different llama-server instance. The trait could be extended later if needed.

---

### Step 5: Route Handler

**File:** `serve/src/api/routes.rs`

```rust
#[tracing::instrument(skip(state, req))]
pub async fn embeddings(
    State(state): State<Arc<AppState>>,
    Json(req): Json<EmbedRequest>,
) -> Result<impl IntoResponse, ServeError> {
    let embedding_provider = state.embedding_provider.as_ref()
        .ok_or_else(|| ServeError::NotSupported("embedding endpoint not configured".into()))?;

    // Check cache
    let cache_key = if let Some(ref cache) = state.cache {
        let key = format!("embed:{}", serde_json::to_string(&req).unwrap_or_default());
        if let Some(cached) = cache.get(&key).await {
            metrics::counter!("embedding_cache_hits").increment(1);
            return Ok(Json(cached).into_response());
        }
        Some(key)
    } else {
        None
    };

    let resp = embedding_provider.embed(&req).await?;

    // Store in cache
    if let (Some(ref cache), Some(ref key)) = (&state.cache, &cache_key) {
        cache.set(key, &resp).await;
    }

    metrics::counter!("embedding_requests_total").increment(1);

    Ok(Json(resp).into_response())
}
```

**File:** `serve/src/api/mod.rs`

Add to AppState:

```rust
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub provider_registry: Arc<ProviderRegistry>,
    pub router: Arc<dyn LlmRouter>,
    pub executor: Arc<Executor>,
    pub metrics_handle: Option<PrometheusHandle>,
    pub cache: Option<Arc<dyn Cache>>,
    pub eval_store: Option<Arc<EvalStore>>,
    pub embedding_provider: Option<Arc<LocalEmbeddingProvider>>,  // NEW
}
```

Register the route:

```rust
.route("/v1/embeddings", post(routes::embeddings))
```

---

### Step 6: Wire up in main.rs

**File:** `serve/src/main.rs`

After starting the chat process, start the embedding process:

```rust
// Start embedding llama-server if configured
let mut embed_process = ProcessManager::new();
let embedding_provider = if let Some(ref embed_cfg) = config.embedding {
    if embed_cfg.enabled {
        info!("Starting embedding model server...");
        embed_process.start_embedding(embed_cfg).await?;
        Some(Arc::new(LocalEmbeddingProvider::new(embed_cfg)?))
    } else {
        None
    }
} else {
    None
};

// Add to AppState
let state = Arc::new(AppState {
    // ... existing fields ...
    embedding_provider,
});

// Shutdown both processes
tokio::select! {
    // ... existing shutdown handling ...
}
chat_process.shutdown().await;
embed_process.shutdown().await;
```

---

### Step 7: Metrics

Add embedding-specific metrics in the handler or provider:

```rust
metrics::counter!("embedding_requests_total").increment(1);
metrics::histogram!("embedding_latency_ms").record(latency_ms as f64);
metrics::counter!("embedding_tokens_total").increment(token_count as u64);
metrics::counter!("embedding_cache_hits").increment(1);  // when cached
```

---

### Step 8: Error Handling

**File:** `serve/src/error.rs`

Add a `NotSupported` variant if it doesn't exist:

```rust
pub enum ServeError {
    // ... existing variants ...
    NotSupported(String),
}
```

Map to HTTP 501:

```rust
ServeError::NotSupported(msg) => (StatusCode::NOT_IMPLEMENTED, msg),
```

---

## Testing Plan

### Unit Tests

1. **Type serialization** — `EmbedRequest`/`EmbedResponse` serialize to/from OpenAI format
2. **EmbedInput::into_vec** — handles both single string and array
3. **Cache key generation** — same request produces same key

### Integration Tests (with wiremock)

4. **Embeddings handler** — mock llama-server, send request, verify response format
5. **Embedding not configured** — returns 501 when `embedding.enabled = false`
6. **Cache hit** — second identical request returns cached result
7. **Error propagation** — llama-server error returns 502 to client

### Manual Tests

8. Start llm-serve with embedding config → verify `/v1/embeddings` responds
9. Point B2 at llm-serve → verify embeddings generated correctly
10. Check `/metrics` for embedding counters

---

## Config Example (complete)

```toml
[server]
host = "0.0.0.0"
port = 11434

[llama]
enabled = true
binary = "../llama.cpp/build/bin/llama-server"
model = "../models/qwen35-27b-q8.gguf"
host = "127.0.0.1"
port = 8080
gpu_layers = 99
ctx_size = 16384
health_check_timeout_secs = 60
health_check_interval_ms = 2000

[embedding]
enabled = true
binary = "../llama.cpp/build/bin/llama-server"
model = "../models/nomic-embed-text-v1.5.Q8_0.gguf"
host = "127.0.0.1"
port = 8081
gpu_layers = 0
ctx_size = 2048
health_check_timeout_secs = 30
health_check_interval_ms = 1000

[providers.local]
name = "local-qwen"
url = "http://localhost:8080"
model = "qwen35-distilled"
timeout_secs = 120

[cache]
enabled = true
max_entries = 1000
ttl_secs = 3600

# ... rest unchanged
```

---

## B2 Integration

Once llm-serve has the embeddings endpoint, B2's config becomes:

```
EMBEDDING_BASE_URL=http://localhost:11434
EMBEDDING_MODEL=nomic-embed-text-v1.5
EMBEDDING_DIMENSIONS=768
EMBEDDING_API_KEY=         # empty — no key needed for local
```

No B2 code changes required — it already speaks OpenAI-compatible embeddings API.

---

## File Change Summary

| File | Change | Lines (est.) |
|------|--------|-------------|
| `config.rs` | Add `EmbeddingConfig` struct + defaults | ~30 |
| `api/types.rs` | Add `EmbedRequest`, `EmbedResponse`, `EmbeddingData`, `EmbedUsage` | ~50 |
| `provider/mod.rs` | Add default `embed()` to Provider trait | ~5 |
| `provider/local.rs` | Add `LocalEmbeddingProvider` struct + `embed()` | ~50 |
| `api/routes.rs` | Add `embeddings` handler | ~30 |
| `api/mod.rs` | Add field to AppState, register route | ~5 |
| `process/mod.rs` | Add `start_embedding()` method | ~25 |
| `main.rs` | Wire up embedding process + provider | ~20 |
| `error.rs` | Add `NotSupported` variant | ~5 |
| `config.toml` | Add `[embedding]` section | ~10 |
| **Total** | | **~230 lines** |

---

## Model Download

```bash
# Download the embedding model (~275MB)
# Option 1: Hugging Face CLI
huggingface-cli download nomic-ai/nomic-embed-text-v1.5-GGUF nomic-embed-text-v1.5.Q8_0.gguf --local-dir ../models/

# Option 2: Direct download
wget -P ../models/ https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q8_0.gguf
```

---

## Build Order

1. Config + types (no runtime dependencies)
2. Process management (start_embedding)
3. Provider (LocalEmbeddingProvider)
4. Handler + route registration
5. main.rs wiring
6. Tests
7. Download model + manual verification
8. Update B2 config defaults
