# LLM Serving Layer — Implementation Plan

## What This Project Is

An intelligent middleware that sits between applications and language models. It decides **which** model to use, **how** to execute the request, and **tracks** what happened. It abstracts model usage behind a unified interface.

This is **not** a chatbot, not an API wrapper, not a model runtime. It is the infrastructure layer behind AI systems — the kind of system built by platform teams at OpenAI, Anthropic, and AWS.

## What Already Exists

| Component | What it does | Role going forward |
|-----------|-------------|-------------------|
| `llama.cpp/` | Inference engine — loads models, runs token generation | Untouched. The LocalProvider connects to it as an external service. |
| `scripts/proxy.py` | Translates Anthropic API format to OpenAI format | Reference only. Its logic informs the LocalProvider implementation. |
| `scripts/start.sh` | Launches llama-server with Qwen 3.5 27B | Reference for launch args. Replaced by built-in process management. |

## Language Choice: Rust

- Predictable latency (no GC pauses)
- Type safety across routing/provider/executor boundaries
- Excellent async ecosystem (Tokio + Axum)
- Low memory footprint
- Systems-level credibility for infrastructure work

The inference bottleneck is in llama.cpp (C++). The serving layer is I/O-bound, but Rust gives us correctness guarantees and production-grade reliability.

## Development Constraints

- **Dev machine** (this machine): No GPU, no model. All development and testing uses MockProvider + wiremock HTTP fixtures.
- **Deployment machine**: Has GPU + model. Integration testing with live llama-server happens here.
- `cargo test` must always pass on the dev machine with zero external dependencies.

---

## Architecture

```
  llm-serve (single binary)
  ┌──────────────────────────────────────────────────┐
  │                                                  │
  │  Process Manager        Spawns & monitors        │
  │  └── llama-server       llama-server as a child  │
  │       (subprocess)      process on startup       │
  │                                                  │
  │  API Layer (Axum)       POST /v1/generate        │
  │       |                 GET /health, /metrics    │
  │       v                                          │
  │  Routing Engine         Decides which provider   │
  │       |                                          │
  │       v                                          │
  │  Executor               Retry/timeout/fallback   │
  │       |                                          │
  │       v                                          │
  │  Providers                                       │
  │  ├── LocalProvider      HTTP to llama-server     │
  │  ├── MockProvider       Canned (dev/test)        │
  │  └── (future providers)                          │
  │       |                                          │
  │       v                                          │
  │  Observability          Metrics, logging, cache  │
  └──────────────────────────────────────────────────┘

One command: `llm-serve` starts everything.
llama-server is managed as a child process — started on boot, health-checked, killed on shutdown.
```

---

## Project Structure

```
serve/                          # Rust project root
├── Cargo.toml
├── config.toml                 # Default configuration
├── src/
│   ├── main.rs                 # Entry: load config, build app, serve
│   ├── lib.rs                  # Re-exports for integration tests
│   ├── config.rs               # AppConfig structs + loading
│   ├── error.rs                # ServeError enum + HTTP mapping
│   ├── api/
│   │   ├── mod.rs
│   │   ├── routes.rs           # Endpoints
│   │   ├── types.rs            # Request/response types
│   │   └── middleware.rs       # Request ID, timing, logging
│   ├── process/
│   │   ├── mod.rs              # Process manager — spawns/monitors llama-server
│   │   └── health.rs           # Health check polling (wait for readiness)
│   ├── provider/
│   │   ├── mod.rs              # Provider trait
│   │   ├── local.rs            # LocalProvider (wraps llama-server)
│   │   ├── mock.rs             # MockProvider (dev/tests)
│   │   └── registry.rs         # ProviderRegistry
│   ├── router/
│   │   ├── mod.rs              # Router trait
│   │   ├── rule_based.rs       # Config-driven rule matching
│   │   └── advanced.rs         # Heuristics, fallbacks, load balancing
│   ├── executor/
│   │   ├── mod.rs              # Routes decision to provider
│   │   └── retry.rs            # Retry + timeout policies
│   ├── cache/
│   │   ├── mod.rs              # Cache trait
│   │   ├── memory.rs           # LRU in-memory cache
│   │   └── metrics.rs          # Hit/miss counters
│   ├── observability/
│   │   ├── mod.rs
│   │   ├── logging.rs          # Structured tracing setup
│   │   └── metrics.rs          # Prometheus counters/histograms
│   └── eval/
│       ├── mod.rs
│       └── store.rs            # Output storage + comparison
└── tests/
    ├── api_test.rs
    ├── provider_test.rs
    ├── router_test.rs
    └── fixtures/
        ├── chat_completion.json
        └── chat_completion_stream.txt
```

---

## Core Traits

### Provider — each backend implements this

```rust
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn generate(&self, req: &GenerateRequest) -> Result<GenerateResponse, ServeError>;
    async fn generate_stream(&self, req: &GenerateRequest) -> Result<ChunkStream, ServeError>;
}
```

### Router — decides which provider handles a request

```rust
pub struct RoutingDecision {
    pub provider_name: String,
    pub reason: String,
    pub fallbacks: Vec<String>,
}

#[async_trait]
pub trait Router: Send + Sync {
    async fn route(&self, req: &GenerateRequest) -> Result<RoutingDecision, ServeError>;
}
```

### Cache — optional response caching

```rust
#[async_trait]
pub trait Cache: Send + Sync {
    async fn get(&self, req: &GenerateRequest) -> Option<GenerateResponse>;
    async fn put(&self, req: &GenerateRequest, resp: &GenerateResponse);
}
```

---

## Application State

```rust
pub struct AppState {
    pub config: Arc<AppConfig>,
    pub router: Arc<dyn Router>,
    pub executor: Arc<Executor>,
    pub cache: Option<Arc<dyn Cache>>,
    pub eval_store: Option<Arc<EvalStore>>,
}
```

---

## Dependencies

| Crate | Purpose |
|-------|---------|
| `axum` | Web framework |
| `tokio` | Async runtime |
| `reqwest` | HTTP client (calls llama-server) |
| `serde` / `serde_json` | Serialization |
| `tracing` / `tracing-subscriber` | Structured logging |
| `thiserror` | Error types |
| `config` | TOML config + env overrides |
| `lru` | In-memory cache |
| `sha2` | Cache key hashing |
| `metrics` / `metrics-exporter-prometheus` | Prometheus metrics |
| `uuid` | Request IDs |
| `async-trait` | Async trait support |
| `tower-http` | Middleware (CORS, tracing, request-id) |
| **Dev:** `wiremock` | Mock HTTP server for tests |

---

## Phased Implementation

### Phase 1 — Unified API + Local Provider

**Goal:** A single endpoint that calls llama-server through a clean provider abstraction.

**Build order:**
1. `Cargo.toml` — dependencies
2. `error.rs` — ServeError enum with HTTP status mapping
3. `config.rs` — load from config.toml + APP_ env prefix (includes llama-server launch args)
4. `api/types.rs` — GenerateRequest, GenerateResponse, ChatMessage, Usage, StreamChunk
5. `process/mod.rs` — ProcessManager: spawns llama-server as child process, polls health endpoint until ready, kills on drop
6. `process/health.rs` — health check loop (retry with backoff until `/health` returns OK)
7. `provider/mod.rs` — Provider trait
8. `provider/mock.rs` — MockProvider with canned responses
9. `provider/local.rs` — LocalProvider calling llama-server's `/v1/chat/completions`
10. `api/routes.rs` — `POST /v1/generate` (non-streaming first), `GET /health`
11. `main.rs` — load config, spawn llama-server via ProcessManager, wait for ready, start API server, graceful shutdown kills child
12. Streaming: `generate_stream` on LocalProvider, SSE via `axum::response::Sse`

**Config:**
```toml
[server]
host = "0.0.0.0"
port = 3000

# llama-server process management
[llama]
enabled = true                              # set false on dev machine (no model)
binary = "./llama.cpp/build/bin/llama-server"
model = "~/llm/models/qwen35-27b-q8.gguf"
host = "0.0.0.0"
port = 8080
gpu_layers = 99
ctx_size = 16384
health_check_timeout_secs = 60              # max time to wait for llama-server ready
health_check_interval_ms = 2000             # poll interval

[providers.local]
name = "local-qwen"
url = "http://localhost:8080"               # matches llama.port above
model = "qwen35-distilled"
timeout_secs = 120
```

**Process Manager behavior:**
- On startup: if `llama.enabled`, spawn llama-server with configured args as a child process
- Poll `GET http://{llama.host}:{llama.port}/health` until OK or timeout
- If llama-server crashes during operation, log error (future: auto-restart)
- On SIGINT/SIGTERM: kill child process, then shut down API server
- On dev machine: set `llama.enabled = false`, llama-server is not spawned, use MockProvider

**Tests:**
- MockProvider unit tests
- LocalProvider against wiremock (recorded llama-server responses)
- API integration tests using MockProvider

**Verification:**
- `cargo test` passes on dev machine
- `cargo run` starts server, `GET /health` returns OK
- On deployment machine: real `POST /v1/generate` returns model output

---

### Phase 2 — Routing Engine

**Goal:** Config-driven request routing based on task type.

**Build:**
1. Router trait + RoutingDecision struct
2. RuleBasedRouter — match on task, prompt length, keywords
3. ProviderRegistry — name-to-provider lookup
4. Update routes to use router instead of hardcoded provider

**Config:**
```toml
[[routing.rules]]
task = "code"
provider = "local-qwen"

[[routing.rules]]
task = "default"
provider = "local-qwen"
```

**Tests:** Unit tests for routing logic, fallback to default, explicit provider override.

---

### Phase 3 — Execution Layer

**Goal:** Separate routing from execution. Add retry, timeout, fallback.

**Build:**
1. Executor — takes RoutingDecision + request, calls provider from registry
2. RetryPolicy — configurable retry with exponential backoff
3. Fallback — on primary failure, try fallback providers in order
4. Per-request timeout via `tokio::time::timeout`

**Tests:** Mock provider that fails N times (test retry), always-fail provider (test fallback), slow provider (test timeout).

---

### Phase 4 — Observability

**Goal:** Structured logging and Prometheus metrics.

**Build:**
1. Tracing subscriber — JSON output, env filter
2. Metrics — `llm_requests_total`, `llm_request_duration_seconds`, `llm_tokens_total`
3. Request middleware — assigns UUID, records timing, logs span
4. `GET /metrics` endpoint

**Tests:** Verify counters increment after mock requests.

---

### Phase 5 — Caching

**Goal:** Deduplicate repeated prompts.

**Build:**
1. Cache trait
2. MemoryCache — LRU keyed by SHA-256 of canonicalized request
3. Executor checks cache before provider, stores on success
4. `cached: true` flag on responses
5. Configurable max_entries and TTL

**Config:**
```toml
[cache]
enabled = true
max_entries = 1000
ttl_secs = 3600
```

**Tests:** Insert/retrieve, TTL expiry, LRU eviction, cache flag in response.

---

### Phase 6 — Advanced Routing

**Goal:** Smarter routing heuristics.

**Build:**
1. AdvancedRouter — prompt length thresholds, keyword regex, load balancing
2. Fallback chains per strategy
3. Config-driven thresholds and patterns

**Tests:** Long prompt routing, keyword detection, distribution across providers.

---

### Phase 7 — Evaluation System

**Goal:** Track and compare model performance over time.

**Build:**
1. EvalStore — stores request/response/metadata
2. `POST /v1/evaluate` — submit quality scores
3. `GET /v1/eval/stats` — best model per task type
4. Quality scoring hook trait

**Tests:** Store entries, query stats, verify aggregation.

---

### Phase 8 — Infrastructure

**Goal:** Containerize for deployment.

**Build:**
1. Multi-stage Dockerfile (Rust builder -> minimal Debian runtime)
2. docker-compose.yml (llm-serve + llama-server)
3. Deployment configs

---

## Setup Steps

1. Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
2. Initialize project: `cargo init serve` in repo root
3. Build Phase 1 with ACE
4. `cargo test` after each phase
5. Deploy and integration-test on GPU machine

## Running

**On deployment machine (has GPU + model):**
```bash
cd serve
cargo run --release
# That's it. Spawns llama-server, waits for ready, starts API on port 3000.
```

**On dev machine (no GPU):**
```bash
# config.toml has llama.enabled = false
cd serve
cargo run
# Starts API with MockProvider only. No llama-server spawned.
```

**Shutdown:** Ctrl+C — gracefully kills llama-server child process and drains connections.
