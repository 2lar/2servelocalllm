# LLM Serving Layer — Setup & Usage Guide
## llm-serve + llama-server on WSL2 (RTX 5090, 32GB VRAM)

---

## How It Works (Before vs Now)

**Before:**
```
Claude Code CLI → proxy.py (port 11434) → llama-server (port 8080) → GPU
```
Three separate pieces, two to start manually, no routing, no metrics, no caching.

**Now:**
```
Any App / curl
      │  (standard HTTP)
      ▼
llm-serve (port 3000)              ← one binary does everything
├── Routing Engine                  decides which model handles the request
├── Executor                        retries, timeouts, fallbacks
├── Cache                           deduplicates repeated prompts
├── Metrics                         Prometheus counters + latency histograms
├── Eval Store                      tracks model quality over time
│
├── Process Manager ──→ llama-server (port 8080, spawned automatically)
│                              │
│                              ▼
│                        Qwen3.5-27B Q8_0 on RTX 5090
│
└── Provider Abstraction        (future: OpenAI, Anthropic, etc.)
```

**One command. One binary. Everything managed.**

---

## Directory Structure

```
~/llm/
├── 2servelocalllm/
│   ├── serve/              ← the Rust project
│   │   ├── config.toml     ← all configuration
│   │   ├── src/            ← source code
│   │   └── target/release/ ← compiled binary (after cargo build)
│   ├── llama.cpp/          ← inference engine
│   ├── scripts/            ← old proxy.py + start.sh (reference only)
│   └── docker-compose.yml  ← for Docker deployments
├── models/
│   └── qwen35-27b-q8.gguf ← model weights
```

---

## First-Time Setup (Deployment Machine)

### 1. Install Rust (one-time)

Skip this if using Docker (see Docker section below).

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source "$HOME/.cargo/env"
```

### 2. Build llama.cpp (if not already built)

```bash
cd ~/llm/2servelocalllm/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j$(nproc)
```

### 3. Build llm-serve

```bash
cd ~/llm/2servelocalllm/serve
cargo build --release
```

### 4. Configure for your machine

Edit `serve/config.toml`:

```toml
[llama]
enabled = true                                          # ← change from false to true
binary = "../llama.cpp/build/bin/llama-server"           # ← path to llama-server binary
model = "~/llm/models/qwen35-27b-q8.gguf"               # ← path to your model
host = "0.0.0.0"
port = 8080
gpu_layers = 99
ctx_size = 16384
health_check_timeout_secs = 60
health_check_interval_ms = 2000
```

Everything else can stay as defaults.

---

## Starting the Stack

```bash
cd ~/llm/2servelocalllm/serve
cargo run --release
```

You'll see:
```
starting llm-serve, llama_enabled=true
spawning llama-server...
waiting for llama-server health check...
llama-server is ready
listening on 0.0.0.0:3000
```

That's it. llm-serve spawns llama-server, waits until the model is loaded onto GPU, then starts accepting requests on port 3000.

### Add a shell alias (optional)

```bash
echo "alias ai-start='cd ~/llm/2servelocalllm/serve && cargo run --release'" >> ~/.bashrc
source ~/.bashrc
```

Then just: `ai-start`

---

## Using It

### Basic generation

```bash
curl -X POST http://localhost:3000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what a hash table is in one paragraph.",
    "max_tokens": 200
  }'
```

Response:
```json
{
  "id": "req_abc123",
  "output": "A hash table is...",
  "model": "qwen35-distilled",
  "provider": "local-qwen",
  "latency_ms": 1523,
  "usage": { "input_tokens": 12, "output_tokens": 87 },
  "cached": false,
  "routing": {
    "matched_rule": "default",
    "provider_name": "local-qwen"
  }
}
```

### Chat-style messages

```bash
curl -X POST http://localhost:3000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is Rust?"}
    ],
    "max_tokens": 300
  }'
```

### Streaming (SSE)

```bash
curl -N -X POST http://localhost:3000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about programming.",
    "stream": true,
    "max_tokens": 200
  }'
```

Returns server-sent events:
```
data: {"delta":"A ","done":false}
data: {"delta":"loop ","done":false}
data: {"delta":"that ","done":false}
...
data: {"delta":"","done":true}
```

### Specify a task (for routing)

```bash
curl -X POST http://localhost:3000/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Fix this Python bug: ...",
    "task": "code",
    "max_tokens": 500
  }'
```

The routing engine uses the `task` field to decide which provider handles the request (configurable in config.toml).

---

## Using with Claude Code

llm-serve is a drop-in replacement for the Anthropic API. Claude Code talks to it natively via `POST /v1/messages`.

### .bashrc aliases

```bash
# Start the full stack (one command)
alias ai-start='cd ~/llm/2servelocalllm/serve && cargo run --release'

# Stop the full stack
alias ai-stop='pkill -f llm-serve; echo Stack stopped.'

# Claude Code → local model (free, private, routed through llm-serve)
alias claude-local='ANTHROPIC_BASE_URL="http://localhost:3000" ANTHROPIC_AUTH_TOKEN="local-dev" ANTHROPIC_API_KEY="" claude'

# Claude Code → Anthropic cloud (requires API key)
alias claude-cloud='unset ANTHROPIC_BASE_URL; unset ANTHROPIC_AUTH_TOKEN; claude'
```

Add these to your `~/.bashrc`:

```bash
cat >> ~/.bashrc << 'EOF'
alias ai-start='cd ~/llm/2servelocalllm/serve && cargo run --release'
alias ai-stop='pkill -f llm-serve; echo Stack stopped.'
alias claude-local='ANTHROPIC_BASE_URL="http://localhost:3000" ANTHROPIC_AUTH_TOKEN="local-dev" ANTHROPIC_API_KEY="" claude'
alias claude-cloud='unset ANTHROPIC_BASE_URL; unset ANTHROPIC_AUTH_TOKEN; claude'
EOF
source ~/.bashrc
```

### Daily workflow

```bash
# Terminal 1: start the stack
ai-start
# Wait for "listening on 0.0.0.0:3000"

# Terminal 2: use Claude Code with your local model
cd ~/your-project
claude-local
```

That's it. Claude Code sends Anthropic API requests to llm-serve on port 3000, which routes them through the serving layer to llama-server on port 8080.

---

## Endpoints Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | HEAD/GET | Liveness check (for Claude Code handshake) |
| `/health` | GET | Health check — returns `{"status": "ok"}` |
| `/v1/messages` | POST | **Anthropic Messages API** (what Claude Code uses) |
| `/v1/generate` | POST | Native API — generate text (JSON or SSE streaming) |
| `/metrics` | GET | Prometheus metrics (request counts, latency, tokens, cache hits) |
| `/v1/evaluate` | POST | Submit a quality score for a past request |
| `/v1/eval/stats` | GET | Aggregated stats per provider/task |
| `/v1/eval/best?task=code` | GET | Best provider for a given task type |

---

## Monitoring

### Health check

```bash
curl http://localhost:3000/health
```

### View metrics

```bash
curl http://localhost:3000/metrics
```

Shows Prometheus-format metrics:
```
llm_requests_total{provider="local-qwen",task="default",status="success"} 42
llm_request_duration_seconds{provider="local-qwen"} ...
llm_tokens_total{direction="input",provider="local-qwen"} 1234
llm_tokens_total{direction="output",provider="local-qwen"} 5678
llm_cache_hits_total 7
llm_cache_misses_total 35
```

### Check GPU usage

```bash
nvidia-smi
```

---

## Evaluation System

After generating responses, you can score them to track which models/providers perform best:

```bash
# Submit a quality score (0.0 to 1.0) for a response
curl -X POST http://localhost:3000/v1/evaluate \
  -H "Content-Type: application/json" \
  -d '{"id": "req_abc123", "score": 0.9}'

# View aggregated stats
curl http://localhost:3000/v1/eval/stats

# Find best provider for a task type
curl "http://localhost:3000/v1/eval/best?task=code"
```

---

## Caching

Identical prompts are automatically cached (in-memory LRU). A cached response returns instantly with `"cached": true`. Streaming requests are not cached.

Default: 1000 entries, 1-hour TTL. Change in `config.toml`:

```toml
[cache]
enabled = true
max_entries = 1000
ttl_secs = 3600
```

---

## Shutting Down

Press `Ctrl+C` in the terminal where llm-serve is running.

llm-serve handles graceful shutdown:
1. Stops accepting new requests
2. Drains in-flight requests
3. Kills the llama-server child process
4. Exits

Verify everything stopped:
```bash
ps aux | grep -E "llm-serve|llama-server"
```

### Force kill if needed

```bash
pkill -f llm-serve        # this also kills the child llama-server
# or by port:
sudo fuser -k 3000/tcp
sudo fuser -k 8080/tcp
```

### Shell alias for stopping

```bash
echo "alias ai-stop='pkill -f llm-serve; echo Stack stopped.'" >> ~/.bashrc
source ~/.bashrc
```

---

## Docker Deployment (No Rust Install Required)

If you don't want to install Rust on the deployment machine:

### Build the image (on dev machine or anywhere with Docker)

```bash
cd ~/llm/2servelocalllm
docker compose build
```

### Run on the deployment machine

Edit `docker-compose.yml` to set your actual paths:

```yaml
volumes:
  - /home/you/llm/llama.cpp/build/bin/llama-server:/llama/llama-server:ro
  - /home/you/llm/models:/models:ro
environment:
  - APP__LLAMA__ENABLED=true
  - APP__LLAMA__BINARY=/llama/llama-server
  - APP__LLAMA__MODEL=/models/qwen35-27b-q8.gguf
```

For GPU access, uncomment the deploy section in `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: all
          capabilities: [gpu]
```

Then:
```bash
docker compose up
```

---

## Configuration Reference

All config lives in `serve/config.toml`. Every value can be overridden via environment variables with the `APP__` prefix (double underscore for nesting):

| Config key | Env override | Default | Purpose |
|-----------|-------------|---------|---------|
| `server.port` | `APP__SERVER__PORT` | 3000 | API port |
| `llama.enabled` | `APP__LLAMA__ENABLED` | false | Spawn llama-server |
| `llama.binary` | `APP__LLAMA__BINARY` | (see config) | Path to llama-server |
| `llama.model` | `APP__LLAMA__MODEL` | (see config) | Path to model file |
| `llama.gpu_layers` | `APP__LLAMA__GPU_LAYERS` | 99 | Layers to offload to GPU |
| `llama.ctx_size` | `APP__LLAMA__CTX_SIZE` | 16384 | Context window size |
| `cache.enabled` | `APP__CACHE__ENABLED` | true | Enable response cache |
| `cache.max_entries` | `APP__CACHE__MAX_ENTRIES` | 1000 | Max cached responses |
| `cache.ttl_secs` | `APP__CACHE__TTL_SECS` | 3600 | Cache entry lifetime |
| `executor.timeout_secs` | `APP__EXECUTOR__TIMEOUT_SECS` | 120 | Per-request timeout |
| `observability.log_format` | `APP__OBSERVABILITY__LOG_FORMAT` | json | "json" or "pretty" |
| `observability.log_level` | `APP__OBSERVABILITY__LOG_LEVEL` | info | Log level filter |

---

## Port Reference

| Port | Service | Exposed? |
|------|---------|----------|
| 3000 | llm-serve (your API) | Yes — this is what apps talk to |
| 8080 | llama-server (inference) | No — managed internally by llm-serve |

---

## Troubleshooting

### llm-serve fails to start with "failed to start llama-server"

Check the `llama.binary` path in config.toml points to a valid llama-server binary:
```bash
ls -la ./llama.cpp/build/bin/llama-server
```
If not built, build llama.cpp first (see setup).

### "health check timeout" on startup

The model is taking too long to load. Increase the timeout:
```toml
[llama]
health_check_timeout_secs = 120
```

Or check that the model file exists and the GPU has enough VRAM.

### "address already in use" on port 3000 or 8080

Previous instance didn't shut down cleanly:
```bash
sudo fuser -k 3000/tcp
sudo fuser -k 8080/tcp
```

### Slow inference (< 20 tok/s)

Check GPU is being used:
```bash
nvidia-smi
```
Ensure `gpu_layers = 99` in config (offloads all layers to GPU).

### WSL2 loses GPU access after sleep/resume

```bash
sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm
```
Then restart: `ai-start`

---

## Updating llama.cpp

```bash
cd ~/llm/2servelocalllm/llama.cpp
git pull
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release -j$(nproc)
```

No changes needed to llm-serve — it just calls the binary.

---

*Setup: WSL2 2.7.0+, Ubuntu, CUDA 12.8, Rust 1.94+, llama.cpp (latest), RTX 5090 32GB*
