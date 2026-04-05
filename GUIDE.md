# LLM Serving Layer — Setup & Usage Guide
## llm-serve + llama-server

---

## How It Works

```
Any App / curl / Claude Code
      │  (standard HTTP)
      ▼
llm-serve (port 11434)             ← one binary does everything
├── Routing Engine                  decides which model handles the request
├── Executor                        retries, timeouts, fallbacks
├── Cache                           deduplicates repeated prompts
├── Metrics                         Prometheus counters + latency histograms
├── Eval Store                      tracks model quality over time
│
├── Process Manager ──→ llama-server (port 8080, spawned automatically)
│                              │
│                              ▼
│                        Your model on GPU
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
│   ├── llama.cpp/          ← inference engine (git submodule)
│   ├── models/             ← model weights (or symlinks)
│   ├── scripts/            ← legacy scripts (reference only)
│   └── docker-compose.yml  ← for Docker deployments
```

---

## Quick Start (Recommended)

The setup script handles everything — detects your OS, installs dependencies, builds llama.cpp (CUDA on Linux, Metal on macOS), and builds llm-serve:

```bash
cd ~/llm/2servelocalllm
./setup.sh
```

The script will:
1. Install system dependencies (cmake, pkg-config, libssl-dev on Linux; cmake on macOS)
2. Install Rust if not present
3. Build llama.cpp with the right GPU backend for your platform
4. Build llm-serve
5. Check for a model file in `models/`
6. Offer to add shell aliases to your `~/.bashrc` or `~/.zshrc`

After setup, place a GGUF model in `models/` and update the filename in `serve/config.toml` if needed.

---

## Supported Models

Any model in GGUF format that llama.cpp supports will work. This includes:

| Model Family | Example |
|-------------|---------|
| **Qwen** | `qwen35-27b-q8.gguf` |
| **Gemma** | `gemma-3-27b-it-Q8_0.gguf` |
| **DeepSeek** | `DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf` |
| **Llama** | `Meta-Llama-3.1-8B-Instruct-Q8_0.gguf` |
| **Mistral** | `Mistral-7B-Instruct-v0.3-Q8_0.gguf` |
| **Phi** | `Phi-4-mini-instruct-Q8_0.gguf` |

To switch models, update the path in `serve/config.toml`:

```toml
[llama]
model = "../models/your-model-file.gguf"
```

Then restart with `ai-start`. No rebuild needed.

> **Tip:** Choose a quantization that fits your VRAM. Q8_0 gives the best quality but is the largest. Q4_K_M is a good balance of quality and size.

---

## Manual Setup (No Docker)

If you prefer to do each step yourself:

### Prerequisites

| Dependency | Linux (Ubuntu/Debian) | macOS |
|-----------|----------------------|-------|
| **Rust** | `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \| sh` | Same |
| **cmake** | `sudo apt-get install cmake` | `brew install cmake` |
| **OpenSSL + pkg-config** | `sudo apt-get install pkg-config libssl-dev` | Not needed (uses system Security framework) |
| **CUDA toolkit** | Required for NVIDIA GPUs — install from [NVIDIA](https://developer.nvidia.com/cuda-downloads) | N/A (macOS uses Metal) |

After installing Rust, reload your shell:
```bash
source "$HOME/.cargo/env"
```

### 1. Build llama.cpp

Choose the command for your platform:

**Linux with NVIDIA GPU (CUDA):**
```bash
cd ~/llm/2servelocalllm/llama.cpp
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j$(nproc)
```

> Set `-DCMAKE_CUDA_ARCHITECTURES` to your GPU's compute capability if `native` doesn't work. For example: `120` for RTX 5090, `89` for RTX 4090, `86` for RTX 3090.

**macOS (Metal — Apple Silicon or Intel with AMD GPU):**
```bash
cd ~/llm/2servelocalllm/llama.cpp
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

### 2. Build llm-serve

```bash
cd ~/llm/2servelocalllm/serve
cargo build --release
```

### 3. Place your model

Put a GGUF model file (or symlink) in the `models/` directory:

```bash
ls ~/llm/2servelocalllm/models/
# e.g. qwen35-27b-q8.gguf
```

### 4. Configure

Edit `serve/config.toml`. The defaults should work if your directory layout matches the structure above. The key settings:

```toml
[llama]
enabled = true                                    # must be true on the GPU machine
binary = "../llama.cpp/build/bin/llama-server"     # relative to serve/
model = "../models/qwen35-27b-q8.gguf"             # relative to serve/
gpu_layers = 99                                    # offload all layers to GPU
ctx_size = 16384                                   # context window
```

### 5. Start

```bash
cd ~/llm/2servelocalllm/serve
./target/release/llm-serve
```

Wait for `listening on 0.0.0.0:11434` — the stack is ready.

---

## Shell Aliases (Optional)

Add to `~/.bashrc` (Linux) or `~/.zshrc` (macOS):

```bash
# Start the full stack (picks model if you have more than one)
alias ai-start='~/2servelocalllm/scripts/ai-start.sh'

# Stop the full stack
alias ai-stop='pkill -f llm-serve; echo Stack stopped.'

# Claude Code → local model (free, private, routed through llm-serve)
alias claude-local='ANTHROPIC_BASE_URL="http://localhost:11434" ANTHROPIC_AUTH_TOKEN="local-dev" ANTHROPIC_API_KEY="" claude'

# Claude Code → Anthropic cloud (requires API key)
alias claude-cloud='unset ANTHROPIC_BASE_URL; unset ANTHROPIC_AUTH_TOKEN; claude'
```

Then reload: `source ~/.bashrc` (or `source ~/.zshrc`)

> **Note:** `setup.sh` offers to add these automatically. If you've already run setup, they may already be in your shell config.

### Daily workflow

```bash
# Terminal 1: start the stack
ai-start
# Wait for "listening on 0.0.0.0:11434"

# Terminal 2: use Claude Code with your local model
cd ~/your-project
claude-local
```

---

## Docker Setup (No Rust Install Required)

Docker builds llm-serve inside the container, so you don't need Rust on the host. You still need llama-server and model files on the host (mounted in).

### Prerequisites

| Dependency | Required for |
|-----------|-------------|
| **Docker + Docker Compose** | Building and running the container |
| **NVIDIA Container Toolkit** | GPU passthrough (Linux only) — [install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |

> **macOS note:** Docker on macOS cannot pass through Metal GPU access. For macOS with Apple Silicon, use the native setup above instead of Docker.

### 1. Update docker-compose.yml

Edit `docker-compose.yml` with your actual host paths:

```yaml
services:
  llm-serve:
    build:
      context: ./serve
      dockerfile: Dockerfile
    ports:
      - "11434:11434"
    volumes:
      # Mount the llama-server binary from the host
      - /home/you/llm/2servelocalllm/llama.cpp/build/bin:/llama/bin:ro
      # Mount model files from the host
      - /home/you/llm/2servelocalllm/models:/models:ro
    environment:
      - APP__SERVER__PORT=11434
      - APP__LLAMA__ENABLED=true
      - APP__LLAMA__BINARY=/llama/bin/llama-server
      - APP__LLAMA__MODEL=/models/qwen35-27b-q8.gguf
    # NVIDIA GPU passthrough (Linux only):
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
```

> **Important:** Mount the entire `bin/` directory (not just the binary) so that shared libraries like `libmtmd.so` are available next to `llama-server`.

### 2. Build and run

```bash
cd ~/llm/2servelocalllm
docker compose up --build
```

### 3. Stop

```bash
docker compose down
```

---

## Using It

### Health check

```bash
curl http://localhost:11434/health
```

### Basic generation

```bash
curl -X POST http://localhost:11434/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain what a hash table is in one paragraph.",
    "max_tokens": 200
  }'
```

### Chat-style messages

```bash
curl -X POST http://localhost:11434/v1/generate \
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
curl -N -X POST http://localhost:11434/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short poem about programming.",
    "stream": true,
    "max_tokens": 200
  }'
```

### Anthropic Messages API (used by Claude Code)

```bash
curl -X POST http://localhost:11434/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: local-dev" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "qwen35-distilled",
    "max_tokens": 100,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

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

### View metrics

```bash
curl http://localhost:11434/metrics
```

### Check GPU usage

```bash
nvidia-smi          # Linux
# or
sudo powermetrics --samplers gpu_power -i 1000  # macOS (Apple Silicon)
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

### Force kill if needed

```bash
pkill -f llm-serve        # this also kills the child llama-server
# or by port:
sudo fuser -k 11434/tcp   # Linux
sudo fuser -k 8080/tcp    # Linux
lsof -ti:11434 | xargs kill  # macOS
lsof -ti:8080 | xargs kill   # macOS
```

---

## Configuration Reference

All config lives in `serve/config.toml`. Every value can be overridden via environment variables with the `APP__` prefix (double underscore for nesting):

| Config key | Env override | Default | Purpose |
|-----------|-------------|---------|---------|
| `server.port` | `APP__SERVER__PORT` | 11434 | API port |
| `llama.enabled` | `APP__LLAMA__ENABLED` | true | Spawn llama-server |
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
| 11434 | llm-serve (your API) | Yes — this is what apps talk to |
| 8080 | llama-server (inference) | No — managed internally by llm-serve |

---

## Troubleshooting

### llm-serve fails to start with "failed to start llama-server"

Check the `llama.binary` path in config.toml points to a valid llama-server binary:
```bash
ls -la ../llama.cpp/build/bin/llama-server
```
If not built, build llama.cpp first (see setup).

### "health check timeout" on startup

The model is taking too long to load. Increase the timeout:
```toml
[llama]
health_check_timeout_secs = 120
```

Or check that the model file exists and the GPU has enough VRAM.

### "address already in use" on port 11434 or 8080

Previous instance didn't shut down cleanly:
```bash
# Linux
sudo fuser -k 11434/tcp
sudo fuser -k 8080/tcp

# macOS
lsof -ti:11434 | xargs kill
lsof -ti:8080 | xargs kill
```

### "cannot open shared object file: libmtmd.so" (Linux)

This is handled automatically by llm-serve (it sets `LD_LIBRARY_PATH` to the binary's directory). If you're running llama-server manually, set it yourself:
```bash
export LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin:$LD_LIBRARY_PATH
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

**Linux (CUDA):**
```bash
cd ~/llm/2servelocalllm/llama.cpp
git pull
rm -rf build
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --config Release -j$(nproc)
```

**macOS (Metal):**
```bash
cd ~/llm/2servelocalllm/llama.cpp
git pull
rm -rf build
cmake -B build -DGGML_METAL=ON
cmake --build build --config Release -j$(sysctl -n hw.ncpu)
```

No changes needed to llm-serve — it just calls the binary.
