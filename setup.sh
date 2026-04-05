#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_DIR="$REPO_DIR/llama.cpp"
SERVE_DIR="$REPO_DIR/serve"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}[+]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[x]${NC} $*"; }

# ──────────────────────────────────────────────
# Detect OS
# ──────────────────────────────────────────────
OS="$(uname -s)"
ARCH="$(uname -m)"

case "$OS" in
    Linux)  PLATFORM="linux" ;;
    Darwin) PLATFORM="macos" ;;
    *)      error "Unsupported OS: $OS"; exit 1 ;;
esac

info "Detected platform: $PLATFORM ($ARCH)"

# ──────────────────────────────────────────────
# Step 1: System dependencies
# ──────────────────────────────────────────────
info "Checking system dependencies..."

install_linux_deps() {
    local missing=()
    command -v cmake  >/dev/null 2>&1 || missing+=(cmake)
    command -v pkg-config >/dev/null 2>&1 || missing+=(pkg-config)
    # Check for libssl-dev by looking for the pkg-config file
    pkg-config --exists openssl 2>/dev/null || missing+=(libssl-dev)

    if [ ${#missing[@]} -gt 0 ]; then
        info "Installing missing packages: ${missing[*]}"
        sudo apt-get update -qq
        sudo apt-get install -y --no-install-recommends "${missing[@]}"
    else
        info "System dependencies OK"
    fi
}

install_macos_deps() {
    if ! command -v brew >/dev/null 2>&1; then
        error "Homebrew not found. Install it from https://brew.sh"
        exit 1
    fi

    if ! command -v cmake >/dev/null 2>&1; then
        info "Installing cmake..."
        brew install cmake
    else
        info "System dependencies OK"
    fi
}

case "$PLATFORM" in
    linux) install_linux_deps ;;
    macos) install_macos_deps ;;
esac

# Check for CUDA on Linux (needed for GPU acceleration)
if [ "$PLATFORM" = "linux" ]; then
    if ! command -v nvcc >/dev/null 2>&1 && ! [ -d /usr/local/cuda ]; then
        warn "CUDA toolkit not found!"
        warn "Without CUDA, llama.cpp will build CPU-only (much slower inference)."
        warn "To install CUDA: https://developer.nvidia.com/cuda-downloads"
        warn "After installing, you may need to reboot and re-run this script."
        echo ""
        echo "Continue without CUDA (CPU-only)? [y/N]"
        read -r answer
        if [[ ! "$answer" =~ ^[Yy]$ ]]; then
            error "Install CUDA first, then re-run ./setup.sh"
            exit 1
        fi
    else
        info "CUDA toolkit found"
    fi
fi

# ──────────────────────────────────────────────
# Step 2: Rust toolchain
# ──────────────────────────────────────────────
if ! command -v cargo >/dev/null 2>&1; then
    info "Installing Rust..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
else
    info "Rust already installed ($(rustc --version))"
fi

# ──────────────────────────────────────────────
# Step 3: Build llama.cpp
# ──────────────────────────────────────────────
if [ ! -d "$LLAMA_DIR" ]; then
    error "llama.cpp directory not found at $LLAMA_DIR"
    error "Clone it first: git clone https://github.com/ggml-org/llama.cpp.git $LLAMA_DIR"
    exit 1
fi

build_llama() {
    info "Building llama.cpp..."
    cd "$LLAMA_DIR"

    local cmake_args=(-B build -DCMAKE_BUILD_TYPE=Release)

    if [ "$PLATFORM" = "linux" ]; then
        if command -v nvcc >/dev/null 2>&1; then
            info "CUDA detected — building with GPU support"
            cmake_args+=(-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native)
        else
            warn "No CUDA found — building CPU-only"
        fi
    elif [ "$PLATFORM" = "macos" ]; then
        info "macOS — building with Metal support"
        cmake_args+=(-DGGML_METAL=ON)
    fi

    cmake "${cmake_args[@]}"

    local nproc_cmd
    if [ "$PLATFORM" = "macos" ]; then
        nproc_cmd=$(sysctl -n hw.ncpu)
    else
        nproc_cmd=$(nproc)
    fi

    cmake --build build --config Release -j"$nproc_cmd"
    cd "$REPO_DIR"
}

if [ ! -f "$LLAMA_DIR/build/bin/llama-server" ]; then
    build_llama
else
    info "llama.cpp already built"
fi

# ──────────────────────────────────────────────
# Step 4: Build llm-serve
# ──────────────────────────────────────────────
info "Building llm-serve..."
cd "$SERVE_DIR"
cargo build --release
cd "$REPO_DIR"

# ──────────────────────────────────────────────
# Step 5: Check for a model
# ──────────────────────────────────────────────
MODEL_DIR="$REPO_DIR/models"
mkdir -p "$MODEL_DIR"

MODEL_COUNT=$(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" 2>/dev/null | wc -l)
if [ "$MODEL_COUNT" -eq 0 ]; then
    warn "No .gguf model found in $MODEL_DIR"
    warn "Download any GGUF model and place it there. Any model llama.cpp supports works:"
    warn "  Qwen, Gemma, DeepSeek, Llama, Mistral, Phi, etc."
    warn ""
    warn "Examples:"
    warn "  huggingface-cli download unsloth/Qwen3.5-27B-GGUF qwen35-27b-q8.gguf --local-dir $MODEL_DIR"
    warn "  huggingface-cli download bartowski/gemma-3-27b-it-GGUF gemma-3-27b-it-Q8_0.gguf --local-dir $MODEL_DIR"
    warn "  huggingface-cli download bartowski/DeepSeek-R1-0528-Qwen3-8B-GGUF DeepSeek-R1-0528-Qwen3-8B-Q8_0.gguf --local-dir $MODEL_DIR"
    warn ""
    warn "Then update serve/config.toml with the model filename:"
    warn "  model = \"../models/your-model.gguf\""
else
    info "Found model(s) in $MODEL_DIR"
fi

# ──────────────────────────────────────────────
# Step 6: Suggest shell aliases
# ──────────────────────────────────────────────
BINARY="$SERVE_DIR/target/release/llm-serve"

if [ "$PLATFORM" = "macos" ]; then
    SHELL_RC="$HOME/.zshrc"
else
    SHELL_RC="$HOME/.bashrc"
fi

echo ""
info "Setup complete!"
echo ""

if ! grep -q "ai-start" "$SHELL_RC" 2>/dev/null; then
    echo "Add these aliases to $SHELL_RC? [y/N]"
    read -r answer
    if [[ "$answer" =~ ^[Yy]$ ]]; then
        cat >> "$SHELL_RC" << EOF

# llm-serve aliases
alias ai-start='$REPO_DIR/scripts/ai-start.sh'
alias ai-stop='pkill -f llm-serve; echo "Stack stopped."'
alias claude-local='ANTHROPIC_BASE_URL="http://localhost:11434" ANTHROPIC_AUTH_TOKEN="local-dev" ANTHROPIC_API_KEY="" claude'
alias claude-cloud='unset ANTHROPIC_BASE_URL; unset ANTHROPIC_AUTH_TOKEN; claude'
EOF
        info "Aliases added to $SHELL_RC"
        info "Run: source $SHELL_RC"
    fi
else
    info "Shell aliases already present in $SHELL_RC"
fi

echo ""
info "To start: ai-start  (or: cd $SERVE_DIR && $BINARY)"
info "To use with Claude Code: claude-local"
