#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SERVE_DIR="$REPO_DIR/serve"
MODEL_DIR="$REPO_DIR/models"
BINARY="$SERVE_DIR/target/release/llm-serve"

if [ ! -f "$BINARY" ]; then
    echo "llm-serve not built yet. Run ./setup.sh first."
    exit 1
fi

# Collect all .gguf files
models=()
while IFS= read -r f; do
    models+=("$f")
done < <(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" -printf '%f\n' 2>/dev/null | sort)

if [ ${#models[@]} -eq 0 ]; then
    echo "No .gguf models found in $MODEL_DIR"
    echo "Download a model first. See GUIDE.md for examples."
    exit 1
fi

# If only one model, use it directly
if [ ${#models[@]} -eq 1 ]; then
    selected="${models[0]}"
    echo "Using model: $selected"
else
    # Interactive model picker
    echo ""
    echo "Available models:"
    echo ""
    for i in "${!models[@]}"; do
        echo "  $((i + 1))) ${models[$i]}"
    done
    echo ""
    read -rp "Select model [1]: " choice
    choice="${choice:-1}"

    if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt ${#models[@]} ]; then
        echo "Invalid selection"
        exit 1
    fi

    selected="${models[$((choice - 1))]}"
    echo ""
    echo "Using model: $selected"
fi

echo ""
cd "$SERVE_DIR"
APP__LLAMA__MODEL="../models/$selected" exec "$BINARY"
