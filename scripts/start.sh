#!/bin/bash

echo "Starting llama-server..."
~/llm/llama.cpp/build/bin/llama-server \
  --model ~/llm/models/qwen35-27b-q8.gguf \
  --n-gpu-layers 99 \
  --port 8080 \
  --host 0.0.0.0 \
  --ctx-size 16384 &

echo "Waiting for llama-server..."
until curl -s http://localhost:8080/health > /dev/null 2>&1; do
  sleep 2
done
echo "llama-server ready!"

echo "Starting proxy..."
python3 ~/llm/scripts/proxy.py &

sleep 2
echo ""
echo "Ready. Run: claude-local"
