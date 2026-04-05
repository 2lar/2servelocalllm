#!/usr/bin/env python3
"""
Simple Anthropic -> OpenAI proxy for Claude Code + llama-server
"""
import json
import httpx
from http.server import HTTPServer, BaseHTTPRequestHandler

LLAMA_SERVER = "http://localhost:8080"
MODEL_NAME = "qwen35-distilled"

class ProxyHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"status": "ok"}).encode())

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length))

        messages = []
        if "system" in body:
            system = body["system"]
            if isinstance(system, list):
                system = " ".join(b.get("text", "") for b in system if isinstance(b, dict))
            messages.append({"role": "system", "content": system})

        for msg in body.get("messages", []):
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    b.get("text", "") for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            messages.append({"role": role, "content": content})

        openai_body = {
            "model": MODEL_NAME,
            "messages": messages,
            "max_tokens": body.get("max_tokens", 4096),
            "stream": body.get("stream", False),
            "temperature": body.get("temperature", 0.6),
        }

        stream = openai_body["stream"]

        try:
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

                input_tokens = sum(len(m["content"].split()) for m in messages)

                self.wfile.write(b"event: message_start\n")
                self.wfile.write(json.dumps({"type": "message_start", "message": {
                    "id": "msg_local", "type": "message", "role": "assistant",
                    "content": [], "model": "claude-local",
                    "usage": {"input_tokens": input_tokens, "output_tokens": 0}
                }}).encode() + b"\n\n")
                self.wfile.write(b"event: content_block_start\n")
                self.wfile.write(json.dumps({"type": "content_block_start",
                    "index": 0, "content_block": {"type": "text", "text": ""}
                }).encode() + b"\n\n")
                self.wfile.flush()

                output_tokens = 0
                with httpx.stream("POST", f"{LLAMA_SERVER}/v1/chat/completions",
                                  json=openai_body, timeout=600) as r:
                    for line in r.iter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            text = delta.get("content", "")
                            if text:
                                output_tokens += 1
                                self.wfile.write(b"event: content_block_delta\n")
                                self.wfile.write(json.dumps({"type": "content_block_delta",
                                    "index": 0, "delta": {"type": "text_delta", "text": text}
                                }).encode() + b"\n\n")
                                self.wfile.flush()
                        except Exception:
                            continue

                self.wfile.write(b"event: content_block_stop\n")
                self.wfile.write(json.dumps({"type": "content_block_stop", "index": 0}).encode() + b"\n\n")
                self.wfile.write(b"event: message_delta\n")
                self.wfile.write(json.dumps({"type": "message_delta",
                    "delta": {"stop_reason": "end_turn"},
                    "usage": {"output_tokens": output_tokens}
                }).encode() + b"\n\n")
                self.wfile.write(b"event: message_stop\n")
                self.wfile.write(json.dumps({"type": "message_stop"}).encode() + b"\n\n")
                self.wfile.flush()

            else:
                r = httpx.post(f"{LLAMA_SERVER}/v1/chat/completions",
                               json=openai_body, timeout=600)
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                input_tokens = sum(len(m["content"].split()) for m in messages)
                response = {
                    "id": "msg_local",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": text}],
                    "model": "claude-local",
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": input_tokens, "output_tokens": len(text.split())}
                }
                body = json.dumps(response).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        except Exception as e:
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode())

if __name__ == "__main__":
    print("Starting Anthropic proxy on port 11434...")
    print(f"Forwarding to llama-server at {LLAMA_SERVER}")
    server = HTTPServer(("0.0.0.0", 11434), ProxyHandler)
    server.serve_forever()
