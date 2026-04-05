# -- Builder stage --
FROM rust:1-slim-bookworm AS builder

WORKDIR /build

# Copy manifests first for better layer caching
COPY Cargo.toml Cargo.lock ./

# Create a dummy main to cache dependency compilation
RUN mkdir src && echo "fn main() {}" > src/main.rs && cargo build --release && rm -rf src

# Copy real source and rebuild
COPY src/ src/
RUN touch src/main.rs && cargo build --release

# -- Runtime stage --
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/target/release/llm-serve /usr/local/bin/llm-serve
COPY config.toml /etc/llm-serve/config.toml

ENV APP__SERVER__HOST=0.0.0.0

EXPOSE 3000

ENTRYPOINT ["llm-serve"]
