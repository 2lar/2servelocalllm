use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use lru::LruCache;
use tokio::sync::Mutex;

use crate::api::types::{GenerateRequest, GenerateResponse};
use crate::config::CacheConfig;

use super::{cache_key, Cache};

struct CacheEntry {
    response: GenerateResponse,
    inserted_at: Instant,
}

pub struct MemoryCache {
    store: Mutex<LruCache<String, CacheEntry>>,
    ttl: Duration,
}

impl MemoryCache {
    pub fn new(config: &CacheConfig) -> Self {
        let cap = NonZeroUsize::new(config.max_entries).unwrap_or(NonZeroUsize::new(1).unwrap());
        Self {
            store: Mutex::new(LruCache::new(cap)),
            ttl: Duration::from_secs(config.ttl_secs),
        }
    }
}

#[async_trait]
impl Cache for MemoryCache {
    async fn get(&self, req: &GenerateRequest) -> Option<GenerateResponse> {
        let key = cache_key(req);
        let mut store = self.store.lock().await;

        let entry = store.get(&key)?;
        if entry.inserted_at.elapsed() > self.ttl {
            store.pop(&key);
            return None;
        }

        Some(entry.response.clone())
    }

    async fn put(&self, req: &GenerateRequest, resp: &GenerateResponse) {
        let key = cache_key(req);
        let entry = CacheEntry {
            response: resp.clone(),
            inserted_at: Instant::now(),
        };
        let mut store = self.store.lock().await;
        store.put(key, entry);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::types::{GenerateRequest, GenerateResponse, Usage};

    fn test_config(max_entries: usize, ttl_secs: u64) -> CacheConfig {
        CacheConfig {
            enabled: true,
            max_entries,
            ttl_secs,
        }
    }

    fn make_request(prompt: &str) -> GenerateRequest {
        GenerateRequest {
            prompt: Some(prompt.to_string()),
            messages: None,
            task: None,
            max_tokens: Some(100),
            temperature: Some(0.7),
            stream: None,
            provider: None,
        }
    }

    fn make_response(output: &str) -> GenerateResponse {
        GenerateResponse {
            id: "test-id".to_string(),
            output: output.to_string(),
            model: "test-model".to_string(),
            provider: "test-provider".to_string(),
            latency_ms: 42,
            usage: Usage {
                input_tokens: 5,
                output_tokens: 10,
            },
            cached: false,
            routing: None,
        }
    }

    #[tokio::test]
    async fn put_then_get_returns_cached_response() {
        let cache = MemoryCache::new(&test_config(10, 3600));
        let req = make_request("Hello");
        let resp = make_response("World");

        cache.put(&req, &resp).await;
        let cached = cache.get(&req).await;

        assert!(cached.is_some());
        let cached = cached.unwrap();
        assert_eq!(cached.output, "World");
        assert_eq!(cached.id, "test-id");
    }

    #[tokio::test]
    async fn get_different_request_returns_none() {
        let cache = MemoryCache::new(&test_config(10, 3600));
        let req1 = make_request("Hello");
        let req2 = make_request("Different");
        let resp = make_response("World");

        cache.put(&req1, &resp).await;
        let cached = cache.get(&req2).await;

        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn ttl_expiry_returns_none() {
        // Use a TTL of 0 seconds so entries expire immediately.
        let cache = MemoryCache::new(&test_config(10, 0));
        let req = make_request("Hello");
        let resp = make_response("World");

        cache.put(&req, &resp).await;
        // Even with TTL=0, Instant::now().elapsed() might be 0.
        // Sleep briefly to guarantee expiry.
        tokio::time::sleep(Duration::from_millis(5)).await;
        let cached = cache.get(&req).await;

        assert!(cached.is_none());
    }

    #[tokio::test]
    async fn lru_eviction_removes_oldest_entry() {
        let cache = MemoryCache::new(&test_config(2, 3600));

        let req_a = make_request("A");
        let req_b = make_request("B");
        let req_c = make_request("C");

        cache.put(&req_a, &make_response("resp-A")).await;
        cache.put(&req_b, &make_response("resp-B")).await;
        // Cache is full (2 entries). Inserting C should evict A (least recently used).
        cache.put(&req_c, &make_response("resp-C")).await;

        assert!(cache.get(&req_a).await.is_none(), "A should be evicted");
        assert!(cache.get(&req_b).await.is_some(), "B should still be cached");
        assert!(cache.get(&req_c).await.is_some(), "C should still be cached");
    }

    #[tokio::test]
    async fn cache_key_determinism() {
        let cache = MemoryCache::new(&test_config(10, 3600));
        let req = make_request("Deterministic");
        let resp = make_response("Response");

        cache.put(&req, &resp).await;

        // Create an identical request — should hit the same cache entry.
        let req2 = make_request("Deterministic");
        let cached = cache.get(&req2).await;
        assert!(cached.is_some());
        assert_eq!(cached.unwrap().output, "Response");
    }
}
