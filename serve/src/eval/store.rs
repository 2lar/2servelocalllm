use std::collections::HashMap;

use tokio::sync::RwLock;

use crate::error::ServeError;

use super::{EvalRecord, EvalStats};

pub struct EvalStore {
    records: RwLock<Vec<EvalRecord>>,
    max_records: usize,
}

impl EvalStore {
    pub fn new(max_records: usize) -> Self {
        Self {
            records: RwLock::new(Vec::new()),
            max_records,
        }
    }

    pub async fn record(&self, record: EvalRecord) {
        let mut records = self.records.write().await;
        if records.len() >= self.max_records {
            records.remove(0);
        }
        records.push(record);
    }

    pub async fn set_score(&self, id: &str, score: f64) -> Result<(), ServeError> {
        let mut records = self.records.write().await;
        let record = records
            .iter_mut()
            .find(|r| r.id == id)
            .ok_or_else(|| ServeError::Internal(format!("eval record not found: {id}")))?;
        record.score = Some(score);
        Ok(())
    }

    pub async fn stats(&self) -> Vec<EvalStats> {
        let records = self.records.read().await;

        // Group by (provider, task).
        let mut groups: HashMap<(String, Option<String>), Vec<&EvalRecord>> = HashMap::new();
        for record in records.iter() {
            let key = (record.provider.clone(), record.task.clone());
            groups.entry(key).or_default().push(record);
        }

        let mut stats: Vec<EvalStats> = groups
            .into_iter()
            .map(|((provider, task), records)| {
                let request_count = records.len() as u64;
                let avg_latency_ms =
                    records.iter().map(|r| r.latency_ms as f64).sum::<f64>() / request_count as f64;

                let scored: Vec<f64> = records.iter().filter_map(|r| r.score).collect();
                let avg_score = if scored.is_empty() {
                    None
                } else {
                    Some(scored.iter().sum::<f64>() / scored.len() as f64)
                };

                EvalStats {
                    provider,
                    task,
                    avg_score,
                    avg_latency_ms,
                    request_count,
                }
            })
            .collect();

        stats.sort_by(|a, b| {
            a.provider
                .cmp(&b.provider)
                .then_with(|| a.task.cmp(&b.task))
        });

        stats
    }

    pub async fn best_provider_for_task(&self, task: &str) -> Option<String> {
        let stats = self.stats().await;

        stats
            .iter()
            .filter(|s| s.task.as_deref() == Some(task))
            .filter_map(|s| s.avg_score.map(|score| (&s.provider, score)))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(provider, _)| provider.clone())
    }

    pub async fn recent(&self, limit: usize) -> Vec<EvalRecord> {
        let records = self.records.read().await;
        records
            .iter()
            .rev()
            .take(limit)
            .cloned()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use chrono::Utc;

    use crate::api::types::{GenerateRequest, GenerateResponse, Usage};

    use super::*;

    fn make_request(task: Option<&str>) -> GenerateRequest {
        GenerateRequest {
            prompt: Some("test".to_string()),
            messages: None,
            task: task.map(String::from),
            max_tokens: None,
            temperature: None,
            stream: None,
            provider: None,
        }
    }

    fn make_response(id: &str, provider: &str) -> GenerateResponse {
        GenerateResponse {
            id: id.to_string(),
            output: "test output".to_string(),
            model: "test-model".to_string(),
            provider: provider.to_string(),
            latency_ms: 100,
            usage: Usage {
                input_tokens: 10,
                output_tokens: 5,
            },
            cached: false,
            routing: None,
        }
    }

    fn make_record(id: &str, provider: &str, task: Option<&str>, latency_ms: u64) -> EvalRecord {
        EvalRecord {
            id: id.to_string(),
            request: make_request(task),
            response: make_response(id, provider),
            provider: provider.to_string(),
            task: task.map(String::from),
            latency_ms,
            score: None,
            created_at: Utc::now(),
        }
    }

    #[tokio::test]
    async fn record_and_retrieve() {
        let store = EvalStore::new(100);
        let record = make_record("r1", "provider-a", Some("code"), 100);

        store.record(record).await;

        let recent = store.recent(10).await;
        assert_eq!(recent.len(), 1);
        assert_eq!(recent[0].id, "r1");
        assert_eq!(recent[0].provider, "provider-a");
    }

    #[tokio::test]
    async fn set_score_updates_record() {
        let store = EvalStore::new(100);
        store.record(make_record("r1", "prov", Some("code"), 50)).await;

        assert!(store.recent(1).await[0].score.is_none());

        store.set_score("r1", 0.85).await.unwrap();

        let recent = store.recent(1).await;
        assert_eq!(recent[0].score, Some(0.85));
    }

    #[tokio::test]
    async fn set_score_returns_error_for_missing_id() {
        let store = EvalStore::new(100);
        let result = store.set_score("nonexistent", 0.5).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn stats_aggregation_multiple_providers_and_tasks() {
        let store = EvalStore::new(100);

        // Provider A, task "code": 2 records
        let mut r1 = make_record("r1", "prov-a", Some("code"), 100);
        r1.score = Some(0.8);
        store.record(r1).await;

        let mut r2 = make_record("r2", "prov-a", Some("code"), 200);
        r2.score = Some(0.6);
        store.record(r2).await;

        // Provider B, task "code": 1 record
        let mut r3 = make_record("r3", "prov-b", Some("code"), 150);
        r3.score = Some(0.9);
        store.record(r3).await;

        // Provider A, task "chat": 1 record, no score
        store.record(make_record("r4", "prov-a", Some("chat"), 50)).await;

        let stats = store.stats().await;
        assert_eq!(stats.len(), 3);

        // Stats are sorted by (provider, task).
        let a_chat = stats.iter().find(|s| s.provider == "prov-a" && s.task.as_deref() == Some("chat")).unwrap();
        assert_eq!(a_chat.request_count, 1);
        assert_eq!(a_chat.avg_latency_ms, 50.0);
        assert!(a_chat.avg_score.is_none());

        let a_code = stats.iter().find(|s| s.provider == "prov-a" && s.task.as_deref() == Some("code")).unwrap();
        assert_eq!(a_code.request_count, 2);
        assert_eq!(a_code.avg_latency_ms, 150.0);
        assert!((a_code.avg_score.unwrap() - 0.7).abs() < f64::EPSILON);

        let b_code = stats.iter().find(|s| s.provider == "prov-b" && s.task.as_deref() == Some("code")).unwrap();
        assert_eq!(b_code.request_count, 1);
        assert_eq!(b_code.avg_score.unwrap(), 0.9);
    }

    #[tokio::test]
    async fn best_provider_returns_highest_scored() {
        let store = EvalStore::new(100);

        let mut r1 = make_record("r1", "prov-a", Some("code"), 100);
        r1.score = Some(0.7);
        store.record(r1).await;

        let mut r2 = make_record("r2", "prov-b", Some("code"), 100);
        r2.score = Some(0.9);
        store.record(r2).await;

        let mut r3 = make_record("r3", "prov-c", Some("code"), 100);
        r3.score = Some(0.8);
        store.record(r3).await;

        let best = store.best_provider_for_task("code").await;
        assert_eq!(best.as_deref(), Some("prov-b"));
    }

    #[tokio::test]
    async fn best_provider_returns_none_when_no_scores() {
        let store = EvalStore::new(100);
        store.record(make_record("r1", "prov-a", Some("code"), 100)).await;

        let best = store.best_provider_for_task("code").await;
        assert!(best.is_none());
    }

    #[tokio::test]
    async fn best_provider_returns_none_for_unknown_task() {
        let store = EvalStore::new(100);

        let mut r1 = make_record("r1", "prov-a", Some("code"), 100);
        r1.score = Some(0.9);
        store.record(r1).await;

        let best = store.best_provider_for_task("chat").await;
        assert!(best.is_none());
    }

    #[tokio::test]
    async fn max_records_eviction_removes_oldest() {
        let store = EvalStore::new(3);

        store.record(make_record("r1", "prov", None, 100)).await;
        store.record(make_record("r2", "prov", None, 100)).await;
        store.record(make_record("r3", "prov", None, 100)).await;

        // Store is full. Adding a 4th should evict r1.
        store.record(make_record("r4", "prov", None, 100)).await;

        let recent = store.recent(10).await;
        assert_eq!(recent.len(), 3);

        let ids: Vec<&str> = recent.iter().map(|r| r.id.as_str()).collect();
        assert!(!ids.contains(&"r1"), "r1 should have been evicted");
        assert!(ids.contains(&"r2"));
        assert!(ids.contains(&"r3"));
        assert!(ids.contains(&"r4"));
    }

    #[tokio::test]
    async fn recent_returns_last_n_in_reverse_order() {
        let store = EvalStore::new(100);

        store.record(make_record("r1", "prov", None, 100)).await;
        store.record(make_record("r2", "prov", None, 100)).await;
        store.record(make_record("r3", "prov", None, 100)).await;
        store.record(make_record("r4", "prov", None, 100)).await;
        store.record(make_record("r5", "prov", None, 100)).await;

        let recent = store.recent(3).await;
        assert_eq!(recent.len(), 3);
        assert_eq!(recent[0].id, "r5");
        assert_eq!(recent[1].id, "r4");
        assert_eq!(recent[2].id, "r3");
    }

    #[tokio::test]
    async fn recent_returns_all_when_limit_exceeds_count() {
        let store = EvalStore::new(100);

        store.record(make_record("r1", "prov", None, 100)).await;
        store.record(make_record("r2", "prov", None, 100)).await;

        let recent = store.recent(10).await;
        assert_eq!(recent.len(), 2);
    }
}
