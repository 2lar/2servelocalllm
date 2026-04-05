use std::collections::HashMap;
use std::sync::Arc;

use super::Provider;

pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn Provider>>,
}

impl ProviderRegistry {
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    pub fn register(&mut self, name: String, provider: Arc<dyn Provider>) {
        self.providers.insert(name, provider);
    }

    pub fn get(&self, name: &str) -> Option<Arc<dyn Provider>> {
        self.providers.get(name).cloned()
    }

    pub fn list_names(&self) -> Vec<&str> {
        self.providers.keys().map(|s| s.as_str()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::mock::MockProvider;
    use std::time::Duration;

    #[test]
    fn register_and_get_provider() {
        let mut registry = ProviderRegistry::new();
        let provider = Arc::new(MockProvider::new(Duration::from_millis(0)));
        registry.register("mock".to_string(), provider);

        let retrieved = registry.get("mock");
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().name(), "mock");
    }

    #[test]
    fn get_missing_provider_returns_none() {
        let registry = ProviderRegistry::new();
        assert!(registry.get("nonexistent").is_none());
    }

    #[test]
    fn list_names_returns_registered_providers() {
        let mut registry = ProviderRegistry::new();
        let p1 = Arc::new(MockProvider::new(Duration::from_millis(0)));
        let p2 = Arc::new(MockProvider::new(Duration::from_millis(0)));
        registry.register("alpha".to_string(), p1);
        registry.register("beta".to_string(), p2);

        let mut names = registry.list_names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
    }

    #[test]
    fn list_names_empty_registry() {
        let registry = ProviderRegistry::new();
        assert!(registry.list_names().is_empty());
    }
}
