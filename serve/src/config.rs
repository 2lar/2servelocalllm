use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Deserialize, Clone)]
pub struct AppConfig {
    pub server: ServerConfig,
    pub llama: LlamaConfig,
    pub providers: HashMap<String, ProviderConfig>,
    #[serde(default)]
    pub routing: Option<RoutingConfig>,
    #[serde(default)]
    pub executor: ExecutorConfig,
    #[serde(default)]
    pub observability: ObservabilityConfig,
    #[serde(default)]
    pub cache: CacheConfig,
    #[serde(default)]
    pub eval: EvalConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ObservabilityConfig {
    #[serde(default = "default_log_format")]
    pub log_format: String,
    #[serde(default = "default_log_level")]
    pub log_level: String,
}

impl Default for ObservabilityConfig {
    fn default() -> Self {
        Self {
            log_format: default_log_format(),
            log_level: default_log_level(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct CacheConfig {
    #[serde(default = "default_cache_enabled")]
    pub enabled: bool,
    #[serde(default = "default_cache_max_entries")]
    pub max_entries: usize,
    #[serde(default = "default_cache_ttl_secs")]
    pub ttl_secs: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: default_cache_enabled(),
            max_entries: default_cache_max_entries(),
            ttl_secs: default_cache_ttl_secs(),
        }
    }
}

fn default_cache_enabled() -> bool {
    true
}

fn default_cache_max_entries() -> usize {
    1000
}

fn default_cache_ttl_secs() -> u64 {
    3600
}

#[derive(Debug, Deserialize, Clone)]
pub struct EvalConfig {
    #[serde(default = "default_eval_enabled")]
    pub enabled: bool,
    #[serde(default = "default_eval_max_records")]
    pub max_records: usize,
}

impl Default for EvalConfig {
    fn default() -> Self {
        Self {
            enabled: default_eval_enabled(),
            max_records: default_eval_max_records(),
        }
    }
}

fn default_eval_enabled() -> bool {
    true
}

fn default_eval_max_records() -> usize {
    10000
}

fn default_log_format() -> String {
    "json".to_string()
}

fn default_log_level() -> String {
    "info".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct ExecutorConfig {
    #[serde(default = "default_timeout_secs")]
    pub timeout_secs: u64,
    #[serde(default)]
    pub retry: RetryConfig,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            timeout_secs: default_timeout_secs(),
            retry: RetryConfig::default(),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
pub struct RetryConfig {
    #[serde(default = "default_max_retries")]
    pub max_retries: u32,
    #[serde(default = "default_initial_backoff_ms")]
    pub initial_backoff_ms: u64,
    #[serde(default = "default_max_backoff_ms")]
    pub max_backoff_ms: u64,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: default_max_retries(),
            initial_backoff_ms: default_initial_backoff_ms(),
            max_backoff_ms: default_max_backoff_ms(),
        }
    }
}

fn default_timeout_secs() -> u64 {
    120
}

fn default_max_retries() -> u32 {
    2
}

fn default_initial_backoff_ms() -> u64 {
    500
}

fn default_max_backoff_ms() -> u64 {
    5000
}

#[derive(Debug, Deserialize, Clone)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LlamaConfig {
    pub enabled: bool,
    pub binary: String,
    pub model: String,
    pub host: String,
    pub port: u16,
    pub gpu_layers: u32,
    pub ctx_size: u32,
    pub health_check_timeout_secs: u64,
    pub health_check_interval_ms: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ProviderConfig {
    pub name: String,
    pub url: String,
    pub model: String,
    pub timeout_secs: u64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RoutingConfig {
    #[serde(default = "default_routing_strategy")]
    pub strategy: String,
    #[serde(default)]
    pub default_provider: Option<String>,
    #[serde(default)]
    pub rules: Vec<RoutingRule>,
    #[serde(default)]
    pub advanced: Option<AdvancedRoutingConfig>,
}

fn default_routing_strategy() -> String {
    "rule_based".to_string()
}

#[derive(Debug, Deserialize, Clone)]
pub struct AdvancedRoutingConfig {
    #[serde(default)]
    pub length_rules: Vec<LengthRule>,
    #[serde(default)]
    pub keyword_rules: Vec<KeywordRule>,
    #[serde(default)]
    pub load_balance: Option<LoadBalanceConfig>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LengthRule {
    pub max_chars: usize,
    pub provider: String,
    #[serde(default)]
    pub fallbacks: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct KeywordRule {
    pub pattern: String,
    pub provider: String,
    #[serde(default)]
    pub fallbacks: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct LoadBalanceConfig {
    pub providers: Vec<String>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct RoutingRule {
    pub name: String,
    pub task: Option<String>,
    pub max_prompt_length: Option<usize>,
    pub keywords: Option<Vec<String>>,
    pub provider: String,
    pub fallbacks: Option<Vec<String>>,
}

pub fn load_config() -> Result<AppConfig, crate::error::ServeError> {
    let config = config::Config::builder()
        .add_source(config::File::with_name("config").required(false))
        .add_source(config::Environment::with_prefix("APP").separator("__"))
        .build()
        .map_err(|e| crate::error::ServeError::Config(e.to_string()))?;

    config
        .try_deserialize()
        .map_err(|e| crate::error::ServeError::Config(e.to_string()))
}
