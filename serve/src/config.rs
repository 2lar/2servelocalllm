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
    pub rules: Vec<RoutingRule>,
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
