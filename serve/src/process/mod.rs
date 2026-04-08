pub mod health;

use std::time::Duration;

use tokio::process::{Child, Command};

use crate::config::{EmbeddingConfig, LlamaConfig};
use crate::error::ServeError;

pub struct ProcessManager {
    child: Option<Child>,
}

impl ProcessManager {
    pub async fn start(config: &LlamaConfig) -> Result<Self, ServeError> {
        let health_url = format!("http://{}:{}/health", config.host, config.port);

        tracing::info!(
            binary = %config.binary,
            model = %config.model,
            port = config.port,
            "starting llama-server"
        );

        let mut cmd = Command::new(&config.binary);

        // Ensure the dynamic linker can find shared libs next to the binary
        // (e.g. libmtmd.so on Linux, libmtmd.dylib on macOS).
        if let Some(bin_dir) = std::path::Path::new(&config.binary)
            .canonicalize()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        {
            let lib_path_var = if cfg!(target_os = "macos") {
                "DYLD_LIBRARY_PATH"
            } else {
                "LD_LIBRARY_PATH"
            };
            let existing = std::env::var(lib_path_var).unwrap_or_default();
            let new_val = if existing.is_empty() {
                bin_dir.to_string_lossy().to_string()
            } else {
                format!("{}:{existing}", bin_dir.display())
            };
            cmd.env(lib_path_var, new_val);
        }

        let child = cmd
            .arg("--model")
            .arg(&config.model)
            .arg("--n-gpu-layers")
            .arg(config.gpu_layers.to_string())
            .arg("--port")
            .arg(config.port.to_string())
            .arg("--host")
            .arg(&config.host)
            .arg("--ctx-size")
            .arg(config.ctx_size.to_string())
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                ServeError::ProcessManager(format!("failed to spawn llama-server: {e}"))
            })?;

        tracing::info!(pid = child.id().unwrap_or(0), "llama-server spawned");

        let mut manager = Self { child: Some(child) };

        if let Err(e) = health::wait_for_health(
            &health_url,
            Duration::from_secs(config.health_check_timeout_secs),
            Duration::from_millis(config.health_check_interval_ms),
        )
        .await
        {
            manager.shutdown().await;
            return Err(e);
        }

        tracing::info!("llama-server is ready");
        Ok(manager)
    }

    pub async fn start_embedding(config: &EmbeddingConfig) -> Result<Self, ServeError> {
        let health_url = format!("http://{}:{}/health", config.host, config.port);

        tracing::info!(
            binary = %config.binary,
            model = %config.model,
            port = config.port,
            "starting embedding llama-server"
        );

        let mut cmd = Command::new(&config.binary);

        if let Some(bin_dir) = std::path::Path::new(&config.binary)
            .canonicalize()
            .ok()
            .and_then(|p| p.parent().map(|d| d.to_path_buf()))
        {
            let lib_path_var = if cfg!(target_os = "macos") {
                "DYLD_LIBRARY_PATH"
            } else {
                "LD_LIBRARY_PATH"
            };
            let existing = std::env::var(lib_path_var).unwrap_or_default();
            let new_val = if existing.is_empty() {
                bin_dir.to_string_lossy().to_string()
            } else {
                format!("{}:{existing}", bin_dir.display())
            };
            cmd.env(lib_path_var, new_val);
        }

        let child = cmd
            .arg("--model")
            .arg(&config.model)
            .arg("--n-gpu-layers")
            .arg(config.gpu_layers.to_string())
            .arg("--port")
            .arg(config.port.to_string())
            .arg("--host")
            .arg(&config.host)
            .arg("--ctx-size")
            .arg(config.ctx_size.to_string())
            .arg("--embedding")
            .kill_on_drop(true)
            .spawn()
            .map_err(|e| {
                ServeError::ProcessManager(format!("failed to spawn embedding llama-server: {e}"))
            })?;

        tracing::info!(pid = child.id().unwrap_or(0), "embedding llama-server spawned");

        let mut manager = Self { child: Some(child) };

        if let Err(e) = health::wait_for_health(
            &health_url,
            Duration::from_secs(config.health_check_timeout_secs),
            Duration::from_millis(config.health_check_interval_ms),
        )
        .await
        {
            manager.shutdown().await;
            return Err(e);
        }

        tracing::info!("embedding llama-server is ready");
        Ok(manager)
    }

    pub async fn shutdown(&mut self) {
        if let Some(mut child) = self.child.take() {
            tracing::info!("shutting down llama-server");
            if let Err(e) = child.kill().await {
                tracing::error!("failed to kill llama-server: {e}");
            } else {
                let _ = child.wait().await;
                tracing::info!("llama-server stopped");
            }
        }
    }
}

impl Drop for ProcessManager {
    fn drop(&mut self) {
        // Best-effort synchronous kill. The async shutdown is preferred.
        if let Some(ref mut child) = self.child {
            let _ = child.start_kill();
        }
    }
}
