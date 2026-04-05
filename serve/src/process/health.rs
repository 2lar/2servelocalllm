use std::time::Duration;

use crate::error::ServeError;

pub async fn wait_for_health(
    url: &str,
    timeout: Duration,
    interval: Duration,
) -> Result<(), ServeError> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .map_err(|e| ServeError::ProcessManager(format!("failed to build health client: {e}")))?;

    let deadline = tokio::time::Instant::now() + timeout;

    loop {
        match client.get(url).send().await {
            Ok(resp) if resp.status().is_success() => {
                tracing::info!("health check passed: {url}");
                return Ok(());
            }
            Ok(resp) => {
                tracing::debug!("health check returned {}: {url}", resp.status());
            }
            Err(e) => {
                tracing::debug!("health check failed: {e}");
            }
        }

        if tokio::time::Instant::now() >= deadline {
            return Err(ServeError::ProcessManager(format!(
                "health check timed out after {}s for {url}",
                timeout.as_secs()
            )));
        }

        tokio::time::sleep(interval).await;
    }
}
