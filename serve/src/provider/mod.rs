pub mod local;
pub mod mock;
pub mod registry;

use std::pin::Pin;

use async_trait::async_trait;
use futures::Stream;

use crate::api::types::{GenerateRequest, GenerateResponse, StreamChunk};
use crate::error::ServeError;

pub type ChunkStream = Pin<Box<dyn Stream<Item = Result<StreamChunk, ServeError>> + Send>>;

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;
    async fn generate(&self, req: &GenerateRequest) -> Result<GenerateResponse, ServeError>;
    async fn generate_stream(&self, req: &GenerateRequest) -> Result<ChunkStream, ServeError>;
}
