//! TensorRT inference and CUDA frame management.
pub mod core;
pub mod engine;
#[cfg(feature = "tensorrt")]
pub mod gpu;
pub mod resize;
