//! Standalone Rust camera pipeline. No Python runtime is used.
pub mod analysis;
pub mod capture;
pub mod catalog;
pub mod config;
pub mod delivery;
pub mod executor;
pub mod frame;
pub mod hazard;
pub mod inference;
pub mod lease;
mod locale;
pub mod metadata;
pub mod outbox;
pub mod overlay;
pub use construction_postprocess_core as postprocess;
pub mod publish;
pub mod render;
pub mod rtsp;
pub mod runtime;
pub mod schema;
pub mod tracking;

pub mod demand;
pub mod relay;
