// Public modules for integration tests and examples
pub mod camera;
pub mod dataset;
pub mod ipc_policy;
pub mod ml;
pub mod storage;
pub mod timing;
pub mod training;

// Re-export commonly used types
pub use camera::Camera;
pub use ml::{MLPipeline, Face};
pub use storage::Storage;
