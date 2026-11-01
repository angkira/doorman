// Public modules for integration tests and examples
pub mod camera;
pub mod dataset;
pub mod decision;
pub mod ipc_policy;
pub mod liveness;
pub mod ml;
pub mod storage;
pub mod sudo_policy;
pub mod timing;
pub mod training;
pub mod unlock_wait;

// Re-export commonly used types
pub use camera::Camera;
pub use ml::{Face, MLPipeline};
pub use storage::Storage;
