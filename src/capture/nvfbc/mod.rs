// Vendored from the `nvfbc` 0.2.0 crate (BSD-2-Clause, Copyright (c) Hans Gaiser;
// see src/capture/nvfbc/LICENSE). Changed so NvFBC is loaded with dlopen at
// runtime instead of being linked, like Sunshine does.

#![allow(dead_code, unused_imports)]

pub mod sys;
mod common;
pub mod cuda;
mod error;
pub mod system;
mod types;

pub use types::*;
pub use error::Error;
pub use cuda::CudaCapturer;
pub use system::SystemCapturer;
