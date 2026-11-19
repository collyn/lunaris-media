//! Unattended screen capture on Linux (Wayland included) without the portal
//! prompt: a small root service owns DRM/KMS and hands the scan-out
//! framebuffers, as read-only DMA-BUFs, to the user's agent over a Unix
//! socket.
//!
//! ```text
//!  root: lunaris-agent --capture-service            user session: agent
//!  ┌──────────────────────────────────┐   /run/lunaris/capture.sock   ┌─────────────────────────┐
//!  │ KmsSource: GETCRTC → GETFB2 →    │  Frame + DMA-BUF fd ───────▶  │ DrmServiceCapture       │
//!  │ PRIME_HANDLE_TO_FD (read-only)   │  Cursor (plane pos + ARGB) ─▶ │  → VAAPI on the scan-out│
//!  │ auth: SO_PEERCRED, logind active │  ◀─── Hello/Start/FrameAck    │    GPU's render node    │
//!  │ uid, /proc/<pid>/exe allow-list  │                               │ (encode + network here) │
//!  └──────────────────────────────────┘                               └─────────────────────────┘
//! ```
//!
//! - [`protocol`]: versioned, bounded, little-endian messages.
//! - [`transport`]: `SOCK_SEQPACKET` + `SCM_RIGHTS`.
//! - [`auth`]: who may connect (pure decision + system probes).
//! - [`kms`]: read-only DRM ioctls, framebuffer export, cursor plane.
//! - [`producer`]: the service.
//! - [`client`]: the [`DrmServiceCapture`] backend.
//!
//! Security model and limits: `docs/linux-capture-service.md` in the Lunaris
//! repository.

pub mod auth;
pub mod client;
pub mod kms;
pub mod producer;
pub mod protocol;
pub mod transport;

pub use client::{service_available, socket_path, DrmServiceCapture};
pub use producer::{serve, ServiceConfig};

/// Socket the service listens on (its directory is root-owned, 0755).
pub const DEFAULT_SOCKET_PATH: &str = "/run/lunaris/capture.sock";

#[cfg(test)]
mod tests;
