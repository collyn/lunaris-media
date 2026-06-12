//! Runtime XRandR virtual display management.
//!
//! This module expects the X server to already expose a disconnected VIRTUAL
//! output (for example via Xorg `VirtualHeads`). It does not write system Xorg
//! configuration.

use std::process::{Command, Output};

use crate::error::MediaError;

pub struct VirtualDisplay {
    output_name: String,
    mode_name: String,
    active: bool,
}

impl VirtualDisplay {
    pub fn create(width: u32, height: u32, fps: u32) -> Result<Self, MediaError> {
        let mode_name = format!("lunaris_{}x{}_{}", width, height, fps);
        let modeline_args = Self::generate_modeline(width, height, fps)?;
        let output_name = Self::find_virtual_output()?;

        let mut newmode_args = vec!["--newmode".to_string(), mode_name.clone()];
        newmode_args.extend(modeline_args);
        Self::run_xrandr_allow_existing(&newmode_args)?;
        Self::run_xrandr_allow_existing(&[
            "--addmode".to_string(),
            output_name.clone(),
            mode_name.clone(),
        ])?;
        Self::run_xrandr(&[
            "--output".to_string(),
            output_name.clone(),
            "--mode".to_string(),
            mode_name.clone(),
        ])?;

        Ok(Self {
            output_name,
            mode_name,
            active: true,
        })
    }

    pub fn output_name(&self) -> &str {
        &self.output_name
    }

    pub fn destroy(&mut self) -> Result<(), MediaError> {
        if !self.active {
            return Ok(());
        }

        let mut first_error = None;
        for args in [
            vec!["--output", self.output_name.as_str(), "--off"],
            vec![
                "--delmode",
                self.output_name.as_str(),
                self.mode_name.as_str(),
            ],
            vec!["--rmmode", self.mode_name.as_str()],
        ] {
            if let Err(e) = Self::run_xrandr(args) {
                if first_error.is_none() {
                    first_error = Some(e);
                }
            }
        }

        self.active = false;
        if let Some(e) = first_error {
            return Err(e);
        }
        Ok(())
    }

    fn generate_modeline(width: u32, height: u32, fps: u32) -> Result<Vec<String>, MediaError> {
        let output = Command::new("cvt")
            .args([width.to_string(), height.to_string(), fps.to_string()])
            .output()
            .map_err(|e| MediaError::CaptureError(format!("cvt failed: {}", e)))?;

        if !output.status.success() {
            return Err(Self::command_error("cvt", &output));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let modeline = stdout
            .lines()
            .find(|line| line.trim_start().starts_with("Modeline"))
            .ok_or_else(|| MediaError::CaptureError("cvt did not return a Modeline".into()))?;

        let args: Vec<String> = modeline
            .split_whitespace()
            .skip(2)
            .map(|part| part.trim_matches('"').to_string())
            .collect();

        if args.is_empty() {
            return Err(MediaError::CaptureError(
                "cvt returned an empty Modeline".into(),
            ));
        }
        Ok(args)
    }

    fn find_virtual_output() -> Result<String, MediaError> {
        let output = Command::new("xrandr")
            .arg("--query")
            .output()
            .map_err(|e| MediaError::CaptureError(format!("xrandr failed: {}", e)))?;

        if !output.status.success() {
            return Err(Self::command_error("xrandr --query", &output));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        stdout
            .lines()
            .filter(|line| !line.starts_with(' '))
            .find_map(|line| {
                let name = line.split_whitespace().next()?;
                if line.contains(" disconnected")
                    && name.to_ascii_uppercase().starts_with("VIRTUAL")
                {
                    Some(name.to_string())
                } else {
                    None
                }
            })
            .ok_or_else(|| {
                MediaError::CaptureError(
                    "No disconnected VIRTUAL output found. Configure Xorg VirtualHeads first."
                        .into(),
                )
            })
    }

    fn run_xrandr<I, S>(args: I) -> Result<(), MediaError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<std::ffi::OsStr>,
    {
        let output = Command::new("xrandr")
            .args(args)
            .output()
            .map_err(|e| MediaError::CaptureError(format!("xrandr failed: {}", e)))?;

        if output.status.success() {
            Ok(())
        } else {
            Err(Self::command_error("xrandr", &output))
        }
    }

    fn run_xrandr_allow_existing(args: &[String]) -> Result<(), MediaError> {
        let output = Command::new("xrandr")
            .args(args)
            .output()
            .map_err(|e| MediaError::CaptureError(format!("xrandr failed: {}", e)))?;

        if output.status.success()
            || String::from_utf8_lossy(&output.stderr).contains("already exists")
        {
            Ok(())
        } else {
            Err(Self::command_error("xrandr", &output))
        }
    }

    fn command_error(command: &str, output: &Output) -> MediaError {
        let stderr = String::from_utf8_lossy(&output.stderr);
        MediaError::CaptureError(format!(
            "{} failed with status {}: {}",
            command,
            output.status,
            stderr.trim()
        ))
    }
}

impl Drop for VirtualDisplay {
    fn drop(&mut self) {
        if let Err(e) = self.destroy() {
            log::warn!("Failed to destroy virtual display: {}", e);
        }
    }
}
