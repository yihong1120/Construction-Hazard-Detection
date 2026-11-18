//! Reusable, owner-local scaling for annotated previews.
use super::{HostFrame, error};
use anyhow::{Result, ensure};
use std::{ffi::c_void, ptr};

unsafe extern "C" {
    fn cm_preview_scale_free(context: *mut c_void);
    fn cm_preview_scale(
        context: *mut *mut c_void,
        host: *const c_void,
        pixels: *const u8,
        source_width: i32,
        source_height: i32,
        target: *mut u8,
        length: usize,
        width: i32,
        height: i32,
    ) -> i32;
}

pub struct BgrScaler(*mut c_void);

// The scale context has exactly one owner. Every conversion requires &mut self.
unsafe impl Send for BgrScaler {}

impl Default for BgrScaler {
    fn default() -> Self {
        Self(ptr::null_mut())
    }
}

impl Drop for BgrScaler {
    fn drop(&mut self) {
        unsafe { cm_preview_scale_free(self.0) };
    }
}

fn validate(shape: (u32, u32), length: usize) -> Result<()> {
    let (width, height) = shape;
    ensure!(
        (1..=16384).contains(&width)
            && (1..=16384).contains(&height)
            && u64::from(width) * u64::from(height) <= 64 * 1024 * 1024
            && length == width as usize * height as usize * 3 + 64,
        "Invalid padded BGR preview buffer"
    );
    Ok(())
}

impl BgrScaler {
    pub fn from_host(
        &mut self,
        source: &HostFrame,
        target: &mut [u8],
        shape: (u32, u32),
    ) -> Result<()> {
        self.convert(source.as_ptr(), &[], (0, 0), target, shape)
    }

    pub fn from_bgr(
        &mut self,
        source: &[u8],
        source_shape: (u32, u32),
        target: &mut [u8],
        shape: (u32, u32),
    ) -> Result<()> {
        validate(source_shape, source.len())?;
        self.convert(ptr::null(), source, source_shape, target, shape)
    }

    fn convert(
        &mut self,
        host: *const c_void,
        source: &[u8],
        source_shape: (u32, u32),
        target: &mut [u8],
        shape: (u32, u32),
    ) -> Result<()> {
        validate(shape, target.len())?;
        let status = unsafe {
            cm_preview_scale(
                &mut self.0,
                host,
                source.as_ptr(),
                source_shape.0 as i32,
                source_shape.1 as i32,
                target.as_mut_ptr(),
                target.len(),
                shape.0 as i32,
                shape.1 as i32,
            )
        };
        if status < 0 {
            return Err(error());
        }
        Ok(())
    }
}
