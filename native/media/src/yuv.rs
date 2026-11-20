//! Writable YUV420 planes are borrowed only between scaling and submission.
use super::{Encoder, HostFrame, error};
use anyhow::Result;
use std::ffi::{c_int, c_void};

#[derive(Default)]
#[repr(C)]
struct PlaneInfo {
    y: usize,
    u: usize,
    v: usize,
    width: c_int,
    height: c_int,
    y_stride: c_int,
    u_stride: c_int,
    v_stride: c_int,
    matrix: c_int,
}

unsafe extern "C" {
    fn cm_encoder_prepare_host(
        encoder: *mut c_void,
        host: *const c_void,
        target: *mut PlaneInfo,
    ) -> c_int;
    fn cm_encoder_submit(encoder: *mut c_void) -> c_int;
}

pub struct YuvPlaneMut<'frame> {
    pixels: &'frame mut [u8],
    width: usize,
    stride: usize,
}

impl YuvPlaneMut<'_> {
    pub fn row_mut(&mut self, row: usize) -> &mut [u8] {
        let start = row * self.stride;
        &mut self.pixels[start..start + self.width]
    }
}

pub struct Yuv420Mut<'frame> {
    pub y: YuvPlaneMut<'frame>,
    pub u: YuvPlaneMut<'frame>,
    pub v: YuvPlaneMut<'frame>,
    pub width: usize,
    pub height: usize,
    pub colour_matrix: i32,
}

impl Encoder {
    /// Edit the encoder's scaled 8-bit YUV420 frame before submitting it.
    /// The closure may access active pixels only, and cannot keep any planes.
    pub fn write_host_with(
        &mut self,
        host: &HostFrame,
        paint: impl for<'frame> FnOnce(Yuv420Mut<'frame>) -> Result<()>,
    ) -> Result<()> {
        let mut info = PlaneInfo::default();
        let encoder = self.pointer.as_ptr();
        self.interrupt.reset();
        if unsafe { cm_encoder_prepare_host(encoder, host.as_ptr(), &mut info) } < 0 {
            return Err(error());
        }
        let (width, height) = (info.width as usize, info.height as usize);
        let plane = |data: usize, row_width: usize, rows: usize, stride: c_int| {
            let stride = stride as usize;
            // SAFETY: prepare succeeds only after making the encoder AVFrame
            // writable. Its three planes are disjoint and are borrowed until
            // the end of this call, before any further encoder operation.
            let pixels = unsafe {
                std::slice::from_raw_parts_mut(data as *mut u8, (rows - 1) * stride + row_width)
            };
            YuvPlaneMut {
                pixels,
                width: row_width,
                stride,
            }
        };
        paint(Yuv420Mut {
            y: plane(info.y, width, height, info.y_stride),
            u: plane(info.u, width / 2, height / 2, info.u_stride),
            v: plane(info.v, width / 2, height / 2, info.v_stride),
            width,
            height,
            colour_matrix: info.matrix,
        })?;
        self.interrupt.reset();
        if unsafe { cm_encoder_submit(encoder) } < 0 {
            return Err(error());
        }
        Ok(())
    }
}
