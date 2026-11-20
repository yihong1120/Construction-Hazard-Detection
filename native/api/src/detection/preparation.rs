use std::{io::Cursor, sync::Arc};

use axum::{body::Bytes, http::StatusCode};
use image::{DynamicImage, ImageDecoder, ImageFormat, ImageReader, Limits};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::task::AbortOnDropHandle;

use super::runtime::Fault;

const MEMORY_UNIT: u64 = 4096;
const MEMORY_OVERHEAD: u64 = 1024 * 1024;
const BYTES_PER_PIXEL: u64 = 8;

pub(super) struct Preparation {
    workers: Arc<Semaphore>,
    pending: Arc<Semaphore>,
    memory: Arc<Semaphore>,
    max_pixels: u64,
}

pub(super) struct PreparedImage {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) bgr: Vec<u8>,
    // The decoder's allowance follows the frame through the GPU queue.
    _memory: OwnedSemaphorePermit,
    _pending: OwnedSemaphorePermit,
}

impl Preparation {
    pub(super) fn new(capacity: usize, max_pixels: u64) -> Self {
        Self {
            workers: Arc::new(Semaphore::new(capacity.min(2))),
            pending: Arc::new(Semaphore::new(capacity + 1)),
            memory: Arc::new(Semaphore::new(memory_units(max_pixels) as usize)),
            max_pixels,
        }
    }

    pub(super) async fn decode(
        &self,
        bytes: Bytes,
        upload: Option<OwnedSemaphorePermit>,
    ) -> Result<PreparedImage, Fault> {
        let pending = self
            .pending
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Fault::unavailable())?;
        let worker = self
            .workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| Fault::unavailable())?;
        let max_pixels = self.max_pixels;
        let memory = self.memory.clone();
        AbortOnDropHandle::new(tokio::task::spawn_blocking(move || {
            let _worker = worker;
            let _upload = upload;
            decode_image(&bytes, max_pixels, memory, pending)
        }))
        .await
        .map_err(|_| Fault::unavailable())?
    }
}

fn memory_bytes(pixels: u64) -> u64 {
    pixels * BYTES_PER_PIXEL + MEMORY_OVERHEAD
}

fn memory_units(pixels: u64) -> u32 {
    memory_bytes(pixels).div_ceil(MEMORY_UNIT) as u32
}

fn decode_image(
    bytes: &[u8],
    max_pixels: u64,
    memory: Arc<Semaphore>,
    pending: OwnedSemaphorePermit,
) -> Result<PreparedImage, Fault> {
    let invalid = || Fault::Code(StatusCode::BAD_REQUEST, "INVALID_IMAGE");
    let mut reader = ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|_| invalid())?;
    if !matches!(reader.format(), Some(ImageFormat::Jpeg | ImageFormat::Png)) {
        return Err(invalid());
    }
    let mut limits = Limits::default();
    limits.max_alloc = Some(memory_bytes(max_pixels));
    reader.limits(limits);
    let mut decoder = reader.into_decoder().map_err(|_| invalid())?;
    let (width, height) = decoder.dimensions();
    let pixels = u64::from(width) * u64::from(height);
    if pixels == 0 || pixels > max_pixels {
        return Err(Fault::Code(
            StatusCode::BAD_REQUEST,
            "IMAGE_DIMENSIONS_EXCEEDED",
        ));
    }
    let memory = memory
        .try_acquire_many_owned(memory_units(pixels))
        .map_err(|_| Fault::Code(StatusCode::SERVICE_UNAVAILABLE, "INFERENCE_MEMORY_BUSY"))?;
    let mut limits = Limits::default();
    limits.max_alloc = Some(memory_bytes(pixels));
    decoder.set_limits(limits).map_err(|_| invalid())?;
    let mut bgr = DynamicImage::from_decoder(decoder)
        .map_err(|_| invalid())?
        .into_rgb8()
        .into_raw();
    for pixel in bgr.chunks_exact_mut(3) {
        pixel.swap(0, 2);
    }
    Ok(PreparedImage {
        width,
        height,
        bgr,
        _memory: memory,
        _pending: pending,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn image(width: u32, height: u32) -> Bytes {
        let mut bytes = Cursor::new(Vec::new());
        DynamicImage::ImageRgb8(RgbImage::from_pixel(width, height, Rgb([10, 20, 30])))
            .write_to(&mut bytes, ImageFormat::Png)
            .unwrap();
        bytes.into_inner().into()
    }

    #[tokio::test]
    async fn rejects_oversize_images_and_decodes_rgb_to_bgr() {
        let limits = Preparation::new(2, 2);
        let Ok(frame) = limits.decode(image(2, 1), None).await else {
            panic!("expected a two-pixel image");
        };
        assert_eq!((frame.width, frame.height), (2, 1));
        assert_eq!(frame.bgr, [30, 20, 10, 30, 20, 10]);
        assert!(matches!(
            limits.decode(image(2, 2), None).await,
            Err(Fault::Code(
                StatusCode::BAD_REQUEST,
                "IMAGE_DIMENSIONS_EXCEEDED"
            ))
        ));
        assert!(matches!(
            limits
                .decode(Bytes::from_static(b"not an image"), None)
                .await,
            Err(Fault::Code(StatusCode::BAD_REQUEST, "INVALID_IMAGE"))
        ));
    }

    #[tokio::test]
    async fn decoded_budget_follows_the_frame_beyond_decode() {
        let preparation = Preparation::new(2, 128 * 128);
        let image = image(128, 128);
        let first = preparation.decode(image.clone(), None).await;
        assert!(first.is_ok());
        assert!(matches!(
            preparation.decode(image.clone(), None).await,
            Err(Fault::Code(
                StatusCode::SERVICE_UNAVAILABLE,
                "INFERENCE_MEMORY_BUSY"
            ))
        ));
        drop(first);
        assert!(preparation.decode(image, None).await.is_ok());
    }
}
