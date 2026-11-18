use std::{
    ops::{Deref, DerefMut},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

#[derive(Clone)]
pub struct Pool(Arc<PoolInner>);
struct PoolInner {
    buffers: Mutex<Vec<Vec<u8>>>,
    bytes: usize,
    capacity: usize,
    allocated: AtomicUsize,
}
pub struct Buffer {
    data: Vec<u8>,
    pool: Pool,
}
impl Pool {
    pub fn new(bytes: usize, capacity: usize) -> Self {
        Self(Arc::new(PoolInner {
            buffers: Mutex::new(Vec::new()),
            bytes,
            capacity,
            allocated: AtomicUsize::new(0),
        }))
    }
    pub fn take(&self) -> Buffer {
        self.try_take().expect("Bounded frame pool exhausted")
    }
    pub fn try_take(&self) -> Option<Buffer> {
        let mut free = self.0.buffers.lock().unwrap_or_else(|e| e.into_inner());
        let data = if let Some(data) = free.pop() {
            data
        } else {
            if self.0.allocated.load(Ordering::Relaxed) >= self.0.capacity {
                return None;
            }
            self.0.allocated.fetch_add(1, Ordering::Relaxed);
            vec![0; self.0.bytes + 64]
        };
        Some(Buffer {
            data,
            pool: self.clone(),
        })
    }
    pub fn allocated(&self) -> usize {
        self.0.allocated.load(Ordering::Relaxed)
    }
    pub fn limit(&self) -> usize {
        self.0.capacity
    }
}
impl Buffer {
    pub(crate) fn padded(&self) -> &[u8] {
        &self.data
    }
    pub(crate) fn padded_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}
impl Deref for Buffer {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        &self.data[..self.pool.0.bytes]
    }
}
impl DerefMut for Buffer {
    fn deref_mut(&mut self) -> &mut [u8] {
        &mut self.data[..self.pool.0.bytes]
    }
}
impl Drop for Buffer {
    fn drop(&mut self) {
        let mut free = self
            .pool
            .0
            .buffers
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        if free.len() < self.pool.0.capacity {
            free.push(std::mem::take(&mut self.data));
        }
    }
}
pub struct Frame {
    pub pixels: Buffer,
    pub device: Option<Arc<construction_media::Nv12Frame>>,
    pub host: Option<Arc<construction_media::HostFrame>>,
    pub width: u32,
    pub height: u32,
    pub sequence: u64,
    pub generation: u64,
    pub timestamp_ms: i64,
    pub captured: Instant,
    pub source_pts: Option<f64>,
    pub source_lag_ms: f64,
}
impl Frame {
    pub fn bgr_padded(&self) -> anyhow::Result<&[u8]> {
        if let Some(host) = &self.host {
            return host.bgr_padded();
        }
        match &self.device {
            Some(device) => device.bgr_padded(),
            None => Ok(self.pixels.padded()),
        }
    }
    pub fn bgr(&self) -> anyhow::Result<&[u8]> {
        let pixels = self.bgr_padded()?;
        Ok(&pixels[..self.width as usize * self.height as usize * 3])
    }
}
pub type SharedFrame = Arc<Frame>;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn pool_is_a_hard_limit_and_preserves_hidden_codec_padding() {
        let pool = Pool::new(12, 2);
        let first = pool.take();
        let mut second = pool.take();
        assert!(pool.try_take().is_none());
        assert_eq!(second.len(), 12);
        assert_eq!(second.padded_mut().len(), 76);
        drop(first);
        assert!(pool.try_take().is_some());
        assert_eq!(pool.allocated(), 2);
    }
    #[test]
    fn pool_retains_exclusive_storage_until_last_owner_drops() {
        let pool = Pool::new(12, 2);
        let mut first = pool.take();
        first[0] = 7;
        let pointer = first.as_ptr();
        let held = Arc::new(first);
        let cloned = held.clone();
        drop(held);
        let second = pool.take();
        assert_ne!(pointer, second.as_ptr());
        drop(cloned);
        let recycled = pool.take();
        assert_eq!(pointer, recycled.as_ptr());
        assert_eq!(recycled[0], 7);
    }
}
