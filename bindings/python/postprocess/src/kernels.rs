//! Numeric kernels preserve Python ordering and integer blending semantics.
use pyo3::buffer::PyBuffer;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

pub fn mst(count: usize, pairs: &[(usize, usize)]) -> Result<Vec<(usize, usize)>, &'static str> {
    if count > 1_000_000 || pairs.iter().any(|&(a, b)| a >= count || b >= count) {
        return Err("Invalid MST vertex index or count");
    }
    let mut parents: Vec<_> = (0..count).collect();
    let mut result = Vec::with_capacity(count.saturating_sub(1));
    for &(a, b) in pairs {
        let (mut left, mut right) = (a, b);
        while parents[left] != left {
            parents[left] = parents[parents[left]];
            left = parents[left];
        }
        while parents[right] != right {
            parents[right] = parents[parents[right]];
            right = parents[right];
        }
        if left == right {
            continue;
        }
        parents[left] = right;
        result.push((a, b));
        if result.len() == count.saturating_sub(1) {
            break;
        }
    }
    Ok(result)
}

#[pyfunction]
pub fn mst_edges(
    py: Python<'_>,
    count: usize,
    pairs: Vec<(usize, usize)>,
) -> PyResult<Vec<(usize, usize)>> {
    py.detach(|| mst(count, &pairs))
        .map_err(PyValueError::new_err)
}

#[pyfunction]
pub fn blend_bgra(
    py: Python<'_>,
    frame: PyBuffer<u8>,
    overlay: PyBuffer<u8>,
    x: i64,
    y: i64,
) -> PyResult<()> {
    if frame.readonly()
        || !frame.is_c_contiguous()
        || frame.dimensions() != 3
        || frame.shape()[2] != 3
        || !overlay.is_c_contiguous()
        || overlay.dimensions() != 3
        || overlay.shape()[2] != 4
    {
        return Err(PyValueError::new_err(
            "Expected writable contiguous BGR frame and contiguous BGRA overlay",
        ));
    }
    let (h, w) = (frame.shape()[0], frame.shape()[1]);
    let (oh, ow) = (overlay.shape()[0], overlay.shape()[1]);
    if h > i32::MAX as usize
        || w > i32::MAX as usize
        || oh > i32::MAX as usize
        || ow > i32::MAX as usize
        || h.checked_mul(w).and_then(|n| n.checked_mul(3)) != Some(frame.len_bytes())
        || oh.checked_mul(ow).and_then(|n| n.checked_mul(4)) != Some(overlay.len_bytes())
    {
        return Err(PyValueError::new_err("Invalid image dimensions"));
    }
    // Exporters remain pinned and the GIL stays held. Read/write through raw
    // pointers without manufacturing Rust references to Python-owned storage.
    let target = frame.buf_ptr().cast::<u8>();
    let source = overlay.buf_ptr().cast::<u8>();
    let target_end = (target as usize)
        .checked_add(frame.len_bytes())
        .ok_or_else(|| PyValueError::new_err("Buffer address overflow"))?;
    let source_end = (source as usize)
        .checked_add(overlay.len_bytes())
        .ok_or_else(|| PyValueError::new_err("Buffer address overflow"))?;
    if (target as usize) < source_end && (source as usize) < target_end {
        return Err(PyValueError::new_err(
            "Overlay must not alias the destination",
        ));
    }
    let _ = py;
    let x1 = x.max(0).min(w as i64);
    let y1 = y.max(0).min(h as i64);
    let x2 = x.saturating_add(ow as i64).max(0).min(w as i64);
    let y2 = y.saturating_add(oh as i64).max(0).min(h as i64);
    if x1 >= x2 || y1 >= y2 {
        return Ok(());
    }
    for row in y1..y2 {
        for col in x1..x2 {
            let src = (((row - y) as usize) * ow + (col - x) as usize) * 4;
            let dst = ((row as usize) * w + col as usize) * 3;
            let alpha = u32::from(unsafe { source.add(src + 3).read() });
            if alpha == 0 {
                continue;
            }
            for channel in 0..3 {
                let value = (u32::from(unsafe { source.add(src + channel).read() }) * alpha
                    + u32::from(unsafe { target.add(dst + channel).read() }) * (255 - alpha)
                    + 127)
                    / 255;
                unsafe { target.add(dst + channel).write(value as u8) };
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn mst_preserves_order_and_rejects_bad_indices() {
        assert_eq!(
            mst(4, &[(0, 1), (1, 0), (2, 3), (1, 2)]).unwrap(),
            vec![(0, 1), (2, 3), (1, 2)]
        );
        assert!(mst(2, &[(0, 2)]).is_err());
        assert_eq!(mst(0, &[]).unwrap(), vec![]);
    }
}

#[pyfunction]
pub fn blend_mask_lut(
    frame: PyBuffer<u8>,
    mask: PyBuffer<u8>,
    lut: PyBuffer<u8>,
    x: usize,
    y: usize,
) -> PyResult<()> {
    if frame.readonly()
        || !frame.is_c_contiguous()
        || frame.dimensions() != 3
        || frame.shape()[2] != 3
        || !mask.is_c_contiguous()
        || mask.dimensions() != 2
        || !lut.is_c_contiguous()
        || lut.shape() != [256, 3]
    {
        return Err(PyValueError::new_err(
            "Expected BGR frame, mask and 256x3 LUT",
        ));
    }
    let (h, w) = (frame.shape()[0], frame.shape()[1]);
    let (mh, mw) = (mask.shape()[0], mask.shape()[1]);
    if x.checked_add(mw).is_none_or(|end| end > w)
        || y.checked_add(mh).is_none_or(|end| end > h)
        || h.checked_mul(w).and_then(|n| n.checked_mul(3)) != Some(frame.len_bytes())
        || mh.checked_mul(mw) != Some(mask.len_bytes())
        || lut.len_bytes() != 768
    {
        return Err(PyValueError::new_err("Invalid mask dimensions or origin"));
    }
    let target = frame.buf_ptr().cast::<u8>();
    let source = mask.buf_ptr().cast::<u8>();
    let table = lut.buf_ptr().cast::<u8>();
    for input in [&mask, &lut] {
        let a = target as usize;
        let b = input.buf_ptr() as usize;
        let end_a = a
            .checked_add(frame.len_bytes())
            .ok_or_else(|| PyValueError::new_err("Buffer address overflow"))?;
        let end_b = b
            .checked_add(input.len_bytes())
            .ok_or_else(|| PyValueError::new_err("Buffer address overflow"))?;
        if a < end_b && b < end_a {
            return Err(PyValueError::new_err(
                "Inputs must not alias the destination",
            ));
        }
    }
    // The GIL is held; each bounded access uses pinned exported storage.
    for row in 0..mh {
        for col in 0..mw {
            if unsafe { source.add(row * mw + col).read() } == 0 {
                continue;
            }
            let base = ((row + y) * w + col + x) * 3;
            for channel in 0..3 {
                unsafe {
                    let original = target.add(base + channel).read() as usize;
                    target
                        .add(base + channel)
                        .write(table.add(original * 3 + channel).read());
                }
            }
        }
    }
    Ok(())
}
