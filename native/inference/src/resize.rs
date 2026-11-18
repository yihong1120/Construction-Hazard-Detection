//! Exact special case of OpenCV INTER_LINEAR, with no floating-point weights.
use crate::core::{Image, Layout};

pub(crate) fn odd_scale(h: usize, w: usize, l: &Layout) -> Option<usize> {
    if l.rh == 0 || l.rw == 0 || h % l.rh != 0 || w % l.rw != 0 {
        return None;
    }
    let scale = w / l.rw;
    (scale > 0 && scale % 2 == 1 && h / l.rh == scale).then_some(scale)
}
pub fn letterbox(image: &Image<'_>, l: &Layout, target: &mut [u8]) -> bool {
    let Some(scale) = odd_scale(image.h, image.w, l) else {
        return false;
    };
    if image.h.checked_mul(image.w).and_then(|n| n.checked_mul(3)) != Some(image.data.len())
        || l.h.checked_mul(l.w).and_then(|n| n.checked_mul(3)) != Some(target.len())
        || l.top.checked_add(l.rh).is_none_or(|v| v > l.h)
        || l.left.checked_add(l.rw).is_none_or(|v| v > l.w)
    {
        return false;
    }
    // For odd integer s, OpenCV's half-pixel coordinate is exactly
    // (x + 0.5) * s - 0.5 = x*s + s/2. The other weights are zero.
    target.fill(114);
    for y in 0..l.rh {
        let sy = y * scale + scale / 2;
        let source = &image.data[sy * image.w * 3..(sy + 1) * image.w * 3];
        let offset = ((y + l.top) * l.w + l.left) * 3;
        let row = &mut target[offset..offset + l.rw * 3];
        if scale == 1 {
            row.copy_from_slice(source);
            continue;
        }
        for (x, pixel) in row.chunks_exact_mut(3).enumerate() {
            let sx = (x * scale + scale / 2) * 3;
            pixel.copy_from_slice(&source[sx..sx + 3]);
        }
    }
    true
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::layout;
    #[test]
    fn unsupported_geometry_keeps_the_output_unmodified() {
        let bytes = vec![3; 1280 * 720 * 3];
        let image = Image {
            h: 720,
            w: 1280,
            data: &bytes,
        };
        let mut output = vec![8; 640 * 640 * 3];
        assert!(!letterbox(
            &image,
            &layout(720, 1280, 640, 32, false),
            &mut output
        ));
        assert!(output.iter().all(|v| *v == 8));
    }
    #[test]
    fn identity_preserves_channels_and_padding() {
        let bytes: Vec<_> = (0..12).collect();
        let image = Image {
            h: 2,
            w: 2,
            data: &bytes,
        };
        let l = Layout {
            h: 4,
            w: 4,
            rh: 2,
            rw: 2,
            top: 1,
            left: 1,
        };
        let mut out = vec![0; 48];
        assert!(letterbox(&image, &l, &mut out));
        assert_eq!(&out[15..21], &bytes[..6]);
        assert_eq!(&out[27..33], &bytes[6..]);
        assert_eq!(&out[..12], &[114; 12]);
    }
}
