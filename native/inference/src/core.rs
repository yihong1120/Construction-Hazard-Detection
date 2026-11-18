use std::fmt;

#[derive(Debug)]
pub enum Error {
    Invalid(String),
    Unsupported(String),
    Runtime(String),
}
impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Invalid(s) | Self::Unsupported(s) | Self::Runtime(s) => write!(f, "{s}"),
        }
    }
}
impl std::error::Error for Error {}
pub struct Image<'a> {
    pub h: usize,
    pub w: usize,
    pub data: &'a [u8],
}
#[derive(Clone, Debug)]
pub struct Layout {
    pub h: usize,
    pub w: usize,
    pub rh: usize,
    pub rw: usize,
    pub top: usize,
    pub left: usize,
}

pub fn validate(images: &[Image<'_>], max_batch: usize) -> Result<(), Error> {
    if images.len() > max_batch {
        return Err(Error::Invalid("Batch exceeds engine capacity".into()));
    }
    for image in images {
        validate_dimensions(image.h, image.w, image.data.len())?;
    }
    Ok(())
}

pub fn validate_dimensions(h: usize, w: usize, bytes: usize) -> Result<(), Error> {
    let pixels = h
        .checked_mul(w)
        .ok_or_else(|| Error::Invalid("Image size overflow".into()))?;
    if pixels == 0 || pixels > 64 * 1024 * 1024 || pixels.checked_mul(3) != Some(bytes) {
        return Err(Error::Invalid(
            "Expected non-empty uint8 HWC BGR image (at most 64M pixels)".into(),
        ));
    }
    Ok(())
}

pub fn layout(h: usize, w: usize, size: usize, stride: usize, rect: bool) -> Layout {
    let gain = (size as f64 / h as f64).min(size as f64 / w as f64);
    let rh = (h as f64 * gain).round_ties_even().max(1.0) as usize;
    let rw = (w as f64 * gain).round_ties_even().max(1.0) as usize;
    let dh = if rect {
        (size - rh) % stride
    } else {
        size - rh
    };
    let dw = if rect {
        (size - rw) % stride
    } else {
        size - rw
    };
    Layout {
        h: rh + dh,
        w: rw + dw,
        rh,
        rw,
        top: (dh as f64 * 0.5 - 0.1).round_ties_even() as usize,
        left: (dw as f64 * 0.5 - 0.1).round_ties_even() as usize,
    }
}

pub fn decode(
    output: &[f32],
    h: usize,
    w: usize,
    th: usize,
    tw: usize,
    confidence: f32,
) -> Result<Vec<[f32; 6]>, Error> {
    let gain = (th as f64 / h as f64).min(tw as f64 / w as f64);
    let nh = (h as f64 * gain).round_ties_even().max(1.0);
    let nw = (w as f64 * gain).round_ties_even().max(1.0);
    let gx = (nw / w as f64) as f32;
    let gy = (nh / h as f64) as f32;
    let px = ((tw as f64 - nw) * 0.5 - 0.1).round_ties_even() as f32;
    let py = ((th as f64 - nh) * 0.5 - 0.1).round_ties_even() as f32;
    let mut result = Vec::new();
    for detection in output.chunks_exact(6) {
        if detection[4] <= confidence {
            continue;
        }
        if detection.iter().any(|x| !x.is_finite())
            || detection[5] < 0.
            || detection[5].fract() != 0.
        {
            return Err(Error::Runtime(
                "Engine returned invalid detection values".into(),
            ));
        }
        let mut row: [f32; 6] = detection.try_into().unwrap();
        for index in [0, 2] {
            row[index] = ((row[index] - px) / gx).clamp(0., w as f32);
        }
        for index in [1, 3] {
            row[index] = ((row[index] - py) / gy).clamp(0., h as f32);
        }
        result.push(row);
        if result.len() == 300 {
            break; // Match the worker's default max_det, including larger exports.
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn square_policy_and_rectangular_policy_are_distinct() {
        let square = layout(360, 640, 640, 32, false);
        let rect = layout(360, 640, 640, 32, true);
        assert_eq!((square.h, square.w, square.top), (640, 640, 140));
        assert_eq!((rect.h, rect.w, rect.top), (384, 640, 12));
    }
    #[test]
    fn validates_storage_and_decodes_strict_confidence_and_clipping() {
        assert!(
            validate(
                &[Image {
                    h: 2,
                    w: 2,
                    data: &[0; 3]
                }],
                1
            )
            .is_err()
        );
        assert!(validate(&[], 0).is_ok());
        let rows = decode(
            &[-4., 139., 650., 510., 0.9, 5., 0., 0., 1., 1., 0.25, 2.],
            360,
            640,
            640,
            640,
            0.25,
        )
        .unwrap();
        assert_eq!(rows, vec![[0., 0., 640., 360., 0.9, 5.]]);
    }

    #[test]
    fn large_output_exports_keep_the_worker_detection_limit() {
        let output = [0., 0., 10., 10., 0.9, 5.].repeat(400);
        assert_eq!(
            decode(&output, 640, 640, 640, 640, 0.25).unwrap().len(),
            300
        );
    }
}
