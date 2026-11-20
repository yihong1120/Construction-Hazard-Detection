//! The host encoder draws the selected language into its own YUV420 frame.
use super::{Bounds, Glyph};
use anyhow::{Result, ensure};
use construction_media::Yuv420Mut;
use image::{GrayImage, Rgb};
use std::sync::Arc;

pub struct Annotation {
    extent: (u32, u32),
    pub(super) commands: Vec<Draw>,
}

pub(super) enum Draw {
    Fill(Bounds, Rgb<u8>, u8),
    Glyph(Arc<Glyph>, i32, i32, Bounds),
    Zone(Arc<GrayImage>, (u32, u32), Rgb<u8>, Rgb<u8>),
}

impl Annotation {
    pub(super) fn new(width: u32, height: u32) -> Self {
        Self {
            extent: (width, height),
            commands: Vec::new(),
        }
    }

    pub(crate) fn paint(&self, mut frame: Yuv420Mut<'_>) -> Result<()> {
        ensure!(
            self.extent == (frame.width as u32, frame.height as u32),
            "Annotation canvas and encoder dimensions differ"
        );
        let matrix = frame.colour_matrix;
        for command in &self.commands {
            match command {
                Draw::Fill(area, color, alpha) => {
                    let color = limited_yuv(*color, matrix);
                    if *alpha == 255 {
                        fill(&mut frame, *area, color);
                    } else {
                        blend(&mut frame, *area, |_, _| (color, *alpha));
                    }
                }
                Draw::Glyph(glyph, x, y, area) => {
                    let original = (*x, *y);
                    // Shadow lies under the white glyph. Both have neutral chroma.
                    blend(&mut frame, *area, |x, y| {
                        let index = (y as i32 - original.1) as usize * glyph.width
                            + (x as i32 - original.0) as usize;
                        ([16, 128, 128], glyph.stroke[index])
                    });
                    blend(&mut frame, *area, |x, y| {
                        let index = (y as i32 - original.1) as usize * glyph.width
                            + (x as i32 - original.0) as usize;
                        let alpha = if glyph.stroke[index] > 0 {
                            glyph.alpha[index]
                        } else {
                            0
                        };
                        ([235, 128, 128], alpha)
                    });
                }
                Draw::Zone(mask, (left, top), fill, stroke) => {
                    let tint = limited_yuv(*fill, matrix);
                    let stroke = limited_yuv(*stroke, matrix);
                    let extent = Bounds {
                        left: *left as i32,
                        top: *top as i32,
                        right: (*left + mask.width() - 1) as i32,
                        bottom: (*top + mask.height() - 1) as i32,
                    };
                    let pixels = mask.as_raw();
                    blend(&mut frame, extent, |x, y| {
                        let offset =
                            (y - *top as usize) * mask.width() as usize + x - *left as usize;
                        match pixels[offset] {
                            2 => (stroke, 255),
                            1 => (tint, 102),
                            _ => (tint, 0),
                        }
                    });
                }
            }
        }
        Ok(())
    }
}

fn limited_yuv(bgr: Rgb<u8>, matrix: i32) -> [u8; 3] {
    let (red, blue) = match matrix {
        1 => (0.2126, 0.0722),      // BT.709
        4 => (0.30, 0.11),          // FCC
        7 => (0.212, 0.087),        // SMPTE 240M
        9 | 10 => (0.2627, 0.0593), // BT.2020
        _ => (0.299, 0.114),        // BT.601
    };
    let [b, g, r] = bgr.0.map(|channel| f64::from(channel) / 255.0);
    let y = red * r + (1.0 - red - blue) * g + blue * b;
    [
        (16.0 + 219.0 * y).round() as u8,
        (128.0 + 112.0 * (b - y) / (1.0 - blue)).round() as u8,
        (128.0 + 112.0 * (r - y) / (1.0 - red)).round() as u8,
    ]
}

fn fill(frame: &mut Yuv420Mut<'_>, area: Bounds, color: [u8; 3]) {
    if area.left > area.right || area.top > area.bottom {
        return;
    }
    let [left, top, right, bottom] = [
        area.left as usize,
        area.top as usize,
        area.right as usize + 1,
        area.bottom as usize + 1,
    ];
    for row in top..bottom {
        frame.y.row_mut(row)[left..right].fill(color[0]);
    }
    for row in top / 2..bottom.div_ceil(2) {
        let coverage_y = bottom.min(2 * row + 2) - top.max(2 * row);
        let u = frame.u.row_mut(row);
        let v = frame.v.row_mut(row);
        for column in left / 2..right.div_ceil(2) {
            let coverage_x = right.min(2 * column + 2) - left.max(2 * column);
            let coverage = (coverage_x * coverage_y) as u32;
            u[column] =
                ((u32::from(u[column]) * (4 - coverage) + u32::from(color[1]) * coverage + 2) / 4)
                    as u8;
            v[column] =
                ((u32::from(v[column]) * (4 - coverage) + u32::from(color[2]) * coverage + 2) / 4)
                    as u8;
        }
    }
}

fn blend(
    frame: &mut Yuv420Mut<'_>,
    area: Bounds,
    mut sample: impl FnMut(usize, usize) -> ([u8; 3], u8),
) {
    if area.left > area.right || area.top > area.bottom {
        return;
    }
    let [left, top, right, bottom] = [
        area.left as usize,
        area.top as usize,
        area.right as usize + 1,
        area.bottom as usize + 1,
    ];
    for row in top / 2..bottom.div_ceil(2) {
        let u = frame.u.row_mut(row);
        let v = frame.v.row_mut(row);
        for column in left / 2..right.div_ceil(2) {
            let (mut alpha_sum, mut cb, mut cr) = (0, 0, 0);
            for y in top.max(2 * row)..bottom.min(2 * row + 2) {
                let luma = frame.y.row_mut(y);
                for x in left.max(2 * column)..right.min(2 * column + 2) {
                    let (color, alpha) = sample(x, y);
                    let alpha = u32::from(alpha);
                    if alpha == 0 {
                        continue;
                    }
                    luma[x] =
                        ((u32::from(luma[x]) * (255 - alpha) + u32::from(color[0]) * alpha + 127)
                            / 255) as u8;
                    alpha_sum += alpha;
                    cb += u32::from(color[1]) * alpha;
                    cr += u32::from(color[2]) * alpha;
                }
            }
            if alpha_sum > 0 {
                // Each sample represents four luma pixels. Average the paint
                // coverage and retain the base chroma under uncovered pixels.
                u[column] = ((u32::from(u[column]) * (1020 - alpha_sum) + cb + 510) / 1020) as u8;
                v[column] = ((u32::from(v[column]) * (1020 - alpha_sum) + cr + 510) / 1020) as u8;
            }
        }
    }
}
