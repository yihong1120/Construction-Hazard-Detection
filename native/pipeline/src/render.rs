use crate::{
    config::{Font, Model},
    frame::{Frame, Pool, SharedFrame},
    hazard::{HazardResult, Polygons, Track},
    locale::{Status, canonical},
};
use anyhow::{Result, ensure};
use cosmic_text::{Attrs, Buffer, Color, Family, FontSystem, Metrics, Shaping, SwashCache};
use image::{GrayImage, Luma, Rgb};
use imageproc::{
    drawing::{draw_line_segment_mut, draw_polygon_mut},
    point::Point,
};
use std::{
    collections::{BTreeMap, HashMap, VecDeque},
    sync::{Arc, Mutex, OnceLock},
};
const MIN_CONFIDENCE: f64 = 0.25;
const MAX_LABELS: usize = 40;
const MAX_WARNING_ROWS: usize = 5;
const WARNING_BORDER: Rgb<u8> = Rgb([0, 0, 255]);
const WARNING_COLOR: Rgb<u8> = Rgb([54, 67, 244]);

fn dictionary() -> &'static serde_json::Value {
    static LABELS: OnceLock<serde_json::Value> = OnceLock::new();
    LABELS.get_or_init(|| {
        serde_json::from_str(include_str!("../assets/labels.json"))
            .expect("compiled label dictionary")
    })
}

fn class_code(id: u32) -> &'static str {
    let keys = [
        "helmet",
        "mask",
        "no_helmet",
        "no_mask",
        "no_safety_vest",
        "person",
        "safety_cone",
        "safety_vest",
        "machinery",
        "utility_pole",
        "vehicle",
    ];
    keys.get(id as usize).copied().unwrap_or("unknown")
}

struct Class<'a> {
    code: &'a str,
    display: Option<&'a str>,
    color: Option<Rgb<u8>>,
    id: u32,
}

impl<'a> Class<'a> {
    fn for_track(model: &'a Model, id: u32) -> Self {
        let metadata = model
            .class_metadata
            .iter()
            .find(|c| c["id"].as_u64() == Some(id.into()));
        Self::from_metadata(metadata, id, class_code(id))
    }

    fn for_person(model: &'a Model) -> Self {
        let metadata = model.class_metadata.iter().find(|c| c["code"] == "person");
        let id = metadata
            .and_then(|m| m["id"].as_u64())
            .and_then(|id| u32::try_from(id).ok())
            .unwrap_or(5);
        Self::from_metadata(metadata, id, "person")
    }

    fn from_metadata(metadata: Option<&'a serde_json::Value>, id: u32, code: &'a str) -> Self {
        Self {
            code: metadata.and_then(|c| c["code"].as_str()).unwrap_or(code),
            display: metadata
                .and_then(|c| c["display_name"].as_str())
                .filter(|x| !x.trim().is_empty()),
            color: metadata
                .and_then(|c| c["color"].as_str())
                .and_then(parse_bgr),
            id,
        }
    }

    fn color(&self, warning: bool) -> Rgb<u8> {
        if let Some(color) = self.color {
            return color;
        }
        if warning {
            return WARNING_COLOR;
        }
        static COLORS: OnceLock<serde_json::Value> = OnceLock::new();
        let palette = COLORS.get_or_init(|| {
            serde_json::from_str(include_str!("../assets/colors.json")).expect("compiled palette")
        });
        palette[self.code]
            .as_str()
            .and_then(parse_bgr)
            .or_else(|| parse_bgr(&crate::schema::color(self.code)))
            .unwrap_or(Rgb([255; 3]))
    }

    fn label<'b>(&'b self, model: &'b Model, language: &str) -> &'b str {
        let model_label = |language: &str| {
            let labels = model.labels.get(&self.id)?;
            labels
                .get(language)
                .filter(|label| !label.trim().is_empty())
                .or_else(|| {
                    labels.iter().find_map(|(alias, label)| {
                        (canonical(alias) == Some(language) && !label.trim().is_empty())
                            .then_some(label)
                    })
                })
                .map(String::as_str)
        };
        model_label(language)
            .or_else(|| dictionary()["CLASS_LABELS"][language][self.code].as_str())
            .or_else(|| dictionary()["CLASS_LABELS"]["en"][self.code].as_str())
            .or_else(|| model_label("en"))
            .or(self.display)
            .unwrap_or(self.code)
    }
}

fn parse_bgr(color: &str) -> Option<Rgb<u8>> {
    let mut rgb = [0; 3];
    hex::decode_to_slice(color.strip_prefix('#')?, &mut rgb).ok()?;
    Some(Rgb([rgb[2], rgb[1], rgb[0]]))
}

fn visible_class<'a>(model: &'a Model, track: &Track) -> Option<Class<'a>> {
    let confidence = track[4];
    if !confidence.is_finite() || confidence < MIN_CONFIDENCE {
        return None;
    }
    let class = Class::for_track(model, track[5] as u32);
    (class.code != "safety_cone").then_some(class)
}

pub(crate) fn has_visible_tracks(model: &Model, tracks: &[Track]) -> bool {
    tracks
        .iter()
        .any(|track| visible_class(model, track).is_some())
}

struct Glyph {
    width: usize,
    height: usize,
    alpha: Vec<u8>,
    stroke: Vec<u8>,
}
struct TextState {
    fonts: FontSystem,
    cache: SwashCache,
    patches: HashMap<(String, u32, String), Arc<Glyph>>,
    order: VecDeque<(String, u32, String)>,
}
pub struct Text {
    state: OnceLock<std::result::Result<Mutex<TextState>, String>>,
    config: BTreeMap<String, Font>,
    families: BTreeMap<String, String>,
}
impl Text {
    pub(crate) fn offline(
        &self,
        frame: &SharedFrame,
        pool: &Pool,
        width: u32,
        height: u32,
        language: &str,
    ) -> Result<SharedFrame> {
        let language = canonical(language).unwrap_or(language);
        let mut pixels = pool.take();
        pixels.fill(20);
        let mut picture = Picture::new(width, height, pixels);
        picture.status(self, language, Status::Offline.text(language), 8)?;
        Ok(Arc::new(Frame {
            pixels: picture.pixels,
            device: None,
            host: None,
            width,
            height,
            sequence: frame.sequence,
            generation: frame.generation,
            timestamp_ms: frame.timestamp_ms,
            captured: frame.captured,
            source_pts: frame.source_pts,
            source_lag_ms: frame.source_lag_ms,
        }))
    }
    pub fn new(config: &BTreeMap<String, Font>) -> Result<Arc<Self>> {
        let families = config
            .iter()
            .filter_map(|(language, font)| {
                font.family
                    .as_ref()
                    .map(|family| (language.clone(), family.clone()))
            })
            .collect();
        Ok(Arc::new(Self {
            state: OnceLock::new(),
            config: config.clone(),
            families,
        }))
    }
    pub fn initialized(&self) -> bool {
        self.state.get().is_some()
    }
    fn state(&self) -> Result<&Mutex<TextState>> {
        self.state
            .get_or_init(|| {
                let load = || -> Result<Mutex<TextState>> {
                    let mut fonts = FontSystem::new();
                    for font in self.config.values() {
                        fonts.db_mut().load_font_file(&font.path)?;
                    }
                    ensure!(
                        fonts.db().faces().next().is_some(),
                        "No fonts available for Rust text rendering"
                    );
                    Ok(Mutex::new(TextState {
                        fonts,
                        cache: SwashCache::new(),
                        patches: HashMap::new(),
                        order: VecDeque::new(),
                    }))
                };
                load().map_err(|e| e.to_string())
            })
            .as_ref()
            .map_err(|e| anyhow::anyhow!(e.clone()))
    }
    fn glyph(&self, language: &str, size: u32, text: &str) -> Result<Arc<Glyph>> {
        let key = (language.to_string(), size, text.chars().take(160).collect());
        let mut state = self.state()?.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(glyph) = state.patches.get(&key) {
            return Ok(glyph.clone());
        }
        let TextState { fonts, cache, .. } = &mut *state;
        let mut buffer = Buffer::new(fonts, Metrics::new(size as f32, size as f32 + 6.));
        buffer.set_size(fonts, Some(2000.), Some(size as f32 + 6.));
        let default_family = match language {
            "zh-TW" => Some("Noto Sans CJK TC"),
            "zh-CN" => Some("Noto Sans CJK SC"),
            "ja" => Some("Noto Sans CJK JP"),
            "th" => Some("Noto Sans Thai"),
            _ => None,
        };
        let family = self
            .families
            .get(language)
            .or_else(|| {
                self.families.iter().find_map(|(alias, family)| {
                    (canonical(alias) == Some(language)).then_some(family)
                })
            })
            .map(String::as_str)
            .or_else(|| {
                default_family.filter(|family| {
                    fonts
                        .db()
                        .faces()
                        .any(|face| face.families.iter().any(|(name, _)| name == family))
                })
            });
        let attrs = if let Some(family) = family {
            Attrs::new().family(Family::Name(family))
        } else {
            Attrs::new()
        };
        buffer.set_text(fonts, &key.2, &attrs, Shaping::Advanced);
        buffer.shape_until_scroll(fonts, true);
        let width = (buffer
            .layout_runs()
            .map(|run| run.line_w)
            .fold(0f32, f32::max)
            .ceil() as usize
            + 4)
        .clamp(1, 2004);
        let height = size as usize + 10;
        let mut alpha = vec![0; width * height];
        buffer.draw(
            fonts,
            cache,
            Color::rgb(255, 255, 255),
            |x, y, w, h, color| {
                for yy in (y + 2).max(0)..(y + 2 + h as i32).min(height as i32) {
                    for xx in (x + 2).max(0)..(x + 2 + w as i32).min(width as i32) {
                        alpha[yy as usize * width + xx as usize] = color.a();
                    }
                }
            },
        );
        // Trim unused font ascent/descent padding, keeping room for the outline.
        let first = alpha
            .chunks_exact(width)
            .position(|row| row.iter().any(|&v| v != 0));
        let last = alpha
            .chunks_exact(width)
            .rposition(|row| row.iter().any(|&v| v != 0));
        let (alpha, height) = if let (Some(first), Some(last)) = (first, last) {
            let top = first.saturating_sub(2);
            let bottom = (last + 3).min(height);
            (alpha[top * width..bottom * width].to_vec(), bottom - top)
        } else {
            (alpha, height)
        };
        let mut stroke = vec![0; width * height];
        for (i, &value) in alpha.iter().enumerate().filter(|(_, value)| **value != 0) {
            let x = (i % width) as i32;
            let y = (i / width) as i32;
            for (dx, dy) in [
                (0, 0),
                (0, 1),
                (0, -1),
                (1, 0),
                (-1, 0),
                (1, 1),
                (-1, -1),
                (1, -1),
                (-1, 1),
            ] {
                let (x, y) = (x + dx, y + dy);
                if x >= 0 && y >= 0 && x < width as i32 && y < height as i32 {
                    let output = &mut stroke[y as usize * width + x as usize];
                    *output = (*output).max(value);
                }
            }
        }
        let glyph = Arc::new(Glyph {
            width,
            height,
            alpha,
            stroke,
        });
        if state.patches.len() >= 512 {
            if let Some(old) = state.order.pop_front() {
                state.patches.remove(&old);
            }
        }
        state.order.push_back(key.clone());
        state.patches.insert(key, glyph.clone());
        Ok(glyph)
    }
}
/// A preview shares one small source image among all currently due languages.
pub(crate) struct Preview {
    scaler: construction_media::BgrScaler,
    pool: Pool,
    shape: (u32, u32),
}

impl Preview {
    pub(crate) fn new(shape: (u32, u32)) -> Self {
        Self {
            scaler: Default::default(),
            pool: Pool::new(shape.0 as usize * shape.1 as usize * 3, 1),
            shape,
        }
    }

    pub(crate) fn scale(&mut self, frame: &Frame) -> Result<SharedFrame> {
        let mut pixels = self.pool.take();
        if let Some(host) = &frame.host {
            self.scaler
                .from_host(host, pixels.padded_mut(), self.shape)?;
        } else {
            self.scaler.from_bgr(
                frame.bgr_padded()?,
                (frame.width, frame.height),
                pixels.padded_mut(),
                self.shape,
            )?;
        }
        Ok(Arc::new(Frame {
            pixels,
            device: None,
            host: None,
            width: self.shape.0,
            height: self.shape.1,
            sequence: frame.sequence,
            generation: frame.generation,
            timestamp_ms: frame.timestamp_ms,
            captured: frame.captured,
            source_pts: frame.source_pts,
            source_lag_ms: frame.source_lag_ms,
        }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Bounds {
    left: i32,
    top: i32,
    right: i32,
    bottom: i32,
}
impl Bounds {
    fn from_coordinates(coords: [f64; 4], extent: (u32, u32), scale: (f64, f64)) -> Option<Self> {
        let [x1, y1, x2, y2] = coords;
        let coordinate = |value: f64, factor: f64, size: u32| {
            ((value * factor).round_ties_even() as i32).clamp(0, size as i32 - 1)
        };
        if !coords.into_iter().all(f64::is_finite) {
            return None;
        }
        let left = coordinate(x1.min(x2), scale.0, extent.0);
        let right = coordinate(x1.max(x2), scale.0, extent.0);
        let top = coordinate(y1.min(y2), scale.1, extent.1);
        let bottom = coordinate(y1.max(y2), scale.1, extent.1);
        (left < right && top < bottom).then_some(Self {
            left,
            top,
            right,
            bottom,
        })
    }

    fn width(self) -> i32 {
        self.right - self.left + 1
    }

    fn height(self) -> i32 {
        self.bottom - self.top + 1
    }

    fn overlaps(self, other: Self) -> bool {
        let width = self.right.min(other.right) - self.left.max(other.left);
        let height = self.bottom.min(other.bottom) - self.top.max(other.top);
        if width <= 0 || height <= 0 {
            return false;
        }
        let intersection = f64::from(width) * f64::from(height);
        let area = |bbox: Self| f64::from(bbox.width() - 1) * f64::from(bbox.height() - 1);
        intersection >= 0.85 * (area(self) + area(other) - intersection)
    }
}

struct Picture {
    pixels: crate::frame::Buffer,
    width: u32,
    height: u32,
}
impl Picture {
    fn new(width: u32, height: u32, pixels: crate::frame::Buffer) -> Self {
        Self {
            pixels,
            width,
            height,
        }
    }

    fn font_size(&self) -> u32 {
        (self.width.min(self.height) + 16)
            .div_euclid(32)
            .clamp(14, 32)
    }

    fn row_mut(&mut self, y: usize, start: usize, end: usize) -> &mut [u8] {
        let row = y * self.width as usize;
        &mut self.pixels[(row + start) * 3..(row + end) * 3]
    }

    fn fill(&mut self, area: Bounds, color: Rgb<u8>) {
        for y in area.top..=area.bottom {
            for pixel in self
                .row_mut(y as usize, area.left as usize, area.right as usize + 1)
                .chunks_exact_mut(3)
            {
                pixel.copy_from_slice(&color.0);
            }
        }
    }

    fn outline(&mut self, bounds: Bounds, color: Rgb<u8>) {
        // The two-pixel box stroke is centered on the requested coordinates.
        let Bounds {
            left,
            top,
            right,
            bottom,
        } = bounds;
        for edge in [
            self.clip(left, top - 1, right, top + 1),
            self.clip(left, bottom - 1, right, bottom + 1),
            self.clip(left - 1, top, left + 1, bottom),
            self.clip(right - 1, top, right + 1, bottom),
        ] {
            self.fill(edge, color);
        }
    }

    fn clip(&self, left: i32, top: i32, right: i32, bottom: i32) -> Bounds {
        Bounds {
            left: left.max(0),
            top: top.max(0),
            right: right.min(self.width as i32 - 1),
            bottom: bottom.min(self.height as i32 - 1),
        }
    }

    fn draw_glyph(&mut self, glyph: &Glyph, x: i32, y: i32, area: Bounds) {
        let start_x = x.max(area.left);
        let end_x = (x + glyph.width as i32).min(area.right + 1);
        let start_y = y.max(area.top);
        let end_y = (y + glyph.height as i32).min(area.bottom + 1);
        if start_x >= end_x {
            return;
        }
        for py in start_y..end_y {
            let first = (py - y) as usize * glyph.width + (start_x - x) as usize;
            let last = first + (end_x - start_x) as usize;
            let row = self.row_mut(py as usize, start_x as usize, end_x as usize);
            for ((pixel, &stroke), &alpha) in row
                .chunks_exact_mut(3)
                .zip(&glyph.stroke[first..last])
                .zip(&glyph.alpha[first..last])
            {
                if stroke == 0 {
                    continue;
                }
                let stroke = u16::from(stroke);
                let alpha = u16::from(alpha);
                for channel in pixel {
                    let shadow = (u16::from(*channel) * (255 - stroke) + 127) / 255;
                    *channel = ((shadow * (255 - alpha) + 255 * alpha + 127) / 255) as u8;
                }
            }
        }
    }

    fn badge(&mut self, glyph: &Glyph, bounds: Bounds, color: Rgb<u8>) {
        let width = glyph.width as i32 + 12;
        let height = glyph.height as i32 + 6;
        let left = bounds.left.min((self.width as i32 - 1 - width).max(0));
        let top = if bounds.top >= height {
            bounds.top - height
        } else {
            bounds.top
        };
        let badge = self.clip(left, top, left + width, top + height);
        self.fill(badge, color);
        self.draw_glyph(glyph, left + 6, top + 3, badge);
    }

    fn warning_panel(&mut self, text: &Text, language: &str, lines: &[(&str, u64)]) -> Result<()> {
        if lines.is_empty() {
            return Ok(());
        }
        let font_size = self.font_size();
        let mut glyphs = Vec::with_capacity(1 + lines.len());
        glyphs.push(text.glyph(language, font_size, Status::Warning.text(language))?);
        for &(label, count) in lines {
            let glyph = if count == 1 {
                text.glyph(language, font_size, label)?
            } else {
                text.glyph(language, font_size, &format!("{label} x{count}"))?
            };
            glyphs.push(glyph);
        }
        let width = glyphs.iter().map(|glyph| glyph.width).max().unwrap_or(0) as i32 + 20;
        let height = glyphs.iter().map(|glyph| glyph.height as i32).sum::<i32>()
            + 5 * (glyphs.len() - 1) as i32
            + 16;
        let left = (self.width as i32 - width - 12).max(0);
        let top = (self.height as i32 - height - 12).max(0);
        let panel = self.clip(left, top, left + width, top + height);
        for y in panel.top..=panel.bottom {
            for channel in self.row_mut(y as usize, panel.left as usize, panel.right as usize + 1) {
                *channel = ((u16::from(*channel) * 22 + 36 * 78 + 50) / 100) as u8;
            }
        }
        self.outline(panel, WARNING_BORDER);
        let mut y = panel.top + 8;
        for glyph in glyphs {
            let x = (panel.right - 10 - glyph.width as i32).max(panel.left + 10);
            self.draw_glyph(&glyph, x, y, panel);
            y += glyph.height as i32 + 5;
        }
        Ok(())
    }

    fn status(&mut self, text: &Text, language: &str, label: &str, top: i32) -> Result<()> {
        let glyph = text.glyph(language, self.font_size(), label)?;
        let bounds = self.clip(
            4,
            top,
            16 + glyph.width as i32,
            top + 6 + glyph.height as i32,
        );
        if bounds.left <= bounds.right && bounds.top <= bounds.bottom {
            self.fill(bounds, Rgb([20; 3]));
            self.draw_glyph(&glyph, 10, top + 3, bounds);
        }
        Ok(())
    }
}

#[derive(Default)]
struct ZoneLayer {
    key: Polygons,
    scale: (f64, f64),
    extent: (u32, u32),
    rasterized: Vec<Vec<Point<i32>>>,
    offset: (u32, u32),
    mask: GrayImage,
}
impl ZoneLayer {
    fn update(&mut self, polygons: &Polygons, extent: (u32, u32), scale: (f64, f64)) {
        if self.key == *polygons && self.scale == scale && self.extent == extent {
            return;
        }
        self.key.clone_from(polygons);
        self.scale = scale;
        let same_extent = self.extent == extent;
        self.extent = extent;
        let mut points: Vec<Vec<_>> = polygons
            .iter()
            .filter_map(|polygon| {
                let normalized = polygon.iter().flatten().all(|v| (0.0..=1.0).contains(v));
                let factors = if normalized {
                    (f64::from(extent.0), f64::from(extent.1))
                } else {
                    scale
                };
                let mut points: Vec<_> = polygon
                    .iter()
                    .map(|p| {
                        Point::new(
                            ((p[0] * factors.0).round_ties_even() as i32)
                                .clamp(0, extent.0 as i32 - 1),
                            ((p[1] * factors.1).round_ties_even() as i32)
                                .clamp(0, extent.1 as i32 - 1),
                        )
                    })
                    .collect();
                points.dedup();
                if points.first() == points.last() {
                    points.pop();
                }
                (points.len() >= 3).then_some(points)
            })
            .collect();
        // Small source-coordinate changes often land on the same preview pixels.
        if same_extent && self.rasterized == points {
            return;
        }
        self.rasterized.clone_from(&points);
        if points.is_empty() {
            self.mask = GrayImage::new(0, 0);
            return;
        }
        let all_points = || points.iter().flatten();
        let left = (all_points().map(|p| p.x).min().unwrap_or_default() - 2).max(0);
        let top = (all_points().map(|p| p.y).min().unwrap_or_default() - 2).max(0);
        let right =
            (all_points().map(|p| p.x).max().unwrap_or_default() + 2).min(extent.0 as i32 - 1);
        let bottom =
            (all_points().map(|p| p.y).max().unwrap_or_default() + 2).min(extent.1 as i32 - 1);
        self.offset = (left as u32, top as u32);
        let size = ((right - left + 1) as u32, (bottom - top + 1) as u32);
        if self.mask.dimensions() == size {
            self.mask.fill(0);
        } else {
            let mut pixels = std::mem::take(&mut self.mask).into_raw();
            pixels.clear();
            pixels.resize(size.0 as usize * size.1 as usize, 0);
            self.mask = GrayImage::from_raw(size.0, size.1, pixels).expect("validated mask size");
        }
        for polygon in &mut points {
            for point in polygon.iter_mut() {
                point.x -= left;
                point.y -= top;
            }
            draw_polygon_mut(&mut self.mask, polygon, Luma([1]));
        }
        for polygon in &points {
            for (start, end) in polygon.iter().zip(polygon.iter().cycle().skip(1)) {
                // Match the three-pixel contour without allocating a second mask.
                for (dx, dy) in [
                    (0, 0),
                    (-1, 0),
                    (1, 0),
                    (0, -1),
                    (0, 1),
                    (-2, 0),
                    (2, 0),
                    (0, -2),
                    (0, 2),
                    (-1, -1),
                    (-1, 1),
                    (1, -1),
                    (1, 1),
                ] {
                    draw_line_segment_mut(
                        &mut self.mask,
                        ((start.x + dx) as f32, (start.y + dy) as f32),
                        ((end.x + dx) as f32, (end.y + dy) as f32),
                        Luma([2]),
                    );
                }
            }
        }
    }

    fn draw(&self, picture: &mut Picture, fill: Rgb<u8>, stroke: Rgb<u8>) {
        let width = self.mask.width() as usize;
        let (left, top) = (self.offset.0 as usize, self.offset.1 as usize);
        if width == 0 {
            return;
        }
        for (y, mask) in self.mask.as_raw().chunks_exact(width).enumerate() {
            let row = picture.row_mut(top + y, left, left + width);
            for (pixel, &value) in row.chunks_exact_mut(3).zip(mask) {
                if value == 2 {
                    pixel.copy_from_slice(&stroke.0);
                } else if value == 1 {
                    for (channel, color) in pixel.iter_mut().zip(fill.0) {
                        *channel = ((u16::from(*channel) * 3 + u16::from(color) * 2 + 2) / 5) as u8;
                    }
                }
            }
        }
    }
}

fn warning_count(value: &serde_json::Value) -> u64 {
    value["count"].as_u64().unwrap_or(0)
}

fn warning_targets(
    hazard: &HazardResult,
    extent: (u32, u32),
    scale: (f64, f64),
) -> std::collections::BTreeSet<Bounds> {
    let mut targets = std::collections::BTreeSet::new();
    for key in ["warning_close_to_machinery", "warning_close_to_vehicle"] {
        let Some(warning) = hazard
            .warnings
            .get(key)
            .filter(|warning| warning_count(warning) > 0)
        else {
            continue;
        };
        for bbox in warning["person_bboxes"].as_array().into_iter().flatten() {
            let Some(coords) = bbox.as_array().filter(|points| points.len() == 4) else {
                continue;
            };
            let (Some(left), Some(top), Some(right), Some(bottom)) = (
                coords[0].as_f64(),
                coords[1].as_f64(),
                coords[2].as_f64(),
                coords[3].as_f64(),
            ) else {
                continue;
            };
            let coords = [left, top, right, bottom];
            let factors = if coords.iter().all(|v| (0.0..=1.0).contains(v)) {
                (f64::from(extent.0), f64::from(extent.1))
            } else {
                scale
            };
            targets.extend(Bounds::from_coordinates(coords, extent, factors));
        }
    }
    targets
}

fn warning_class(code: &str, warnings: &crate::hazard::Warnings) -> bool {
    match code {
        "no_helmet" => warnings.contains_key("warning_no_hardhat"),
        "no_safety_vest" => warnings.contains_key("warning_no_safety_vest"),
        "person" => {
            warnings.contains_key("warning_people_in_controlled_area")
                || warnings.contains_key("warning_people_in_utility_pole_controlled_area")
        }
        "machinery" | "vehicle" => warnings.contains_key("detect_machinery_close_to_pole"),
        _ => false,
    }
}

fn inferred_key(code: &str) -> Option<&'static str> {
    match code {
        "no_helmet" => Some("warning_no_hardhat"),
        "no_mask" => Some("warning_no_mask"),
        "no_safety_vest" => Some("warning_no_safety_vest"),
        _ => None,
    }
}

fn warning_lines<'a>(
    warnings: &'a crate::hazard::Warnings,
    inferred: &[(&'a str, u64)],
    language: &str,
) -> Vec<(&'a str, u64)> {
    warnings
        .iter()
        .map(|(key, value)| (key.as_str(), warning_count(value)))
        .filter(|(_, count)| *count > 0)
        .chain(inferred.iter().copied().filter(|(key, _)| {
            !warnings
                .get(*key)
                .is_some_and(|value| warning_count(value) > 0)
        }))
        .take(MAX_WARNING_ROWS)
        .map(|(key, count)| {
            let label = dictionary()["WARNING_LABELS"][language][key]
                .as_str()
                .or_else(|| dictionary()["WARNING_LABELS"]["en"][key].as_str())
                .unwrap_or(key);
            (label, count)
        })
        .collect()
}

pub struct Renderer {
    pool: Pool,
    text: Arc<Text>,
    cones: ZoneLayer,
    poles: ZoneLayer,
}
impl Renderer {
    pub fn new(width: u32, height: u32, text: Arc<Text>) -> Self {
        Self {
            pool: Pool::new(width as usize * height as usize * 3, 4),
            text,
            cones: ZoneLayer::default(),
            poles: ZoneLayer::default(),
        }
    }

    pub fn render(
        &mut self,
        frame: &SharedFrame,
        tracks: &[Track],
        hazard: &HazardResult,
        model: &Model,
        language: &str,
    ) -> Result<SharedFrame> {
        self.render_status(frame, tracks, hazard, model, language, None)
    }

    pub fn render_status(
        &mut self,
        frame: &SharedFrame,
        tracks: &[Track],
        hazard: &HazardResult,
        model: &Model,
        language: &str,
        status: Option<&str>,
    ) -> Result<SharedFrame> {
        self.render_scaled(
            frame,
            (frame.width, frame.height),
            tracks,
            hazard,
            model,
            language,
            status,
        )
        .and_then(|image| image.ok_or_else(|| anyhow::anyhow!("Overlay frame pool exhausted")))
    }

    /// Draw source-space detections on a frame already scaled for the rendition.
    pub(crate) fn render_scaled(
        &mut self,
        frame: &SharedFrame,
        source: (u32, u32),
        tracks: &[Track],
        hazard: &HazardResult,
        model: &Model,
        language: &str,
        status: Option<&str>,
    ) -> Result<Option<SharedFrame>> {
        let language = canonical(language).unwrap_or(language);
        let extent = (frame.width, frame.height);
        let scale = (
            f64::from(frame.width) / f64::from(source.0),
            f64::from(frame.height) / f64::from(source.1),
        );
        let Some(mut pixels) = self.pool.try_take() else {
            return Ok(None);
        };
        pixels.copy_from_slice(frame.bgr()?);
        let mut picture = Picture::new(frame.width, frame.height, pixels);
        self.cones.update(&hazard.cone_polys, extent, scale);
        self.cones
            .draw(&mut picture, Rgb([129, 64, 255]), Rgb([99, 30, 233]));
        self.poles.update(&hazard.pole_polys, extent, scale);
        self.poles
            .draw(&mut picture, Rgb([255, 138, 68]), Rgb([255, 138, 68]));

        let targets = warning_targets(hazard, extent, scale);
        let mut inferred = Vec::<(&str, u64)>::new();
        let mut labels = 0;
        let font_size = picture.font_size();
        for track in tracks {
            let Some(class) = visible_class(model, track) else {
                continue;
            };
            let &[x1, y1, x2, y2, ..] = track;
            let Some(bounds) = Bounds::from_coordinates([x1, y1, x2, y2], extent, scale) else {
                continue;
            };
            let warning = warning_class(class.code, &hazard.warnings)
                || class.code == "person" && targets.iter().any(|target| bounds.overlaps(*target));
            let color = class.color(warning);
            picture.outline(bounds, color);
            if labels < MAX_LABELS {
                let glyph = self
                    .text
                    .glyph(language, font_size, class.label(model, language))?;
                picture.badge(&glyph, bounds, color);
                labels += 1;
            }
            if let Some(key) = inferred_key(class.code) {
                if let Some((_, count)) = inferred.iter_mut().find(|(entry, _)| *entry == key) {
                    *count += 1;
                } else {
                    inferred.push((key, 1));
                }
            }
        }
        let person = Class::for_person(model);
        let color = person.color(true);
        for bounds in targets {
            picture.outline(bounds, color);
            if labels < MAX_LABELS {
                let glyph = self
                    .text
                    .glyph(language, font_size, person.label(model, language))?;
                picture.badge(&glyph, bounds, color);
                labels += 1;
            }
        }
        let lines = warning_lines(&hazard.warnings, &inferred, language);
        picture.warning_panel(&self.text, language, &lines)?;
        if let Some(status) = status {
            picture.status(
                &self.text,
                language,
                status,
                (frame.height as i32 - 30).max(0),
            )?;
        }
        Ok(Some(Arc::new(Frame {
            pixels: picture.pixels,
            device: None,
            host: None,
            width: frame.width,
            height: frame.height,
            sequence: frame.sequence,
            generation: frame.generation,
            timestamp_ms: frame.timestamp_ms,
            captured: frame.captured,
            source_pts: frame.source_pts,
            source_lag_ms: frame.source_lag_ms,
        })))
    }
}
pub fn jpeg(frame: &SharedFrame) -> Result<Vec<u8>> {
    let pixels = frame.bgr()?;
    let mut rgb = Vec::with_capacity(pixels.len());
    for pixel in pixels.chunks_exact(3) {
        rgb.extend_from_slice(&[pixel[2], pixel[1], pixel[0]]);
    }
    let mut bytes = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut bytes, 88).encode(
        &rgb,
        frame.width,
        frame.height,
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::time::Instant;

    fn model() -> Model {
        serde_json::from_value(json!({
            "key": "fixture",
            "artifact": "fixture",
            "sha256": "0".repeat(64),
            "version": "fixture",
            "labels": {"5": {"en": "Worker"}},
            "class_metadata": [{
                "id": 5,
                "code": "person",
                "display_name": "Worker",
                "color": "#FF9800"
            }]
        }))
        .unwrap()
    }

    fn frame() -> SharedFrame {
        let mut pixels = Pool::new(640 * 360 * 3, 1).take();
        pixels.fill(128);
        Arc::new(Frame {
            pixels,
            device: None,
            host: None,
            width: 640,
            height: 360,
            sequence: 1,
            generation: 1,
            timestamp_ms: 0,
            captured: Instant::now(),
            source_pts: None,
            source_lag_ms: 0.,
        })
    }

    fn pixel(frame: &Frame, x: usize, y: usize) -> &[u8] {
        let offset = (y * frame.width as usize + x) * 3;
        &frame.pixels[offset..offset + 3]
    }

    #[test]
    fn translates_catalog_classes_and_warning_rows_in_each_locale() {
        let mut model = model();
        model.labels.clear();
        model.class_metadata[0]["display_name"] = json!("人員");
        model.class_metadata.push(json!({
            "id": 42, "code": "trolley", "display_name": "Site trolley"
        }));
        let class = Class::for_track(&model, 5);
        let warnings = BTreeMap::from([(
            "warning_people_in_controlled_area".into(),
            json!({"count": 2}),
        )]);
        for (language, label, warning) in [
            ("en", "person", "In restricted area"),
            ("zh-TW", "人員", "進入管制區"),
            ("zh-CN", "人员", "进入管制区"),
            ("ja", "作業員", "立入禁止区域内"),
            ("vi", "người", "Trong khu vực hạn chế"),
            ("id", "orang", "Di area terbatas"),
            ("fr", "personne", "Zone restreinte"),
            ("th", "คนงาน", "อยู่ในพื้นที่ควบคุม"),
        ] {
            assert_eq!(class.label(&model, language), label, "{language}");
            assert_eq!(warning_lines(&warnings, &[], language), [(warning, 2)]);
        }
        model.labels.insert(
            5,
            BTreeMap::from([
                ("fr-CA".into(), "Ouvrier".into()),
                ("de".into(), "Arbeitskraft".into()),
            ]),
        );
        assert_eq!(Class::for_person(&model).label(&model, "fr"), "Ouvrier");
        assert_eq!(
            Class::for_person(&model).label(&model, "de"),
            "Arbeitskraft"
        );
        assert_eq!(
            Class::for_track(&model, 42).label(&model, "ja"),
            "Site trolley"
        );
    }

    #[test]
    fn draws_localized_frames_and_normalizes_the_render_language() {
        let source = frame();
        let mut model = model();
        model.class_metadata[0]["display_name"] = json!("人員");
        model
            .labels
            .get_mut(&5)
            .unwrap()
            .insert("de".into(), "Person".into());
        let track = [[40., 70., 180., 190., 0.95, 5., 19., 0.]];
        let text = Text::new(&BTreeMap::new()).unwrap();
        let mut renderer = Renderer::new(640, 360, text);
        let hazard = HazardResult {
            warnings: BTreeMap::from([(
                "warning_people_in_controlled_area".into(),
                json!({"count": 1}),
            )]),
            ..Default::default()
        };
        let indonesian = renderer
            .render(&source, &track, &hazard, &model, "id")
            .unwrap();
        let alias = renderer
            .render(&source, &track, &hazard, &model, "id_ID")
            .unwrap();
        let english = renderer
            .render(&source, &track, &hazard, &model, "en")
            .unwrap();
        let german = renderer
            .render(&source, &track, &hazard, &model, "de")
            .unwrap();
        assert!(indonesian.pixels[..] == alias.pixels[..]);
        assert!(indonesian.pixels[..] != english.pixels[..]);
        assert!(german.pixels[..] != english.pixels[..]);
    }

    #[test]
    fn overlay_uses_python_zone_box_badge_and_warning_colors() {
        let hazard = HazardResult {
            warnings: BTreeMap::from([("warning_close_to_machinery".into(), json!({"count": 2}))]),
            cone_polys: vec![vec![[220., 60.], [300., 60.], [300., 180.], [220., 180.]]],
            pole_polys: vec![vec![[0.6, 0.2], [0.7, 0.2], [0.7, 0.5], [0.6, 0.5]]],
        };
        let tracks = [
            [40., 40., 180., 190., 0.25, 5., 19., 0.],
            [480., 40., 510., 190., 0.95, 6., 20., 0.],
            [530., 40., 560., 190., 0.24, 7., 21., 0.],
        ];
        let source = frame();
        let model = model();
        let mut renderer = Renderer::new(640, 360, Text::new(&BTreeMap::new()).unwrap());
        for language in ["en", "zh-TW"] {
            let output = renderer
                .render(&source, &tracks, &hazard, &model, language)
                .unwrap();
            assert_eq!(pixel(&output, 40, 100), [0, 152, 255]);
            assert_eq!(pixel(&output, 44, 30), [0, 152, 255]);
            assert_eq!(pixel(&output, 480, 100), [128; 3]);
            assert_eq!(pixel(&output, 530, 100), [128; 3]);
            assert_eq!(pixel(&output, 220, 100), [99, 30, 233]);
            assert_eq!(pixel(&output, 230, 100), [128, 102, 179]);
            assert_eq!(pixel(&output, 384, 100), [255, 138, 68]);
            assert_eq!(pixel(&output, 398, 100), [179, 132, 104]);
            assert_eq!(pixel(&output, 627, 346), [0, 0, 255]);
            assert_eq!(pixel(&output, 626, 346), [56; 3]);
        }
        assert_eq!(Class::for_track(&model, 5).label(&model, "en"), "Worker");
    }

    #[test]
    fn renders_proximity_warning_target_without_track_and_prioritizes_explicit_counts() {
        let hazard = HazardResult {
            warnings: BTreeMap::from([
                (
                    "warning_close_to_machinery".into(),
                    json!({
                        "count": 1, "person_bboxes": [[0.25, 0.25, 0.5, 0.5]]
                    }),
                ),
                ("warning_no_hardhat".into(), json!({"count": 3})),
            ]),
            ..Default::default()
        };
        let source = frame();
        let mut model = model();
        model.class_metadata.clear();
        let mut renderer = Renderer::new(640, 360, Text::new(&BTreeMap::new()).unwrap());
        let output = renderer
            .render(&source, &[], &hazard, &model, "en")
            .unwrap();
        assert_eq!(pixel(&output, 160, 140), WARNING_COLOR.0);
        assert_eq!(pixel(&output, 159, 140), WARNING_COLOR.0);
        assert_eq!(pixel(&output, 158, 140), [128; 3]);
        let lines = warning_lines(
            &hazard.warnings,
            &[("warning_no_hardhat", 8), ("warning_no_mask", 1)],
            "en",
        );
        assert_eq!(
            lines.iter().map(|(_, count)| *count).collect::<Vec<_>>(),
            [1, 3, 1]
        );
    }
}
