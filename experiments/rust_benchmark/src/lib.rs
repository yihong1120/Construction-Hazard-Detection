//! Faithful port of the default centroid tracker and the PPE/proximity subset.
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{HashMap, HashSet};

pub type Detection = [f64; 6];
pub type Tracked = [f64; 8];

#[derive(Deserialize)]
pub struct Case {
    pub name: String,
    pub kind: String,
    pub frames: Vec<Vec<Detection>>,
}

#[derive(Deserialize)]
pub struct Corpus {
    pub cases: Vec<Case>,
}

struct Track {
    id: u64,
    center: (f64, f64),
    class: i64,
    last_seen: u64,
}

#[derive(Default)]
pub struct Tracker {
    frame_count: u64,
    next_id: u64,
    tracks: Vec<Track>,
}

#[derive(Serialize)]
pub struct FrameResult {
    pub tracks: Vec<Tracked>,
    pub warnings: Value,
}

impl Tracker {
    pub fn track(&mut self, detections: &[Detection]) -> Vec<Tracked> {
        self.frame_count += 1;
        let mut assigned = Vec::with_capacity(detections.len());
        let mut used = HashSet::new();
        for det in detections {
            let cx = (det[0] + det[2]) * 0.5;
            let cy = (det[1] + det[3]) * 0.5;
            let mut best = None;
            let mut best_distance = f64::INFINITY;
            for (index, track) in self.tracks.iter().enumerate() {
                if used.contains(&track.id) || track.class as f64 != det[5] {
                    continue;
                }
                let dx = cx - track.center.0;
                let dy = cy - track.center.1;
                let distance = dx * dx + dy * dy;
                if distance < best_distance && distance < 6400.0 {
                    best = Some(index);
                    best_distance = distance;
                }
            }
            let (id, moving) = if let Some(index) = best {
                let entry = &mut self.tracks[index];
                used.insert(entry.id);
                let moving = if best_distance > 1600.0 { 1.0 } else { 0.0 };
                entry.center = (cx, cy);
                entry.class = det[5] as i64;
                entry.last_seen = self.frame_count;
                (entry.id, moving)
            } else {
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.push(Track {
                    id,
                    center: (cx, cy),
                    class: det[5] as i64,
                    last_seen: self.frame_count,
                });
                // Production permits a newly-created track to match again
                // within this frame; intentionally do not insert into used.
                (id, 0.0)
            };
            assigned.push([
                det[0], det[1], det[2], det[3], det[4], det[5], id as f64, moving,
            ]);
        }
        if self.frame_count.is_multiple_of(10) {
            let threshold = self.frame_count.saturating_sub(10);
            self.tracks.retain(|track| track.last_seen >= threshold);
        }
        assigned
    }

    pub fn process(&mut self, detections: &[Detection]) -> FrameResult {
        let tracks = self.track(detections);
        let warnings = hazards(&tracks);
        FrameResult { tracks, warnings }
    }
}

fn driver(person: &Tracked, vehicle: &Tracked) -> bool {
    let width = person[2] - person[0];
    let height = person[3] - person[1];
    person[3] < vehicle[3]
        && vehicle[3] - person[3] >= height / 2.0
        && person[0] >= vehicle[0] - width / 2.0
        && person[2] <= vehicle[2] + width / 2.0
        && person[1] > vehicle[1]
        && height <= (vehicle[3] - vehicle[1]) / 2.0
}

fn close(person: &Tracked, vehicle: &Tracked) -> bool {
    let width = person[2] - person[0];
    let height = person[3] - person[1];
    let area = (vehicle[2] - vehicle[0]) * (vehicle[3] - vehicle[1]);
    if area <= 0.0 || width * height / area > if vehicle[5] == 10.0 { 0.1 } else { 0.05 } {
        return false;
    }
    let dx = (person[2] - vehicle[0])
        .abs()
        .min((person[0] - vehicle[2]).abs());
    let dy = (person[3] - vehicle[1])
        .abs()
        .min((person[1] - vehicle[3]).abs());
    dx <= 5.0 * width && dy <= 1.5 * height
}

fn bounds(row: &Tracked, cell: f64) -> (i64, i64, i64, i64) {
    (
        (row[0] / cell).floor() as i64,
        (row[1] / cell).floor() as i64,
        (row[2] / cell).floor() as i64,
        (row[3] / cell).floor() as i64,
    )
}

pub fn hazards(tracks: &[Tracked]) -> Value {
    let mut rows = tracks.to_vec();
    for row in &mut rows {
        let (left, right) = (row[0].min(row[2]), row[0].max(row[2]));
        let (top, bottom) = (row[1].min(row[3]), row[1].max(row[3]));
        row[0] = left;
        row[1] = top;
        row[2] = right;
        row[3] = bottom;
    }
    let mut warnings = serde_json::Map::new();
    for (class, label) in [(2.0, "warning_no_hardhat"), (4.0, "warning_no_safety_vest")] {
        let count = rows.iter().filter(|row| row[5] == class).count();
        if count > 0 {
            warnings.insert(label.to_string(), json!({"count": count}));
        }
    }
    let machinery: Vec<&Tracked> = rows
        .iter()
        .filter(|row| row[5] == 8.0 || row[5] == 10.0)
        .collect();
    if machinery.is_empty() {
        return Value::Object(warnings);
    }
    let mut dimensions: Vec<f64> = machinery
        .iter()
        .map(|row| (row[2] - row[0]).abs().max((row[3] - row[1]).abs()))
        .collect();
    dimensions.sort_by(f64::total_cmp);
    let cell = dimensions[dimensions.len() / 2].clamp(64.0, 256.0);
    let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
    for (index, row) in machinery.iter().enumerate() {
        let (x1, y1, x2, y2) = bounds(row, cell);
        for x in x1..=x2 {
            for y in y1..=y2 {
                grid.entry((x, y)).or_default().push(index);
            }
        }
    }
    let mut boxes: [Vec<[f64; 4]>; 2] = [Vec::new(), Vec::new()];
    let mut ids: [Vec<String>; 2] = [Vec::new(), Vec::new()];
    for person in rows.iter().filter(|row| row[5] == 5.0) {
        let dx = (5.0 * (person[2] - person[0])).max(1.0);
        let dy = (1.5 * (person[3] - person[1])).max(1.0);
        let mut query = *person;
        query[0] -= dx;
        query[1] -= dy;
        query[2] += dx;
        query[3] += dy;
        let (x1, y1, x2, y2) = bounds(&query, cell);
        let mut seen = HashSet::new();
        let mut is_driver = false;
        let mut flags = [false, false];
        'cells: for x in x1..=x2 {
            for y in y1..=y2 {
                if let Some(items) = grid.get(&(x, y)) {
                    for &index in items {
                        if !seen.insert(index) {
                            continue;
                        }
                        let vehicle = machinery[index];
                        if driver(person, vehicle) {
                            is_driver = true;
                            break 'cells;
                        }
                        if close(person, vehicle) {
                            flags[if vehicle[5] == 8.0 { 0 } else { 1 }] = true;
                        }
                    }
                }
            }
        }
        if !is_driver {
            for kind in 0..2 {
                if flags[kind] {
                    boxes[kind].push([person[0], person[1], person[2], person[3]]);
                    ids[kind].push(format!("{}", person[6] as u64));
                }
            }
        }
    }
    for (kind, label) in ["warning_close_to_machinery", "warning_close_to_vehicle"]
        .iter()
        .enumerate()
    {
        if !boxes[kind].is_empty() {
            warnings.insert(label.to_string(), json!({
            "count":boxes[kind].len(), "person_bboxes":boxes[kind], "person_track_ids":ids[kind]
        }));
        }
    }
    Value::Object(warnings)
}

#[unsafe(no_mangle)]
pub extern "C" fn tracker_new() -> *mut Tracker {
    Box::into_raw(Box::new(Tracker::default()))
}

/// # Safety
/// Pointer must come from tracker_new and may be freed only once.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tracker_free(pointer: *mut Tracker) {
    if !pointer.is_null() {
        drop(unsafe { Box::from_raw(pointer) });
    }
}

/// # Safety
/// Input contains count * 6 doubles. Returned C string must be released.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tracker_frame(
    pointer: *mut Tracker,
    data: *const Detection,
    count: usize,
) -> *mut std::ffi::c_char {
    if pointer.is_null() || data.is_null() {
        return std::ptr::null_mut();
    }
    let result =
        unsafe { &mut *pointer }.process(unsafe { std::slice::from_raw_parts(data, count) });
    std::ffi::CString::new(serde_json::to_string(&result).unwrap())
        .unwrap()
        .into_raw()
}

/// # Safety
/// Pointer must be a live string returned by tracker_frame.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn tracker_string_free(pointer: *mut std::ffi::c_char) {
    if !pointer.is_null() {
        drop(unsafe { std::ffi::CString::from_raw(pointer) });
    }
}
