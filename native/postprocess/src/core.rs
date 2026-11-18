//! Python-independent algorithms; field ordering and thresholds match the runtime.
use std::collections::{HashMap, HashSet};

pub type Detection = [f64; 6];

pub struct Tracked {
    pub detection: Detection,
    pub id: u64,
    pub moving: u8,
}

struct Track {
    id: u64,
    center: (f64, f64),
    class: f64,
    last_seen: u64,
}

#[derive(Default)]
pub struct Tracker {
    next_id: u64,
    tracks: Vec<Track>,
}

impl Tracker {
    pub fn track(
        &mut self,
        detections: &[Detection],
        frame: u64,
        threshold: f64,
        keep: u64,
    ) -> Vec<Tracked> {
        let mut assigned = Vec::with_capacity(detections.len());
        let mut used = HashSet::new();
        for det in detections {
            let center = ((det[0] + det[2]) * 0.5, (det[1] + det[3]) * 0.5);
            let mut best = None;
            let mut distance = f64::INFINITY;
            for (index, track) in self.tracks.iter().enumerate() {
                if used.contains(&track.id) || track.class != det[5] {
                    continue;
                }
                let dx = center.0 - track.center.0;
                let dy = center.1 - track.center.1;
                let candidate = dx * dx + dy * dy;
                if candidate < distance && candidate < threshold * 4.0 {
                    best = Some(index);
                    distance = candidate;
                }
            }
            let (id, moving) = if let Some(index) = best {
                let entry = &mut self.tracks[index];
                used.insert(entry.id);
                entry.center = center;
                entry.class = det[5].trunc();
                entry.last_seen = frame;
                (entry.id, u8::from(distance > threshold))
            } else {
                let id = self.next_id;
                self.next_id += 1;
                self.tracks.push(Track {
                    id,
                    center,
                    class: det[5].trunc(),
                    last_seen: frame,
                });
                // Match the existing same-frame treatment of new tracks.
                (id, 0)
            };
            assigned.push(Tracked {
                detection: *det,
                id,
                moving,
            });
        }
        if frame % 10 == 0 && frame >= keep {
            self.tracks.retain(|track| track.last_seen >= frame - keep);
        }
        assigned
    }
}

pub struct SafetyRow {
    pub bbox: [f64; 4],
    pub class: f64,
    pub track_id: Option<String>,
}

#[derive(Default)]
pub struct Warnings {
    pub no_hardhat: usize,
    pub no_vest: usize,
    pub machinery: Vec<usize>,
    pub vehicle: Vec<usize>,
}

fn driver(p: &SafetyRow, v: &SafetyRow) -> bool {
    let (p, v) = (&p.bbox, &v.bbox);
    let width = p[2] - p[0];
    let height = p[3] - p[1];
    p[3] < v[3]
        && v[3] - p[3] >= height / 2.0
        && p[0] >= v[0] - width / 2.0
        && p[2] <= v[2] + width / 2.0
        && p[1] > v[1]
        && height <= (v[3] - v[1]) / 2.0
}

fn close(p: &SafetyRow, v: &SafetyRow) -> bool {
    let ratio = if v.class == 10.0 { 0.1 } else { 0.05 };
    let (p, v) = (&p.bbox, &v.bbox);
    let width = p[2] - p[0];
    let height = p[3] - p[1];
    let area = (v[2] - v[0]) * (v[3] - v[1]);
    if area <= 0.0 || width * height / area > ratio {
        return false;
    }
    let dx = (p[2] - v[0]).abs().min((p[0] - v[2]).abs());
    let dy = (p[3] - v[1]).abs().min((p[1] - v[3]).abs());
    dx <= 5.0 * width && dy <= 1.5 * height
}

fn cell_index(value: f64, cell: f64) -> f64 {
    // CPython float // uses fmod; floor(value / cell) can differ at decimal
    // boundaries, for example 641.0 // 64.1.
    let remainder = value % cell;
    let mut quotient = (value - remainder) / cell;
    if remainder < 0.0 {
        quotient -= 1.0;
    }
    let floor = quotient.floor();
    if quotient - floor > 0.5 {
        floor + 1.0
    } else {
        floor
    }
}

fn bounds(bbox: [f64; 4], cell: f64) -> [f64; 4] {
    bbox.map(|value| cell_index(value, cell))
}

fn small_bounds(bbox: [f64; 4]) -> Option<[i64; 4]> {
    if bbox
        .iter()
        .any(|value| !value.is_finite() || value.abs() > (1u64 << 50) as f64)
    {
        return None;
    }
    let b = bbox.map(|value| value as i64);
    let cells = (b[2] - b[0] + 1).checked_mul(b[3] - b[1] + 1)?;
    if !(0..=250_000).contains(&cells) {
        return None;
    }
    Some(b)
}

struct Grid {
    cell: f64,
    cells: Option<HashMap<(i64, i64), Vec<usize>>>,
    boxes: Vec<[f64; 4]>,
}

impl Grid {
    fn new(machinery: &[&SafetyRow]) -> Self {
        let mut sizes: Vec<f64> = machinery
            .iter()
            .map(|row| {
                let b = row.bbox;
                (b[2] - b[0]).abs().max((b[3] - b[1]).abs())
            })
            .collect();
        sizes.sort_by(f64::total_cmp);
        let cell = sizes[sizes.len() / 2].clamp(64.0, 256.0);
        let boxes: Vec<_> = machinery.iter().map(|row| bounds(row.bbox, cell)).collect();
        let mut grid: HashMap<(i64, i64), Vec<usize>> = HashMap::new();
        let mut total = 0;
        for (index, bbox) in boxes.iter().enumerate() {
            let Some(b) = small_bounds(*bbox) else {
                return Self {
                    cell,
                    cells: None,
                    boxes,
                };
            };
            total += (b[2] - b[0] + 1) * (b[3] - b[1] + 1);
            if total > 250_000 {
                return Self {
                    cell,
                    cells: None,
                    boxes,
                };
            }
            for x in b[0]..=b[2] {
                for y in b[1]..=b[3] {
                    grid.entry((x, y)).or_default().push(index);
                }
            }
        }
        Self {
            cell,
            cells: Some(grid),
            boxes,
        }
    }

    fn candidates(&self, person: &SafetyRow) -> Vec<usize> {
        let b = person.bbox;
        let dx = (5.0 * (b[2] - b[0])).max(1.0);
        let dy = (1.5 * (b[3] - b[1])).max(1.0);
        let query = bounds([b[0] - dx, b[1] - dy, b[2] + dx, b[3] + dy], self.cell);
        if let (Some(grid), Some(b)) = (&self.cells, small_bounds(query)) {
            let mut seen = HashSet::new();
            let mut result = Vec::new();
            for x in b[0]..=b[2] {
                for y in b[1]..=b[3] {
                    if let Some(items) = grid.get(&(x, y)) {
                        for &index in items {
                            if seen.insert(index) {
                                result.push(index);
                            }
                        }
                    }
                }
            }
            return result;
        }
        // Exact cell-rectangle intersection without unbounded cell enumeration.
        self.boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                b[0] <= query[2] && b[2] >= query[0] && b[1] <= query[3] && b[3] >= query[1]
            })
            .map(|(index, _)| index)
            .collect()
    }
}

pub fn safety(rows: &[SafetyRow], ppe: bool, proximity: bool) -> Warnings {
    let mut warnings = Warnings::default();
    if ppe {
        warnings.no_hardhat = rows.iter().filter(|row| row.class == 2.0).count();
        warnings.no_vest = rows.iter().filter(|row| row.class == 4.0).count();
    }
    if !proximity {
        return warnings;
    }
    let machinery: Vec<_> = rows
        .iter()
        .filter(|row| row.class == 8.0 || row.class == 10.0)
        .collect();
    if machinery.is_empty() {
        return warnings;
    }
    let grid = Grid::new(&machinery);
    for (index, person) in rows.iter().enumerate().filter(|(_, row)| row.class == 5.0) {
        let mut flags = [false, false];
        let mut is_driver = false;
        for candidate in grid.candidates(person) {
            let vehicle = machinery[candidate];
            if driver(person, vehicle) {
                is_driver = true;
                break;
            }
            if close(person, vehicle) {
                flags[usize::from(vehicle.class == 10.0)] = true;
            }
        }
        if !is_driver {
            if flags[0] {
                warnings.machinery.push(index);
            }
            if flags[1] {
                warnings.vehicle.push(index);
            }
        }
    }
    warnings
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decimal_grid_boundaries_match_python_floor_division() {
        assert_eq!(cell_index(641.0, 64.1), 10.0);
        assert_eq!(cell_index(-641.0, 64.1), -11.0);
        assert_eq!(cell_index(-128.0, 64.0), -2.0);
    }

    #[test]
    fn preserves_same_frame_matching_and_expiry() {
        let mut tracker = Tracker::default();
        let row = [0.0, 0.0, 10.0, 20.0, 0.9, 5.0];
        let first = tracker.track(&[row, row], 1, 1600.0, 10);
        assert_eq!((first[0].id, first[1].id), (0, 0));
        tracker.track(&[], 20, 1600.0, 10);
        assert_eq!(tracker.track(&[row], 21, 1600.0, 10)[0].id, 1);
    }

    #[test]
    fn large_spatial_extent_uses_bounded_exact_candidates() {
        let vehicle = SafetyRow {
            bbox: [0.0, 0.0, 1e12, 1e12],
            class: 10.0,
            track_id: None,
        };
        let grid = Grid::new(&[&vehicle]);
        assert!(grid.cells.is_none());
        let person = SafetyRow {
            bbox: [1.0, 1.0, 10.0, 20.0],
            class: 5.0,
            track_id: None,
        };
        assert_eq!(grid.candidates(&person), vec![0]);
    }
}
