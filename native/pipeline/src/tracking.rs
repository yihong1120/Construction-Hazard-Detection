use crate::{config::TrackerKind, hazard::Track, inference::Detection, postprocess};
use std::collections::{BTreeMap, HashSet};
struct State {
    id: u64,
    bbox: [f64; 4],
    class: i64,
    last: u64,
}
pub struct Tracker {
    mode: TrackerKind,
    frame: u64,
    centroid: postprocess::Tracker,
    tracks: Vec<State>,
    next: u64,
}
impl Tracker {
    pub fn new(mode: TrackerKind) -> Self {
        Self {
            mode,
            frame: 0,
            centroid: postprocess::Tracker::default(),
            tracks: vec![],
            next: 0,
        }
    }
    pub fn update(&mut self, detections: &[Detection]) -> Vec<Track> {
        self.frame += 1;
        if matches!(self.mode, TrackerKind::Centroid) {
            return self
                .centroid
                .track(detections, self.frame, 1600., 10)
                .into_iter()
                .map(|row| {
                    let d = row.detection;
                    [
                        d[0],
                        d[1],
                        d[2],
                        d[3],
                        d[4],
                        d[5],
                        row.id as f64,
                        row.moving as f64,
                    ]
                })
                .collect();
        }
        if self.tracks.is_empty() {
            return detections
                .iter()
                .map(|d| {
                    let id = self.next;
                    self.next += 1;
                    self.tracks.push(State {
                        id,
                        bbox: [d[0], d[1], d[2], d[3]],
                        class: d[5] as i64,
                        last: self.frame,
                    });
                    [d[0], d[1], d[2], d[3], d[4], d[5], id as f64, 0.]
                })
                .collect();
        }
        let mut groups = Vec::<(i64, Vec<usize>)>::new();
        for (i, d) in detections.iter().enumerate() {
            let class = d[5] as i64;
            if let Some((_, indices)) = groups.iter_mut().find(|(key, _)| *key == class) {
                indices.push(i)
            } else {
                groups.push((class, vec![i]))
            }
        }
        let mut assignments = vec![];
        let mut unmatched = vec![];
        for (class, indices) in groups {
            let tracks: Vec<_> = self
                .tracks
                .iter()
                .enumerate()
                .filter(|(_, s)| s.class == class)
                .map(|(i, _)| i)
                .collect();
            let costs: Vec<Vec<f64>> = indices
                .iter()
                .map(|&i| {
                    tracks
                        .iter()
                        .map(|&j| cost(&detections[i], &self.tracks[j].bbox))
                        .collect()
                })
                .collect();
            let matches = assign(&costs, 0.7);
            let used: HashSet<_> = matches.iter().map(|(i, _)| *i).collect();
            assignments.extend(matches.into_iter().map(|(i, j)| (indices[i], tracks[j])));
            unmatched.extend(
                indices
                    .into_iter()
                    .enumerate()
                    .filter(|(i, _)| !used.contains(i))
                    .map(|(_, i)| i),
            );
        }
        let mut output = vec![];
        for (i, j) in assignments {
            let d = detections[i];
            let state = &mut self.tracks[j];
            let moving = u8::from(distance(&d[..4], &state.bbox) > 1600.);
            state.bbox = [d[0], d[1], d[2], d[3]];
            state.last = self.frame;
            output.push([
                d[0],
                d[1],
                d[2],
                d[3],
                d[4],
                d[5],
                state.id as f64,
                moving as f64,
            ]);
        }
        for i in unmatched {
            let d = detections[i];
            let id = self.next;
            self.next += 1;
            self.tracks.push(State {
                id,
                bbox: [d[0], d[1], d[2], d[3]],
                class: d[5] as i64,
                last: self.frame,
            });
            output.push([d[0], d[1], d[2], d[3], d[4], d[5], id as f64, 0.]);
        }
        if self.frame % 10 == 0 {
            self.tracks
                .retain(|s| s.last >= self.frame.saturating_sub(10));
        }
        output
    }
}
fn distance(a: &[f64], b: &[f64]) -> f64 {
    let x = (a[0] + a[2] - b[0] - b[2]) * 0.5;
    let y = (a[1] + a[3] - b[1] - b[3]) * 0.5;
    x * x + y * y
}
fn cost(a: &Detection, b: &[f64; 4]) -> f64 {
    let inter =
        (a[2].min(b[2]) - a[0].max(b[0])).max(0.) * (a[3].min(b[3]) - a[1].max(b[1])).max(0.);
    let union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    0.5 * (1. - if union > 0. { inter / union } else { 0. })
        + 0.5 * (distance(&a[..4], b) / 6400.).min(1.)
}
fn assign(cost: &[Vec<f64>], threshold: f64) -> Vec<(usize, usize)> {
    if cost.is_empty() || cost[0].is_empty() {
        return vec![];
    }
    let mut remaining: BTreeMap<usize, ()> = cost
        .iter()
        .enumerate()
        .filter(|(_, r)| r.iter().any(|v| *v <= threshold))
        .map(|(i, _)| (i, ()))
        .collect();
    let mut result = vec![];
    while let Some((&first, _)) = remaining.first_key_value() {
        let mut rows = HashSet::from([first]);
        let mut cols = HashSet::new();
        let mut changed = true;
        while changed {
            changed = false;
            for row in rows.clone() {
                for (col, &v) in cost[row].iter().enumerate() {
                    if v <= threshold && cols.insert(col) {
                        changed = true
                    }
                }
            }
            for (row, values) in cost.iter().enumerate() {
                if cols.iter().any(|&col| values[col] <= threshold) && rows.insert(row) {
                    changed = true
                }
            }
        }
        let mut rows: Vec<_> = rows.into_iter().collect();
        rows.sort();
        for row in &rows {
            remaining.remove(row);
        }
        let mut cols: Vec<_> = cols.into_iter().collect();
        cols.sort();
        let transposed = rows.len() > cols.len();
        let (n, m) = if transposed {
            (cols.len(), rows.len())
        } else {
            (rows.len(), cols.len())
        };
        let mut matrix = vec![vec![0.; m]; n];
        for (i, row) in matrix.iter_mut().enumerate() {
            for (j, value) in row.iter_mut().enumerate() {
                let (r, c) = if transposed {
                    (rows[j], cols[i])
                } else {
                    (rows[i], cols[j])
                };
                *value = if cost[r][c] <= threshold {
                    cost[r][c]
                } else {
                    (threshold + 1.).max(1.)
                };
            }
        }
        let mut component_matches = Vec::new();
        for (i, j) in hungarian(&matrix) {
            let (r, c) = if transposed {
                (rows[j], cols[i])
            } else {
                (rows[i], cols[j])
            };
            if cost[r][c] <= threshold {
                component_matches.push((r, c));
            }
        }
        component_matches.sort();
        result.extend(component_matches);
    }
    result
}
fn hungarian(cost: &[Vec<f64>]) -> Vec<(usize, usize)> {
    let n = cost.len();
    if n == 0 {
        return vec![];
    }
    let m = cost[0].len();
    let (mut u, mut v, mut p, mut way) = (
        vec![0.; n + 1],
        vec![0.; m + 1],
        vec![0usize; m + 1],
        vec![0usize; m + 1],
    );
    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0;
        let mut min = vec![f64::INFINITY; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = f64::INFINITY;
            let mut j1 = 0;
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < min[j] {
                        min[j] = cur;
                        way[j] = j0;
                    }
                    if min[j] < delta {
                        delta = min[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    min[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut matches: Vec<_> = (1..=m)
        .filter(|&j| p[j] > 0)
        .map(|j| (p[j] - 1, j - 1))
        .collect();
    matches.sort();
    matches
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn global_assignment_rejects_forbidden_edges() {
        assert_eq!(
            assign(&[vec![0.1, 0.8], vec![0.2, 0.3]], 0.7),
            vec![(0, 0), (1, 1)]
        );
    }
    #[test]
    fn modes_preserve_ids_and_cross_class_separation() {
        for mode in [TrackerKind::Centroid, TrackerKind::Hungarian] {
            let mut t = Tracker::new(mode);
            let a = t.update(&[[0., 0., 10., 10., 0.9, 5.]]);
            let b = t.update(&[[1., 0., 11., 10., 0.9, 5.]]);
            assert_eq!(a[0][6], b[0][6]);
            let c = t.update(&[[1., 0., 11., 10., 0.9, 8.]]);
            assert_ne!(c[0][6], a[0][6]);
        }
    }
}
