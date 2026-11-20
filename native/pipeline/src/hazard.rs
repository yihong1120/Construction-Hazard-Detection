use crate::{
    config::Rules,
    postprocess::{self, SafetyRow},
};
use anyhow::{Result, ensure};
use construction_postprocess_core::geometry::{circle, cone_regions, coordinates, pole_regions};
use geo::{Contains, Intersects, LineString, MultiPolygon, Point, Polygon};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{
    collections::{BTreeMap, HashSet},
    sync::Arc,
};

pub type Track = [f64; 8];
pub type Warnings = BTreeMap<String, Value>;
pub use construction_postprocess_core::geometry::Polygons;
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct HazardResult {
    pub warnings: Warnings,
    pub cone_polys: Polygons,
    pub pole_polys: Polygons,
}
pub struct HazardEngine {
    cone_points: Vec<[f64; 2]>,
    cones: Vec<Polygon>,
    cone_coordinates: Polygons,
    pole_circles: Vec<[f64; 3]>,
    poles: MultiPolygon,
    pole_coordinates: Polygons,
    pub cache_hits: u64,
    pub cache_misses: u64,
}
impl Default for HazardEngine {
    fn default() -> Self {
        Self {
            cone_points: vec![],
            cones: vec![],
            cone_coordinates: Polygons::default(),
            pole_circles: vec![],
            poles: MultiPolygon::new(vec![]),
            pole_coordinates: Polygons::default(),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

fn same_geometry<const N: usize>(
    input: impl Iterator<Item = [f64; N]>,
    previous: &[[f64; N]],
) -> bool {
    // Clustering and MST tie-breaks depend on detection order. Compare the
    // exact ordered geometry inputs without rounding or building a new key.
    let bits = |point: [f64; N]| point.map(f64::to_bits);
    input.map(bits).eq(previous.iter().copied().map(bits))
}
fn count_people(polygons: &[Polygon], tracks: &[Track]) -> usize {
    let mut unique = HashSet::new();
    for p in tracks.iter().filter(|p| p[5] == 5.) {
        let x = (p[0] + p[2]) * 0.5;
        let y = (p[1] + p[3]) * 0.5;
        if polygons.iter().any(|poly| poly.contains(&Point::new(x, y))) {
            unique.insert((x.to_bits(), y.to_bits()));
        }
    }
    unique.len()
}
impl HazardEngine {
    pub fn detect(&mut self, tracks: &[Track], rules: &Rules) -> Result<HazardResult> {
        ensure!(
            tracks.len() <= 3000 && tracks.iter().flatten().all(|v| v.is_finite()),
            "Invalid tracked detections"
        );
        let tracks: Vec<Track> = tracks
            .iter()
            .filter(|r| !((r[5] == 8. || r[5] == 10.) && r[6] == -1. && r[7] != 1.))
            .map(|r| {
                let mut n = *r;
                n[0] = r[0].min(r[2]);
                n[2] = r[0].max(r[2]);
                n[1] = r[1].min(r[3]);
                n[3] = r[1].max(r[3]);
                n
            })
            .collect();
        let rows: Vec<_> = tracks
            .iter()
            .map(|r| SafetyRow {
                bbox: [r[0], r[1], r[2], r[3]],
                class: r[5],
                track_id: (r[6] >= 0.).then(|| format!("{}", r[6] as u64)),
            })
            .collect();
        let safety = postprocess::safety(
            &rows,
            rules.detect_no_safety_vest_or_helmet,
            rules.detect_near_machinery_or_vehicle,
        );
        let mut out = HazardResult::default();
        for (key, count) in [
            ("warning_no_hardhat", safety.no_hardhat),
            ("warning_no_safety_vest", safety.no_vest),
        ] {
            if count > 0 {
                out.warnings.insert(key.into(), json!({"count":count}));
            }
        }
        for (key, indices) in [
            ("warning_close_to_machinery", safety.machinery),
            ("warning_close_to_vehicle", safety.vehicle),
        ] {
            if !indices.is_empty() {
                out.warnings.insert(key.into(),json!({"count":indices.len(),
                "person_bboxes":indices.iter().map(|&i|rows[i].bbox).collect::<Vec<_>>(),
                "person_track_ids":indices.iter().filter_map(|&i|rows[i].track_id.clone()).collect::<Vec<_>>()}));
            }
        }
        if rules.detect_in_restricted_area {
            let points = || {
                tracks
                    .iter()
                    .filter(|r| r[5] == 6.)
                    .map(|r| [(r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5])
            };
            if same_geometry(points(), &self.cone_points) {
                self.cache_hits += 1;
            } else {
                self.cache_misses += 1;
                let points: Vec<_> = points().collect();
                let cones = cone_regions(&points)?;
                self.cone_coordinates = coordinates(&cones);
                self.cones = cones;
                self.cone_points = points;
            }
            let count = count_people(&self.cones, &tracks);
            if count > 0 {
                out.warnings.insert(
                    "warning_people_in_controlled_area".into(),
                    json!({"count":count}),
                );
            }
            out.cone_polys = Arc::clone(&self.cone_coordinates);
        }
        if rules.detect_in_utility_pole_restricted_area {
            let circles = || {
                tracks
                    .iter()
                    .filter(|r| r[5] == 9. && r[3] > r[1])
                    .map(|r| [(r[0] + r[2]) * 0.5, r[3], (r[3] - r[1]) * 0.35])
            };
            if same_geometry(circles(), &self.pole_circles) {
                self.cache_hits += 1;
            } else {
                self.cache_misses += 1;
                let poles: Vec<_> = circles().collect();
                self.poles = pole_regions(&poles)?;
                self.pole_coordinates = coordinates(&self.poles.0);
                self.pole_circles = poles;
            }
            let count = count_people(&self.poles.0, &tracks);
            if count > 0 {
                out.warnings.insert(
                    "warning_people_in_utility_pole_controlled_area".into(),
                    json!({"count":count}),
                );
            }
            out.pole_polys = Arc::clone(&self.pole_coordinates);
        }
        if rules.detect_machinery_close_to_pole {
            let mut count = 0;
            for pole in tracks.iter().filter(|r| r[5] == 9. && r[3] > r[1]) {
                let height = pole[3] - pole[1];
                let area = circle((pole[0] + pole[2]) * 0.5, pole[3], height * 0.35, 64);
                for machine in tracks.iter().filter(|r| r[5] == 8. || r[5] == 10.) {
                    if pole[1] <= machine[1]
                        && machine[1] <= pole[1] + 2. * height / 3.
                        && area.intersects(&LineString::from(vec![
                            (machine[0], machine[3]),
                            (machine[2], machine[3]),
                        ]))
                    {
                        count += 1;
                    }
                }
            }
            if count > 0 {
                out.warnings.insert(
                    "detect_machinery_close_to_pole".into(),
                    json!({"count":count}),
                );
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn ppe_and_region_flags_count_people_on_current_frame() {
        let mut engine = HazardEngine::default();
        let mut rules = Rules::default();
        rules.detect_in_utility_pole_restricted_area = true;
        let tracks = [
            [0., 0., 10., 20., 0.9, 2., 0., 0.],
            [0., 0., 10., 20., 0.9, 4., 1., 0.],
            [95., 90., 105., 100., 0.9, 5., 2., 0.],
            [95., 0., 105., 100., 0.9, 9., 3., 0.],
        ];
        let first = engine.detect(&tracks, &rules).unwrap();
        assert_eq!(first.warnings["warning_no_hardhat"]["count"], 1);
        assert_eq!(
            first.warnings["warning_people_in_utility_pole_controlled_area"]["count"],
            1
        );
        let second = engine.detect(&tracks[3..], &rules).unwrap();
        assert!(
            !second
                .warnings
                .contains_key("warning_people_in_utility_pole_controlled_area")
        );
        assert!(engine.cache_hits > 0);
    }

    #[test]
    fn geometry_snapshots_share_exact_centres_and_ignore_flat_poles() {
        let mut engine = HazardEngine::default();
        let rules = Rules {
            detect_in_utility_pole_restricted_area: true,
            ..Rules::default()
        };
        let points = [
            [0., 0.],
            [10., 0.],
            [0., 10.],
            [10., 10.],
            [200., 200.],
            [210., 200.],
            [200., 210.],
            [210., 210.],
        ];
        let mut tracks: Vec<_> = points
            .iter()
            .map(|&[x, y]| [x - 1., y - 1., x + 1., y + 1., 0.9, 6., -1., 0.])
            .collect();
        tracks.extend([
            [4., 4., 6., 6., 0.9, 5., 1., 0.],
            [95., 90., 105., 100., 0.9, 5., 2., 0.],
            [95., 0., 105., 100., 0.9, 9., 3., 0.],
        ]);
        let first = engine.detect(&tracks, &rules).unwrap();
        assert!(!first.cone_polys.is_empty());
        assert!(!first.pole_polys.is_empty());
        assert_eq!(
            first.warnings["warning_people_in_controlled_area"]["count"],
            1
        );
        assert_eq!(
            first.warnings["warning_people_in_utility_pole_controlled_area"]["count"],
            1
        );
        assert_eq!(engine.cache_misses, 2);
        for r in &mut tracks[..points.len()] {
            r[0] -= 1.;
            r[1] -= 2.;
            r[2] += 1.;
            r[3] += 2.;
        }
        tracks[8] = [30., 30., 34., 34., 0.9, 5., 1., 0.];
        tracks[10][0] = 90.;
        tracks[10][2] = 110.;
        tracks.push([10., 50., 14., 50., 0.9, 9., 4., 0.]);

        let second = engine.detect(&tracks, &rules).unwrap();
        assert_eq!(engine.cache_hits, 2);
        assert_eq!(engine.cache_misses, 2);
        assert!(Arc::ptr_eq(&first.cone_polys, &second.cone_polys));
        assert!(Arc::ptr_eq(&first.pole_polys, &second.pole_polys));
        assert!(
            !second
                .warnings
                .contains_key("warning_people_in_controlled_area")
        );
        assert_eq!(
            second.warnings["warning_people_in_utility_pole_controlled_area"]["count"],
            1
        );

        let geometry = serde_json::to_value(&second).unwrap();
        assert_eq!(geometry["cone_polys"], json!(first.cone_polys.as_ref()));
        assert_eq!(geometry["pole_polys"], json!(first.pole_polys.as_ref()));

        // A different bottom/radius invalidates the pole cache and keeps
        // previous snapshots readable while a renderer still holds them.
        tracks[10][3] = 150.;
        let third = engine.detect(&tracks, &rules).unwrap();
        assert_eq!(engine.cache_hits, 3);
        assert_eq!(engine.cache_misses, 3);
        assert!(!Arc::ptr_eq(&second.pole_polys, &third.pole_polys));
        assert_eq!(geometry["pole_polys"], json!(first.pole_polys.as_ref()));
    }
}
