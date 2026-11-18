use crate::{
    config::Rules,
    postprocess::{self, SafetyRow},
};
use anyhow::{Result, ensure};
use geo::{
    BooleanOps, Contains, ConvexHull, Intersects, LineString, MultiPoint, MultiPolygon, Point,
    Polygon,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashSet};

pub type Track = [f64; 8];
pub type Warnings = BTreeMap<String, Value>;
pub type Polygons = Vec<Vec<[f64; 2]>>;
#[derive(Clone, Default, Serialize, Deserialize)]
pub struct HazardResult {
    pub warnings: Warnings,
    pub cone_polys: Polygons,
    pub pole_polys: Polygons,
}
pub struct HazardEngine {
    cone_key: Vec<[u64; 4]>,
    cones: Vec<Polygon>,
    pole_key: Vec<[u64; 4]>,
    poles: MultiPolygon,
    pub cache_hits: u64,
    pub cache_misses: u64,
}
impl Default for HazardEngine {
    fn default() -> Self {
        Self {
            cone_key: vec![],
            cones: vec![],
            pole_key: vec![],
            poles: MultiPolygon::new(vec![]),
            cache_hits: 0,
            cache_misses: 0,
        }
    }
}

fn clusters(points: &[[f64; 2]]) -> Result<Vec<Vec<usize>>> {
    if points.len() < 3 {
        return Ok((0..points.len()).map(|i| vec![i]).collect());
    }
    let data = ndarray::Array2::from_shape_vec(
        (points.len(), 2),
        points.iter().flatten().copied().collect(),
    )?;
    let core = hdbscan_rs::core_distance::compute_core_distances(
        &data.view(),
        &hdbscan_rs::Metric::Euclidean,
        3,
    );
    let edges = hdbscan_rs::mst::prim::prim_mst(
        &data.view(),
        &core.view(),
        &hdbscan_rs::Metric::Euclidean,
        1.,
    );
    let edges = order_density_edges(edges, points.len());
    // hdbscan-rs's condenser replays representative-leaf merges to rebuild
    // the hierarchy. Preserve that convention while controlling tie order.
    let mut uf = hdbscan_rs::union_find::UnionFind::new(points.len());
    let mut linkage = vec![];
    for edge in edges {
        let left = uf.find(edge.u);
        let right = uf.find(edge.v);
        let size = uf.size_of(edge.u) + uf.size_of(edge.v);
        uf.union(left, right);
        linkage.push(hdbscan_rs::SingleLinkageMerge {
            left,
            right,
            distance: edge.weight,
            size,
        });
    }
    let condensed = hdbscan_rs::condensed_tree::build_condensed_tree(&linkage, points.len(), 2);
    let selection = hdbscan_rs::cluster_selection::select_clusters(
        &condensed,
        points.len(),
        hdbscan_rs::ClusterSelectionMethod::Eom,
        0.,
        false,
    );
    let mut selected: Vec<_> = selection.selected_clusters.into_iter().collect();
    selected.sort_unstable();
    let mut parents = std::collections::HashMap::new();
    for edge in &condensed {
        parents.insert(edge.child, edge.parent);
    }
    let labels: Vec<i32> = (0..points.len())
        .map(|point| {
            let mut node = point;
            while let Some(&next) = parents.get(&node) {
                node = next;
                if let Ok(label) = selected.binary_search(&node) {
                    return label as i32;
                }
            }
            -1
        })
        .collect();
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut keys = Vec::new();
    for (index, &label) in labels.iter().enumerate() {
        let key = if label < 0 {
            -(index as i64) - 1
        } else {
            label as i64
        };
        if let Some(group) = keys.iter().position(|k| *k == key) {
            groups[group].push(index)
        } else {
            keys.push(key);
            groups.push(vec![index])
        }
    }
    Ok(groups)
}
// Equal reachability distances form one density level. Attach points to an
// existing component before forming a new component at that same level, so
// binary linkage ordering cannot invent zero-lifetime sibling clusters.
fn order_density_edges(mut edges: Vec<hdbscan_rs::MstEdge>, n: usize) -> Vec<hdbscan_rs::MstEdge> {
    edges.sort_by(|a, b| a.weight.total_cmp(&b.weight));
    let mut parent: Vec<_> = (0..n).collect();
    let mut size = vec![1usize; n];
    let mut result = Vec::with_capacity(edges.len());
    let mut start = 0;
    while start < edges.len() {
        let mut end = start + 1;
        while end < edges.len() && edges[end].weight == edges[start].weight {
            end += 1
        }
        for i in start..end {
            let mut best = i;
            let mut weight = 0;
            for (j, edge) in edges.iter().enumerate().take(end).skip(i) {
                let mut a = edge.u;
                let mut b = edge.v;
                while parent[a] != a {
                    a = parent[a]
                }
                while parent[b] != b {
                    b = parent[b]
                }
                let score = size[a] + size[b];
                if score > weight {
                    best = j;
                    weight = score;
                }
            }
            edges.swap(i, best);
            let edge = edges[i];
            let mut a = edge.u;
            let mut b = edge.v;
            while parent[a] != a {
                a = parent[a]
            }
            while parent[b] != b {
                b = parent[b]
            }
            parent[b] = a;
            size[a] += size[b];
            result.push(edge);
        }
        start = end;
    }
    result
}
fn circle(x: f64, y: f64, r: f64, segments: usize) -> Polygon {
    Polygon::new(
        LineString::from(
            (0..=segments)
                .map(|i| {
                    let a = -(i as f64) * std::f64::consts::TAU / segments as f64;
                    (x + r * a.cos(), y + r * a.sin())
                })
                .collect::<Vec<_>>(),
        ),
        vec![],
    )
}
fn union(polygons: impl IntoIterator<Item = Polygon>) -> MultiPolygon {
    polygons
        .into_iter()
        .fold(MultiPolygon::new(Vec::new()), |result, p| result.union(&p))
}
fn capsule(a: [f64; 2], b: [f64; 2], r: f64) -> Polygon {
    let theta = (b[1] - a[1]).atan2(b[0] - a[0]);
    let mut points = Vec::new();
    for (center, start) in [
        (b, theta - std::f64::consts::FRAC_PI_2),
        (a, theta + std::f64::consts::FRAC_PI_2),
    ] {
        for i in 0..=64 {
            let t = start + i as f64 * std::f64::consts::PI / 64.;
            points.push((center[0] + r * t.cos(), center[1] + r * t.sin()));
        }
    }
    if let Some(first) = points.first().copied() {
        points.push(first)
    }
    Polygon::new(LineString::from(points), vec![])
}
fn tangents(a: [f64; 3], b: [f64; 3]) -> Vec<Polygon> {
    let (a, b) = if a[2] >= b[2] { (a, b) } else { (b, a) };
    let d = (b[0] - a[0]).hypot(b[1] - a[1]);
    if d < 1e-9 || d < (a[2] - b[2]).abs() {
        return vec![];
    }
    let alpha = ((a[2] - b[2]) / d).clamp(-1., 1.).acos();
    let theta = (b[1] - a[1]).atan2(b[0] - a[0]);
    [1., -1.]
        .into_iter()
        .map(|sign| {
            let angle = theta + sign * alpha;
            capsule(
                [a[0] + a[2] * angle.cos(), a[1] + a[2] * angle.sin()],
                [b[0] + b[2] * angle.cos(), b[1] + b[2] * angle.sin()],
                0.05,
            )
        })
        .collect()
}
fn mst(poles: &[[f64; 3]]) -> Vec<(usize, usize)> {
    let mut centres = std::collections::HashMap::<(u64, u64), usize>::new();
    let mut groups: Vec<Vec<usize>> = vec![];
    for (index, pole) in poles.iter().enumerate() {
        let key = (
            if pole[0] == 0. { 0 } else { pole[0].to_bits() },
            if pole[1] == 0. { 0 } else { pole[1].to_bits() },
        );
        let next = groups.len();
        let group = *centres.entry(key).or_insert_with(|| {
            groups.push(vec![]);
            next
        });
        groups[group].push(index);
    }
    let representatives: Vec<_> = groups
        .iter()
        .map(|group| {
            *group
                .iter()
                .max_by(|&&a, &&b| poles[a][2].total_cmp(&poles[b][2]).then(b.cmp(&a)))
                .expect("nonempty centre")
        })
        .collect();
    let unique: Vec<_> = representatives.iter().map(|&i| poles[i]).collect();
    let mut result: Vec<_> = groups
        .iter()
        .zip(&representatives)
        .flat_map(|(group, &rep)| {
            group
                .iter()
                .copied()
                .filter(move |&i| i != rep)
                .map(move |i| (rep, i))
        })
        .collect();
    result.extend(
        unique_mst(&unique)
            .into_iter()
            .map(|(a, b)| (representatives[a], representatives[b])),
    );
    result
}
fn unique_mst(poles: &[[f64; 3]]) -> Vec<(usize, usize)> {
    if poles.len() < 2 {
        return vec![];
    }
    let points: Vec<_> = poles
        .iter()
        .map(|v| delaunator::Point { x: v[0], y: v[1] })
        .collect();
    let triangulation = delaunator::triangulate(&points);
    let mut pairs = HashSet::new();
    for face in triangulation.triangles.chunks_exact(3) {
        for (a, b) in [(face[0], face[1]), (face[1], face[2]), (face[2], face[0])] {
            pairs.insert((a.min(b), a.max(b)));
        }
    }
    if pairs.is_empty() {
        for a in 0..poles.len() {
            for b in a + 1..poles.len() {
                pairs.insert((a, b));
            }
        }
    }
    let mut edges: Vec<_> = pairs
        .into_iter()
        .map(|(a, b)| {
            let d = ((poles[a][0] - poles[b][0]).hypot(poles[a][1] - poles[b][1])
                - poles[a][2]
                - poles[b][2])
                .max(0.);
            (d, a, b)
        })
        .collect();
    edges.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(&b.2)));
    let mut parent: Vec<_> = (0..poles.len()).collect();
    let mut result = Vec::new();
    for (_, a, b) in edges {
        let (mut x, mut y) = (a, b);
        while parent[x] != x {
            x = parent[x]
        }
        while parent[y] != y {
            y = parent[y]
        }
        if x != y {
            parent[x] = y;
            result.push((a, b));
        }
    }
    result
}
fn coordinates(polygons: &[Polygon]) -> Polygons {
    polygons
        .iter()
        .filter(|p| p.exterior().0.len() >= 4)
        .map(|p| p.exterior().0.iter().map(|c| [c.x, c.y]).collect())
        .collect()
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
            let key: Vec<_> = tracks
                .iter()
                .filter(|r| r[5] == 6.)
                .map(|r| {
                    [
                        r[0].to_bits(),
                        r[1].to_bits(),
                        r[2].to_bits(),
                        r[3].to_bits(),
                    ]
                })
                .collect();
            if key == self.cone_key {
                self.cache_hits += 1;
            } else {
                self.cache_misses += 1;
                self.cones.clear();
                let points: Vec<_> = tracks
                    .iter()
                    .filter(|r| r[5] == 6.)
                    .map(|r| [(r[0] + r[2]) * 0.5, (r[1] + r[3]) * 0.5])
                    .collect();
                if points.len() >= 3 {
                    for group in clusters(&points)? {
                        if group.len() >= 3 {
                            self.cones.push(
                                MultiPoint::from(
                                    group
                                        .iter()
                                        .map(|&i| Point::new(points[i][0], points[i][1]))
                                        .collect::<Vec<_>>(),
                                )
                                .convex_hull(),
                            );
                        }
                    }
                }
                self.cone_key = key;
            }
            let count = count_people(&self.cones, &tracks);
            if count > 0 {
                out.warnings.insert(
                    "warning_people_in_controlled_area".into(),
                    json!({"count":count}),
                );
            }
            out.cone_polys = coordinates(&self.cones);
        }
        if rules.detect_in_utility_pole_restricted_area {
            let key: Vec<_> = tracks
                .iter()
                .filter(|r| r[5] == 9.)
                .map(|r| {
                    [
                        r[0].to_bits(),
                        r[1].to_bits(),
                        r[2].to_bits(),
                        r[3].to_bits(),
                    ]
                })
                .collect();
            if key == self.pole_key {
                self.cache_hits += 1;
            } else {
                self.cache_misses += 1;
                let poles: Vec<_> = tracks
                    .iter()
                    .filter(|r| r[5] == 9. && r[3] > r[1])
                    .map(|r| [(r[0] + r[2]) * 0.5, r[3], (r[3] - r[1]) * 0.35])
                    .collect();
                let points: Vec<_> = poles.iter().map(|p| [p[0], p[1]]).collect();
                let groups = clusters(&points)?;
                let mut shapes = Vec::new();
                for group in groups {
                    let cluster: Vec<_> = group.iter().map(|&i| poles[i]).collect();
                    shapes.extend(cluster.iter().map(|p| circle(p[0], p[1], p[2], 256)));
                    for (a, b) in mst(&cluster) {
                        shapes.extend(tangents(cluster[a], cluster[b]));
                    }
                }
                self.poles = union(shapes);
                self.pole_key = key;
            }
            let count = count_people(&self.poles.0, &tracks);
            if count > 0 {
                out.warnings.insert(
                    "warning_people_in_utility_pole_controlled_area".into(),
                    json!({"count":count}),
                );
            }
            out.pole_polys = coordinates(&self.poles.0);
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
    fn density_noise_matches_sklearn_fixture() {
        let points = [
            [192.5, 227.],
            [498.5, 290.],
            [434.5, 287.],
            [460., 165.],
            [154., 64.],
            [260., 337.],
        ];
        assert_eq!(
            clusters(&points).unwrap(),
            vec![vec![0], vec![1], vec![2], vec![3], vec![4], vec![5]]
        );
    }
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
}
