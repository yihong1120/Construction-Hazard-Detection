from __future__ import annotations

import math

import numpy as np
from scipy.spatial import Delaunay
from scipy.spatial import QhullError
from shapely.geometry import LineString
from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon
from shapely.ops import unary_union
from sklearn.cluster import HDBSCAN

# ---- Geometry/tuning constants for utility pole area construction ----
CIRCLE_BUFFER_SEGMENTS: int = 64
TANGENT_BUFFER_WIDTH: float = 0.05
TANGENT_BUFFER_SEGMENTS: int = 32
UTILITY_POLE_RADIUS_FACTOR: float = 0.35


def normalise_bbox(bbox: list[float]) -> list[float]:
    """Normalises the bounding box coordinates.

    Args:
        bbox (List[float]): The bounding box coordinates.

    Returns:
        List[float]: Normalised coordinates.
    """
    left_x = min(bbox[0], bbox[2])
    right_x = max(bbox[0], bbox[2])
    top_y = min(bbox[1], bbox[3])
    bottom_y = max(bbox[1], bbox[3])
    return [left_x, top_y, right_x, bottom_y, *bbox[4:]]


def detect_polygon_from_cones(
    datas: list[list[float]],
    clusterer: HDBSCAN,
) -> list[Polygon]:
    """Detects polygons from the safety cones in the detection data.

    Args:
        datas (list[list[float]]): The detection data.

    Returns:
        list[Polygon]: A list of polygons formed by the safety cones.
    """
    if not datas:
        return []

    # Get positions of safety cones
    cone_positions = np.array(
        [
            (
                (float(data[0]) + float(data[2])) / 2,
                (float(data[1]) + float(data[3])) / 2,
            )
            for data in datas
            if data[5] == 6
        ],
    )

    # Check if there are at least three safety cones to form a polygon
    if len(cone_positions) < 3:
        return []

    # Cluster the safety cones
    labels = clusterer.fit_predict(cone_positions)

    # Extract clusters
    clusters: dict[int, list[np.ndarray]] = {}
    for point, label in zip(cone_positions, labels):
        if label == -1:
            continue  # Skip noise points
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(point)

    # Create polygons from clusters
    polygons = []
    for cluster_points in clusters.values():
        if len(cluster_points) >= 3:
            polygon = MultiPoint(cluster_points).convex_hull
            polygons.append(polygon)

    return polygons


def calculate_people_in_controlled_area(
    polygons: list[Polygon],
    datas: list[list[float]],
) -> int:
    """Calculates the number of people within the safety cone area.

    Args:
        polygons (list[Polygon]): Polygons representing controlled areas.
        datas (list[list[float]]): The detection data.

    Returns:
        int: The number of people within the controlled area.
    """
    # Check if there are any detections
    if not datas:
        return 0

    # Check if there are valid polygons
    if not polygons:
        return 0

    # Use a set to track unique people
    unique_people = set()

    # Count the number of people within the controlled area
    for data in datas:
        if data[5] == 5:  # Check if it's a person
            x_center = (data[0] + data[2]) / 2
            y_center = (data[1] + data[3]) / 2
            point = Point(x_center, y_center)
            for polygon in polygons:
                if polygon.contains(point):
                    # Update the set of unique people
                    unique_people.add((x_center, y_center))
                    break  # No need to check other polygons

    return len(unique_people)


def build_utility_pole_union(
    datas: list[list[float]],
    clusterer: HDBSCAN,
) -> Polygon:
    """Builds a union Polygon representing the controlled area for utility
    poles.

    This method clusters detected utility poles, constructs minimum
    spanning trees (MST), calculates outer tangents, and unions the
    resulting polygons to form the final controlled area.

    Args:
        datas (list[list[float]]): Detection data, each entry is a list
            of floats representing bounding box and class info.
        clusterer (HDBSCAN): Clustering algorithm instance for grouping
            utility poles.

    Returns:
        Polygon: The union of all utility pole controlled areas. May be
            empty if no poles are detected.

    Notes:
        - If only one utility pole is detected, returns a single buffered
          circle.
        - If the number of poles is less than clusterer.min_samples,
          unions all circles directly.
        - Otherwise, clusters poles, builds MSTs, and unions circles and
          tangents.
    """
    # 1) Collect poles (centre-bottom and radius derived from bbox)
    utility_poles = _extract_utility_poles(datas)
    if not utility_poles:
        return Polygon()

    # 2) Handle trivial cases
    if len(utility_poles) == 1:
        cx, cy, r = utility_poles[0]
        return Point(cx, cy).buffer(r, quad_segs=64)

    # 3) Too few poles for clustering → union all circles directly
    if len(utility_poles) < clusterer.min_samples:
        return _union_circles(utility_poles)

    # 4) Cluster poles and build per-cluster unions
    clusters = _cluster_utility_poles(utility_poles, clusterer)
    cluster_polys: list[Polygon] = [
        _build_cluster_union(circles_in_cluster)
        for circles_in_cluster in clusters.values()
    ]
    return unary_union(cluster_polys)


def _extract_utility_poles(
    datas: list[list[float]],
) -> list[tuple[float, float, float]]:
    """Extract utility pole centre-bottom and radius from detections.

    Args:
        datas (list[list[float]]): Detection data, each entry is a list
            of floats representing bounding box and class info.

    Returns:
        list[tuple[float, float, float]]: List of utility pole
            centres and radii (cx, cy, r).
    """
    poles: list[tuple[float, float, float]] = []
    for d in datas:
        if d[5] == 9:  # class == 9 => utility pole
            left, top, right, bottom, *_ = d
            cx: float = (left + right) / 2.0
            cy: float = bottom
            height: float = bottom - top
            radius: float = UTILITY_POLE_RADIUS_FACTOR * height
            if radius > 0:
                poles.append((cx, cy, radius))
    return poles


def _union_circles(poles: list[tuple[float, float, float]]) -> Polygon:
    """Union buffered circles for all given poles.

    Args:
        poles (list[tuple[float, float, float]]): List of utility pole
            centres and radii (cx, cy, r).

    Returns:
        Polygon: The union polygon for the cluster.
    """
    circle_polys: list[Polygon] = [
        Point(cx, cy).buffer(r, quad_segs=CIRCLE_BUFFER_SEGMENTS)
        for (cx, cy, r) in poles
    ]
    return unary_union(circle_polys)


def _cluster_utility_poles(
    poles: list[tuple[float, float, float]],
    clusterer: HDBSCAN,
) -> dict[str | int, list[tuple[float, float, float]]]:
    """Cluster poles using HDBSCAN, grouping noise separately.

    Args:
        poles (list[tuple[float, float, float]]): List of utility pole
            centres and radii (cx, cy, r).
        clusterer (HDBSCAN): HDBSCAN clustering algorithm instance.

    Returns:
        dict[str | int, list[tuple[float, float, float]]]: Clusters of
            utility poles, keyed by cluster label.
    """
    # Prepare data for clustering
    coords: np.ndarray = np.array([(p[0], p[1]) for p in poles])
    labels: np.ndarray = clusterer.fit_predict(coords)

    # Map labels to original poles
    clusters: dict[str | int, list[tuple[float, float, float]]] = {}
    for idx, (circle, label) in enumerate(zip(poles, labels)):
        if label == -1:
            key: str = f"noise_{idx}"
            clusters.setdefault(key, []).append(circle)
        else:
            clusters.setdefault(int(label), []).append(circle)
    return clusters


def _build_cluster_union(
    circles_in_cluster: list[tuple[float, float, float]],
) -> Polygon:
    """Build union polygon for a cluster of poles (circles + tangents).

    Args:
        circles_in_cluster (list[tuple[float, float, float]]):
            List of circles represented by (cx, cy, radius) tuples.

    Returns:
        Polygon: The union polygon for the cluster.
    """
    # Single pole: just a buffered circle
    if len(circles_in_cluster) == 1:
        cx, cy, r = circles_in_cluster[0]
        return Point(cx, cy).buffer(
            r,
            quad_segs=CIRCLE_BUFFER_SEGMENTS,
        )

    # Multiple poles: circles + outer tangents along MST edges
    circle_polys_: list[Polygon] = [
        Point(cx, cy).buffer(r, quad_segs=CIRCLE_BUFFER_SEGMENTS)
        for (cx, cy, r) in circles_in_cluster
    ]
    tangent_buffers = _build_mst_tangent_buffers(circles_in_cluster)
    return unary_union(circle_polys_ + tangent_buffers)


def _build_mst_tangent_buffers(
    circles_in_cluster: list[tuple[float, float, float]],
) -> list[Polygon]:
    """Build buffered polygons from outer tangents along MST edges.

    Args:
        circles_in_cluster (list[tuple[float, float, float]]):
            List of circles represented by (cx, cy, radius) tuples.

    Returns:
        list[Polygon]: List of buffered polygons.
    """
    mst_edges: list[tuple[int, int]] = build_mst_pairs(
        circles_in_cluster,
    )
    lines: list[LineString] = []
    for u, v in mst_edges:
        cx1, cy1, r1 = circles_in_cluster[u]
        cx2, cy2, r2 = circles_in_cluster[v]
        lines.extend(get_outer_tangents(cx1, cy1, r1, cx2, cy2, r2))
    return [
        ls.buffer(
            TANGENT_BUFFER_WIDTH,
            quad_segs=TANGENT_BUFFER_SEGMENTS,
        )
        for ls in lines
    ]


def build_mst_pairs(
    poles: list[tuple[float, float, float]],
) -> list[tuple[int, int]]:
    """Builds a minimum spanning tree (MST) for a set of utility poles.

    Args:
        poles (list[tuple[float, float, float]]): List of utility pole
            centres and radii (cx, cy, r).

    Returns:
        list[tuple[int, int]]: List of MST edges as index pairs.

    Notes:
        - Uses Euclidean distance minus radii as edge weights.
        - Returns edges as index pairs for use in tangent calculation.
    """
    count = len(poles)
    if count < 2:
        return []
    if count == 2:
        return [(0, 1)]

    # Coincident detections make Qhull fall back to quadratic Prim.  A largest
    # radius representative preserves the minimum external edge weight for
    # each centre; the remaining detections join it with zero-cost edges.
    centres: dict[tuple[float, float], list[int]] = {}
    for index, (center_x, center_y, _radius) in enumerate(poles):
        centres.setdefault((center_x, center_y), []).append(index)

    representative_indices: list[int] = []
    duplicate_edges: list[tuple[int, int]] = []
    for indices in centres.values():
        representative = max(indices, key=lambda index: poles[index][2])
        representative_indices.append(representative)
        duplicate_edges.extend(
            (representative, index)
            for index in indices
            if index != representative
        )

    if len(representative_indices) == 1:
        return duplicate_edges

    unique_poles = [poles[index] for index in representative_indices]
    points = np.asarray([(pole[0], pole[1]) for pole in unique_poles])
    try:
        triangulation = Delaunay(points, qhull_options='QJ')
    except QhullError:
        # Degenerate (for example, coincident) detections are unusual.
        # Keep the former exact result in that case without allocating a
        # complete graph.
        unique_edges = _dense_prim_mst_pairs(unique_poles)
        return duplicate_edges + [
            (representative_indices[left], representative_indices[right])
            for left, right in unique_edges
        ]

    candidate_pairs: set[tuple[int, int]] = set()
    for simplex in triangulation.simplices:
        for left, right in (
            (int(simplex[0]), int(simplex[1])),
            (int(simplex[1]), int(simplex[2])),
            (int(simplex[0]), int(simplex[2])),
        ):
            candidate_pairs.add((min(left, right), max(left, right)))

    weighted_edges = sorted(
        (
            max(
                0.0,
                math.dist(points[left], points[right])
                - (unique_poles[left][2] + unique_poles[right][2]),
            ),
            left,
            right,
        )
        for left, right in candidate_pairs
    )
    unique_count = len(unique_poles)
    parents = list(range(unique_count))

    def find(index: int) -> int:
        """Perform find.

        Args:
            index: Value used by this callable.

        Returns:
            The callable result.
        """
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    edges: list[tuple[int, int]] = []
    for _weight, left, right in weighted_edges:
        left_root, right_root = find(left), find(right)
        if left_root == right_root:
            continue
        parents[left_root] = right_root
        edges.append((left, right))
        if len(edges) == unique_count - 1:
            return duplicate_edges + [
                (representative_indices[first], representative_indices[second])
                for first, second in edges
            ]

    # Qhull normally produces a connected triangulation.  Preserve an
    # exact, low-memory fallback if malformed input does not.
    unique_edges = _dense_prim_mst_pairs(unique_poles)
    return duplicate_edges + [
        (representative_indices[left], representative_indices[right])
        for left, right in unique_edges
    ]


def _dense_prim_mst_pairs(
    poles: list[tuple[float, float, float]],
) -> list[tuple[int, int]]:
    """Build an exact MST with O(n) auxiliary memory for rare fallbacks."""
    count = len(poles)
    selected = [False] * count
    best_weight = [float('inf')] * count
    parents = [-1] * count
    best_weight[0] = 0.0
    edges: list[tuple[int, int]] = []

    for _ in range(count):
        current = min(
            (index for index in range(count) if not selected[index]),
            key=best_weight.__getitem__,
        )
        selected[current] = True
        if parents[current] >= 0:
            edges.append((parents[current], current))
        cx, cy, radius = poles[current]
        for candidate, (other_x, other_y, other_radius) in enumerate(poles):
            if selected[candidate]:
                continue
            weight = max(
                0.0,
                math.dist((cx, cy), (other_x, other_y))
                - (radius + other_radius),
            )
            if weight < best_weight[candidate]:
                best_weight[candidate] = weight
                parents[candidate] = current
    return edges


def get_outer_tangents(
    cx1: float,
    cy1: float,
    r1: float,
    cx2: float,
    cy2: float,
    r2: float,
    eps: float = 1e-9,
) -> list[LineString]:
    """Calculates the outer tangents between two circles.

    Args:
        cx1 (float): Centre x-coordinate of the first circle.
        cy1 (float): Centre y-coordinate of the first circle.
        r1 (float): Radius of the first circle.
        cx2 (float): Centre x-coordinate of the second circle.
        cy2 (float): Centre y-coordinate of the second circle.
        r2 (float): Radius of the second circle.
        eps (float): Small epsilon to avoid division by zero.

    Returns:
        list[LineString]: List of LineString objects representing outer
            tangents.

    Notes:
        - Returns empty list if circles overlap or are coincident.
        - Ensures r1 >= r2 for calculation stability.
    """
    dx: float = cx2 - cx1
    dy: float = cy2 - cy1
    d2: float = dx * dx + dy * dy
    d: float = math.sqrt(d2)
    if d < abs(r1 - r2):
        return []  # Circles overlap, no outer tangents
    if d < eps:
        return []  # Circles are coincident

    # Ensure r1 >= r2 for calculation stability
    if r2 > r1:
        cx1, cx2 = cx2, cx1
        cy1, cy2 = cy2, cy1
        r1, r2 = r2, r1
        dx, dy = -dx, -dy

    d2 = (cx2 - cx1) ** 2 + (cy2 - cy1) ** 2
    d = math.sqrt(d2)
    rdiff: float = r1 - r2
    if d < rdiff:
        return []  # Circles overlap

    alpha: float = math.acos(rdiff / d)
    theta: float = math.atan2((cy2 - cy1), (cx2 - cx1))

    lines: list[LineString] = []
    for sign in [1, -1]:
        phi: float = theta + sign * alpha
        x1t: float = cx1 + r1 * math.cos(phi)
        y1t: float = cy1 + r1 * math.sin(phi)
        x2t: float = cx2 + r2 * math.cos(phi)
        y2t: float = cy2 + r2 * math.sin(phi)
        ls: LineString = LineString(
            [
                (x1t, y1t),
                (x2t, y2t),
            ],
        )
        lines.append(ls)

    return lines


def count_people_in_polygon(
    poly: Polygon,
    datas: list[list[float]],
) -> int:
    """Counts the number of people within a specified polygon.

    Args:
        poly (Polygon): The polygon representing the area of interest.
        datas (list[list[float]]): Detection data, each entry is a list
            of floats representing bounding box and class info.

    Returns:
        int: The number of unique people found within the polygon.

    Notes:
        - Only considers entries with class == 5 (person).
        - Uses centre point of bounding box for inclusion test.
    """
    persons: list[list[float]] = [d for d in datas if d[5] == 5]
    found_people: set[tuple[float, float]] = set()
    for p in persons:
        left, top, right, bottom, *_ = p
        px: float = (left + right) / 2.0
        py: float = (top + bottom) / 2.0
        if poly.contains(Point(px, py)):
            found_people.add((px, py))
    return len(found_people)


def polygons_to_coords(polygons: list[Polygon]) -> list[list[list[float]]]:
    """Converts Polygon or MultiPolygon objects to a list of lists of [x, y]
    coordinates.

    Args:
        polygons (list[Polygon]): List of Polygon or MultiPolygon objects.

    Returns:
        list[list[list[float]]]: List of coordinate lists for each
            polygon.

    Notes:
        - Skips empty polygons.
        - For MultiPolygon, extracts coordinates from each sub-polygon.
    """
    coords_list: list[list[list[float]]] = []
    for poly in polygons:
        if poly.is_empty:
            continue  # Skip empty polygons
        if poly.geom_type == 'Polygon':
            coords_list.append([list(pt) for pt in poly.exterior.coords])
        elif poly.geom_type == 'MultiPolygon':
            for subpoly in poly.geoms:
                if not subpoly.is_empty and subpoly.geom_type == 'Polygon':
                    coords_list.append(
                        [list(pt) for pt in subpoly.exterior.coords],
                    )
    return coords_list
