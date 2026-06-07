from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Self

from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.cluster import HDBSCAN

from src.utils import Utils


_SPATIAL_MIN_CELL_SIZE = 64.0
_SPATIAL_MAX_CELL_SIZE = 256.0


@dataclass
class _SpatialIndex:
    """Grid index used to limit person-to-machine distance checks."""

    cell_size: float
    cells: dict[tuple[int, int], list[list[float]]]


class DangerDetector:
    """
    A class to detect potential safety hazards based on the detection data.
    """

    def __init__(self, detection_items: dict[str, bool] | None = None) -> None:
        """Initialise the danger detector.

        Args:
            detection_items: Optional feature flags for individual safety
                checks. Missing or incomplete mappings enable the default
                checks.
        """
        self.clusterer = HDBSCAN(min_samples=3, min_cluster_size=2, copy=True)

        required_keys = {
            'detect_no_safety_vest_or_helmet',
            'detect_near_machinery_or_vehicle',
            'detect_in_restricted_area',
            'detect_in_utility_pole_restricted_area',
            'detect_machinery_close_to_pole',
        }

        self.detection_items = (
            detection_items
            if detection_items and required_keys.issubset(detection_items)
            else {}
        )

    def detect_danger(
        self,
        datas: list[list[float]],
    ) -> tuple[dict[str, dict[str, object]], list[Polygon], list[Polygon]]:
        """
        Detects potential safety violations in a construction site.

        Returns:
            Tuple[
                dict[str, dict[str, object]],  # warnings
                list[Polygon],              # cone_polygons
                list[Polygon],              # pole_polygons
            ]
        """
        # 0. Filter static machinery / vehicles and normalize bboxes.
        datas = self._filter_and_normalise_static_machinery(datas)
        warnings: dict[str, dict[str, object]] = {}

        # 2. Collect polygons
        cone_polygons_raw: list[Polygon] = []
        pole_polygons_raw: list[Polygon] = []

        # (A) detect_in_restricted_area:
        # Check if personnel enter the controlled area
        # formed by the safety cone
        if (
            not self.detection_items
            or self.detection_items.get('detect_in_restricted_area', False)
        ):
            self.check_cone_restricted_area(datas, warnings, cone_polygons_raw)

        detect_safety = (
            not self.detection_items
            or self.detection_items.get(
                'detect_no_safety_vest_or_helmet', False,
            )
        )
        detect_proximity = (
            not self.detection_items
            or self.detection_items.get(
                'detect_near_machinery_or_vehicle', False,
            )
        )

        persons: list[list[float]] = []
        machinery_vehicles: list[list[float]] = []
        count_no_hardhat = 0
        count_no_vest = 0
        if detect_safety or detect_proximity:
            for detection in datas:
                class_id = detection[5]
                if class_id == 5:
                    persons.append(detection)
                elif class_id == 8 or class_id == 10:
                    machinery_vehicles.append(detection)
                elif detect_safety and class_id == 2:
                    count_no_hardhat += 1
                elif detect_safety and class_id == 4:
                    count_no_vest += 1

        # (C) detect_no_safety_vest_or_helmet
        if detect_safety:
            self.check_safety_violations(
                count_no_hardhat,
                count_no_vest,
                warnings,
            )

        # (D) detect_near_machinery_or_vehicle
        if detect_proximity:
            self.check_proximity_violations(
                persons, machinery_vehicles, warnings,
            )

        # (E) detect_machinery_close_to_pole
        if (
            self.detection_items
            and self.detection_items.get(
                'detect_machinery_close_to_pole', False,
            )
        ):
            self.check_machinery_near_utility_pole(
                datas, warnings, circle_ratio=0.35,
            )

        # (F) detect_in_utility_pole_restricted_area:
        # Check if personnel enter the controlled area
        # formed by the utility pole
        if (
            self.detection_items
            and self.detection_items.get(
                'detect_in_utility_pole_restricted_area', False,
            )
        ):
            self.check_pole_restricted_area(datas, warnings, pole_polygons_raw)

        # 3. Convert polygon coordinates (for front-end visualization)
        cone_polygons_coords = Utils.polygons_to_coords(cone_polygons_raw)
        pole_polygons_coords = Utils.polygons_to_coords(pole_polygons_raw)

        return warnings, cone_polygons_coords, pole_polygons_coords

    # Checks if personnel enter the controlled area formed by the safety cone
    def check_cone_restricted_area(
        self,
        datas: list[list[float]],
        warnings: dict[str, dict[str, object]],
        polygons: list[Polygon],
    ) -> None:
        """
        Checks if personnel enter the controlled area
        formed by the safety cone.

        Arg:
            datas: The input data containing personnel information.
            warnings: A dictionary to store warning messages.
            polygons: A list to store the detected polygon areas.
        """
        new_polygons = Utils.detect_polygon_from_cones(datas, self.clusterer)
        polygons.extend(new_polygons)

        people_count = Utils.calculate_people_in_controlled_area(
            new_polygons, datas,
        )
        if people_count > 0:
            warnings['warning_people_in_controlled_area'] = {
                'count': people_count,
            }

    def check_pole_restricted_area(
        self,
        datas: list[list[float]],
        warnings: dict[str, dict[str, object]],
        pole_polygons: list[Polygon],
    ) -> None:
        """
        Checks if personnel enter the controlled area
        formed by the utility pole.

        Arg:
            datas: The input data containing personnel information.
            warnings: A dictionary to store warning messages.
            pole_polygons: A list to store the detected polygon areas.
        """
        pole_union_poly = Utils.build_utility_pole_union(datas, self.clusterer)
        if not pole_union_poly.is_empty:
            pole_polygons.append(pole_union_poly)

            # Count people in the utility pole controlled area
            count_in_pole_area = Utils.count_people_in_polygon(
                pole_union_poly, datas,
            )
            if count_in_pole_area > 0:
                warnings['warning_people_in_utility_pole_controlled_area'] = {
                    'count': count_in_pole_area,
                }

    def check_safety_violations(
        self,
        count_no_hardhat: int,
        count_no_vest: int,
        warnings: dict[str, dict[str, object]],
    ) -> None:
        """
        Checks for safety violations among personnel.

        Arg:
            datas: The input data containing personnel information.
            warnings: A dictionary to store warning messages.
            polygons: A list to store the detected polygon areas.
        """
        if count_no_hardhat > 0:
            warnings['warning_no_hardhat'] = {'count': count_no_hardhat}

        if count_no_vest > 0:
            warnings['warning_no_safety_vest'] = {'count': count_no_vest}

    # Checks if personnel are dangerously close to machinery/vehicles
    def check_proximity_violations(
        self,
        persons: list[list[float]],
        machinery_vehicles: list[list[float]],
        warnings: dict[str, dict[str, object]],
    ) -> None:
        """Warn when personnel are close to machinery or vehicles.

        Args:
            persons: Person detections in YOLO ``xyxy`` format.
            machinery_vehicles: Machinery and vehicle detections.
            warnings: Warning dictionary updated in place.
        """
        count_machinery = 0
        count_vehicle = 0
        machinery_person_bboxes: list[list[float]] = []
        machinery_person_track_ids: list[str] = []
        vehicle_person_bboxes: list[list[float]] = []
        vehicle_person_track_ids: list[str] = []

        if not persons or not machinery_vehicles:
            return

        spatial_index = self._build_spatial_index(machinery_vehicles)
        for person in persons:
            is_driver = False
            close_to_machinery = False
            close_to_vehicle = False
            for mv in self._nearby_machinery_vehicles(person, spatial_index):
                if self._is_driver_detection(person, mv):
                    is_driver = True
                    break

                class_id = mv[5]
                if (
                    class_id == 8
                    and not close_to_machinery
                    and self._is_dangerously_close_detection(
                        person, mv, 'machinery',
                    )
                ):
                    close_to_machinery = True
                elif (
                    class_id == 10
                    and not close_to_vehicle
                    and self._is_dangerously_close_detection(
                        person, mv, 'vehicle',
                    )
                ):
                    close_to_vehicle = True

            if is_driver:
                continue

            if close_to_machinery:
                count_machinery += 1
                machinery_person_bboxes.append(self._bbox(person))
                self._append_track_id(machinery_person_track_ids, person)
            if close_to_vehicle:
                count_vehicle += 1
                vehicle_person_bboxes.append(self._bbox(person))
                self._append_track_id(vehicle_person_track_ids, person)

        if count_machinery > 0:
            warnings['warning_close_to_machinery'] = {
                'count': count_machinery,
                'person_bboxes': machinery_person_bboxes,
                'person_track_ids': machinery_person_track_ids,
            }

        if count_vehicle > 0:
            warnings['warning_close_to_vehicle'] = {
                'count': count_vehicle,
                'person_bboxes': vehicle_person_bboxes,
                'person_track_ids': vehicle_person_track_ids,
            }

    @staticmethod
    def _append_track_id(track_ids: list[str], detection: list[float]) -> None:
        """Collect a non-empty tracker id from a detection row."""
        if len(detection) < 7 or detection[6] in {None, '', -1}:
            return
        track_ids.append(str(detection[6]))

    @staticmethod
    def _bbox(detection: list[float]) -> list[float]:
        """Return the four co-ordinates from a detection row."""
        return [detection[0], detection[1], detection[2], detection[3]]

    @classmethod
    def _build_spatial_index(
        cls: type[Self],
        machinery_vehicles: list[list[float]],
    ) -> _SpatialIndex:
        """Build a simple grid index for machinery and vehicle boxes."""
        cell_size = cls._spatial_cell_size(machinery_vehicles)
        cells: dict[tuple[int, int], list[list[float]]] = {}
        for item in machinery_vehicles:
            for cell in cls._cells_for_bbox(item, cell_size):
                cells.setdefault(cell, []).append(item)
        return _SpatialIndex(
            cell_size=cell_size,
            cells=cells,
        )

    @staticmethod
    def _spatial_cell_size(machinery_vehicles: list[list[float]]) -> float:
        """Choose a grid size from the median detection dimension.

        Args:
            machinery_vehicles: Machinery and vehicle detection rows.

        Returns:
            Cell size constrained by configured minimum and maximum bounds.
        """
        dimensions = sorted(
            max(abs(item[2] - item[0]), abs(item[3] - item[1]))
            for item in machinery_vehicles
        )
        if not dimensions:
            return _SPATIAL_MIN_CELL_SIZE
        median = dimensions[len(dimensions) // 2]
        return min(
            _SPATIAL_MAX_CELL_SIZE,
            max(_SPATIAL_MIN_CELL_SIZE, median),
        )

    @classmethod
    def _nearby_machinery_vehicles(
        cls: type[Self],
        person: list[float],
        spatial_index: _SpatialIndex,
    ) -> Iterator[list[float]]:
        """Yield machinery and vehicles from grid cells near a person."""
        query_bbox = cls._proximity_query_bbox(person)
        query_bounds = cls._cell_bounds_for_bbox(
            query_bbox,
            spatial_index.cell_size,
        )

        seen: set[int] = set()
        for cell in cls._cells_from_bounds(query_bounds):
            for item in spatial_index.cells.get(cell, ()):
                item_id = id(item)
                if item_id in seen:
                    continue
                seen.add(item_id)
                yield item

    @staticmethod
    def _proximity_query_bbox(person: list[float]) -> list[float]:
        """Expand a person box to the area used for proximity lookup."""
        person_width = person[2] - person[0]
        person_height = person[3] - person[1]
        horizontal_margin = max(1.0, 5 * person_width)
        vertical_margin = max(1.0, 1.5 * person_height)
        return [
            person[0] - horizontal_margin,
            person[1] - vertical_margin,
            person[2] + horizontal_margin,
            person[3] + vertical_margin,
        ]

    @staticmethod
    def _cells_for_bbox(
        bbox: list[float],
        cell_size: float,
    ) -> Iterator[tuple[int, int]]:
        """Yield grid cells touched by a bounding box."""
        yield from DangerDetector._cells_from_bounds(
            DangerDetector._cell_bounds_for_bbox(bbox, cell_size),
        )

    @staticmethod
    def _cell_bounds_for_bbox(
        bbox: list[float],
        cell_size: float,
    ) -> tuple[int, int, int, int]:
        """Convert a bounding box into inclusive grid bounds."""
        left = min(bbox[0], bbox[2])
        right = max(bbox[0], bbox[2])
        top = min(bbox[1], bbox[3])
        bottom = max(bbox[1], bbox[3])
        return (
            int(left // cell_size),
            int(right // cell_size),
            int(top // cell_size),
            int(bottom // cell_size),
        )

    @staticmethod
    def _cells_from_bounds(
        bounds: tuple[int, int, int, int],
    ) -> Iterator[tuple[int, int]]:
        """Yield every grid cell within inclusive bounds."""
        start_x, end_x, start_y, end_y = bounds
        for cell_x in range(start_x, end_x + 1):
            for cell_y in range(start_y, end_y + 1):
                yield (cell_x, cell_y)

    @staticmethod
    def _is_driver_detection(
        person: list[float],
        vehicle: list[float],
    ) -> bool:
        """Return whether a person detection appears to be inside a vehicle."""
        person_bottom_y = person[3]
        person_top_y = person[1]
        person_left_x = person[0]
        person_right_x = person[2]
        person_width = person[2] - person[0]
        person_height = person[3] - person[1]

        vehicle_top_y = vehicle[1]
        vehicle_bottom_y = vehicle[3]
        vehicle_left_x = vehicle[0]
        vehicle_right_x = vehicle[2]
        vehicle_height = vehicle[3] - vehicle[1]

        return (
            person_bottom_y < vehicle_bottom_y
            and vehicle_bottom_y - person_bottom_y >= person_height / 2
            and person_left_x >= vehicle_left_x - person_width / 2
            and person_right_x <= vehicle_right_x + person_width / 2
            and person_top_y > vehicle_top_y
            and person_height <= vehicle_height / 2
        )

    @staticmethod
    def _is_dangerously_close_detection(
        person: list[float],
        machinery_vehicle: list[float],
        label: str,
    ) -> bool:
        """Return whether a person is too close to machinery or a vehicle."""
        person_width = person[2] - person[0]
        person_height = person[3] - person[1]
        person_area = person_width * person_height
        machinery_vehicle_area = (
            (machinery_vehicle[2] - machinery_vehicle[0])
            * (machinery_vehicle[3] - machinery_vehicle[1])
        )
        if machinery_vehicle_area <= 0:
            return False

        acceptable_ratio = 0.1 if label == 'vehicle' else 0.05
        if person_area / machinery_vehicle_area > acceptable_ratio:
            return False

        danger_distance_horizontal = 5 * person_width
        danger_distance_vertical = 1.5 * person_height
        horizontal_distance = min(
            abs(person[2] - machinery_vehicle[0]),
            abs(person[0] - machinery_vehicle[2]),
        )
        vertical_distance = min(
            abs(person[3] - machinery_vehicle[1]),
            abs(person[1] - machinery_vehicle[3]),
        )
        return (
            horizontal_distance <= danger_distance_horizontal
            and vertical_distance <= danger_distance_vertical
        )

    def check_machinery_near_utility_pole(
        self,
        datas: list[list[float]],
        warnings: dict[str, dict[str, object]],
        circle_ratio: float = 3.5,
    ) -> None:
        """
        Checks if machinery/vehicles are near the utility pole.

        Args:
            datas: The input data containing personnel information.
            warnings: A dictionary to store warning messages.
            circle_ratio: The ratio to define the radius of the circle at the
                bottom of the utility pole (default 3.5).
        """
        # 1. Count violations
        poles = [d for d in datas if d[5] == 9]
        machinery_vehicles = [
            d for d in datas if d[5]
            in [8, 10]
        ]

        if not poles or not machinery_vehicles:
            return

        # 2. Count intersections
        intersect_count = 0

        for pole in poles:
            px1, py1, px2, py2, *_ = pole
            pole_height = (py2 - py1)
            if pole_height <= 0:
                continue

            # Compute 2/3 height position
            two_thirds_y = py1 + (2.0/3.0) * pole_height

            # Create the circle at the bottom of the utility pole
            circle_radius = circle_ratio * pole_height
            circle_center = ((px1 + px2) / 2.0, py2)
            pole_circle = Point(circle_center).buffer(circle_radius)

            # 2. Check if machinery/vehicles meet both conditions
            for mv in machinery_vehicles:
                mx1, my1, mx2, my2, *_ = mv
                # Top of the machinery must be within [pole_top, 2/3 height]
                if not (py1 <= my1 <= two_thirds_y):
                    continue

                # Create the bottom line of the machinery
                bottom_line = LineString([(mx1, my2), (mx2, my2)])

                dist_to_circle = bottom_line.distance(pole_circle)

                if dist_to_circle <= 0:
                    # Machinery/vehicle is close to the utility pole
                    intersect_count += 1

        if intersect_count > 0:
            warnings['detect_machinery_close_to_pole'] = {
                'count': intersect_count,
            }

    # Filter static machinery/vehicles
    @staticmethod
    def _filter_static_machinery(
        datas: list[list[float]],
    ) -> list[list[float]]:
        """
        Filter static machinery/vehicles from the input data.

        Args:
            datas: The input data containing machinery/vehicle information.

        Returns:
            A list of filtered machinery/vehicle data.
        """
        return [
            d for d in datas
            if (
                (d[5] in (8, 10) and len(d) > 7 and (d[6] != -1 or d[7] == 1))
                or (d[5] not in (8, 10))
            )
        ]

    @staticmethod
    def _filter_and_normalise_static_machinery(
        datas: list[list[float]],
    ) -> list[list[float]]:
        """
        Filter static machinery/vehicles and normalise bboxes in one pass.
        """
        normalised: list[list[float]] = []
        for detection in datas:
            class_id = detection[5]
            if (
                class_id in (8, 10)
                and len(detection) > 7
                and detection[6] == -1
                and detection[7] != 1
            ):
                continue
            normalised.append(Utils.normalise_bbox(detection))
        return normalised


def main() -> None:
    """
    Main function to demonstrate the usage of the DangerDetector class.
    """
    detector = DangerDetector()

    data: list[list[float]] = [
        [50, 50, 150, 150, 0.95, 0],    # Hardhat
        [200, 200, 300, 300, 0.85, 5],  # Person
        [400, 400, 500, 500, 0.75, 2],  # NO-Hardhat
        [0, 0, 10, 10, 0.88, 6],  # Safety cone
        [0, 1000, 10, 1010, 0.87, 6],  # Safety cone
        [1000, 0, 1010, 10, 0.89, 6],  # Safety cone
        [100, 100, 120, 120, 0.9, 6],  # Safety cone
        [150, 150, 170, 170, 0.85, 6],  # Safety cone
        [200, 200, 220, 220, 0.89, 6],  # Safety cone
        [250, 250, 270, 270, 0.85, 6],  # Safety cone
        [450, 450, 470, 470, 0.92, 6],  # Safety cone
        [500, 500, 520, 520, 0.88, 6],  # Safety cone
        [550, 550, 570, 570, 0.86, 6],  # Safety cone
        [600, 600, 620, 620, 0.84, 6],  # Safety cone
        [650, 650, 670, 670, 0.82, 6],  # Safety cone
        [700, 700, 720, 720, 0.80, 6],  # Safety cone
        [750, 750, 770, 770, 0.78, 6],  # Safety cone
        [800, 800, 820, 820, 0.76, 6],  # Safety cone
        [850, 850, 870, 870, 0.74, 6],  # Safety cone

        [100, 100, 120, 200, 0.9, 9],   # pole
        [200, 180, 230, 210, 0.85, 8],  # machinery
        [180, 190, 195, 205, 0.88, 8],  # machinery
    ]

    warnings, cone_polygons, pole_polygons = detector.detect_danger(data)
    print(f"Warnings: {warnings}")
    print(f"cone_polygons: {cone_polygons}")
    print(f"pole_polygons: {pole_polygons}")


if __name__ == '__main__':
    main()
