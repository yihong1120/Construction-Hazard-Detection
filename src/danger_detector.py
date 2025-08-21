from __future__ import annotations

from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.geometry import Polygon
from sklearn.cluster import HDBSCAN

from src.utils import Utils


class DangerDetector:
    """
    A class to detect potential safety hazards based on the detection data.
    """

    def __init__(self, detection_items: dict[str, bool] = {}):
        """
        Initialises the danger detector.

        Args:
            detection_items (Dict[str, bool]): A dictionary of detection items
                to enable/disable specific safety checks. The keys are:
                - 'detect_no_safety_vest_or_helmet'
                - 'detect_near_machinery_or_vehicle'
                - 'detect_in_restricted_area'
                - 'detect_in_utility_pole_restricted_area'
                - 'detect_machinery_close_to_pole'
        """
        self.clusterer = HDBSCAN(min_samples=3, min_cluster_size=2)

        required_keys = {
            'detect_no_safety_vest_or_helmet',
            'detect_near_machinery_or_vehicle',
            'detect_in_restricted_area',
            'detect_in_utility_pole_restricted_area',
            'detect_machinery_close_to_pole',
        }

        if (
            isinstance(detection_items, dict)
            and all(
                isinstance(k, str) and isinstance(v, bool)
                for k, v in detection_items.items()
            )
            and required_keys.issubset(detection_items.keys())
        ):
            self.detection_items = detection_items
        else:
            self.detection_items = {}

    def detect_danger(
        self,
        datas: list[list[float]],
    ) -> tuple[dict[str, dict[str, int]], list[Polygon], list[Polygon]]:
        """
        Detects potential safety violations in a construction site.

        Returns:
            Tuple[
                dict[str, dict[str, int]],  # warnings
                list[Polygon],              # cone_polygons
                list[Polygon],              # pole_polygons
            ]
        """
        # 0. Filter static machinery / vehicles
        datas = self._filter_static_machinery(datas)

        # 1. Normalize bounding box data
        datas = Utils.normalise_data(datas)
        warnings: dict[str, dict[str, int]] = {}

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

        # (B) Classify data
        persons = [d for d in datas if d[5] == 5]
        hardhat_violations = [d for d in datas if d[5] == 2]
        safety_vest_violations = [d for d in datas if d[5] == 4]
        machinery_vehicles = [d for d in datas if d[5] in [8, 10]]

        # Filter out potential drivers
        if machinery_vehicles:
            persons = [
                p for p in persons
                if not any(
                    Utils.is_driver(p[:4], mv[:4]) for mv in machinery_vehicles
                )
            ]

        # (C) detect_no_safety_vest_or_helmet
        if (
            not self.detection_items
            or self.detection_items.get(
                'detect_no_safety_vest_or_helmet', False,
            )
        ):
            self.check_safety_violations(
                persons, hardhat_violations, safety_vest_violations, warnings,
            )

        # (D) detect_near_machinery_or_vehicle
        if (
            not self.detection_items
            or self.detection_items.get(
                'detect_near_machinery_or_vehicle', False,
            )
        ):
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
        warnings: dict[str, dict[str, int]],
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
        warnings: dict[str, dict[str, int]],
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

    # -------------------------------------------------------------------------
    # Checks if personnel enter the controlled area formed by the utility pole
    # -------------------------------------------------------------------------
    # def check_safety_violations(
    #     self,
    #     persons: list[list[float]],
    #     hardhat_violations: list[list[float]],
    #     safety_vest_violations: list[list[float]],
    #     warnings: dict[str, dict[str, int]],
    # ) -> None:
    #     count_no_hardhat = 0
    #     count_no_vest = 0

    #     for violation in hardhat_violations:
    #         # overlap_percentage > 0.5 視為缺安全帽
    #         if any(
    #             Utils.overlap_percentage(violation[:4], p[:4]) > 0.5
    #             for p in persons
    #         ):
    #             count_no_hardhat += 1

    #     for violation in safety_vest_violations:
    #         # overlap_percentage > 0.5 視為缺背心
    #         if any(
    #             Utils.overlap_percentage(violation[:4], p[:4]) > 0.5
    #             for p in persons
    #         ):
    #             count_no_vest += 1

    #     if count_no_hardhat > 0:
    #         warnings['warning_no_hardhat'] = {'count': count_no_hardhat}

    #     if count_no_vest > 0:
    #         warnings['warning_no_safety_vest'] = {'count': count_no_vest}

    def check_safety_violations(
        self,
        persons: list[list[float]],
        hardhat_violations: list[list[float]],
        safety_vest_violations: list[list[float]],
        warnings: dict[str, dict[str, int]],
    ) -> None:
        """
        Checks for safety violations among personnel.

        Arg:
            datas: The input data containing personnel information.
            warnings: A dictionary to store warning messages.
            polygons: A list to store the detected polygon areas.
        """
        # 1. Count violations
        count_no_hardhat = len(hardhat_violations)
        count_no_vest = len(safety_vest_violations)

        # 2. Write warnings
        if count_no_hardhat > 0:
            warnings['warning_no_hardhat'] = {'count': count_no_hardhat}

        if count_no_vest > 0:
            warnings['warning_no_safety_vest'] = {'count': count_no_vest}

    # Checks if personnel are dangerously close to machinery/vehicles
    def check_proximity_violations(
        self,
        persons: list[list[float]],
        machinery_vehicles: list[list[float]],
        warnings: dict[str, dict[str, int]],
    ) -> None:
        count_machinery = 0
        count_vehicle = 0

        for person in persons:
            for mv in machinery_vehicles:
                label = 'machinery' if mv[5] == 8 else 'vehicle'
                if Utils.is_dangerously_close(person[:4], mv[:4], label):
                    if label == 'machinery':
                        count_machinery += 1
                    else:
                        count_vehicle += 1

        if count_machinery > 0:
            warnings['warning_close_to_machinery'] = {'count': count_machinery}

        if count_vehicle > 0:
            warnings['warning_close_to_vehicle'] = {'count': count_vehicle}

    def check_machinery_near_utility_pole(
        self,
        datas: list[list[float]],
        warnings: dict[str, dict[str, int]],
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

            # 2. Check if machinery/vehicles meet both conditions
            for mv in machinery_vehicles:
                mx1, my1, mx2, my2, *_ = mv
                # Top of the machinery must be within [pole_top, 2/3 height]
                if not (py1 <= my1 <= two_thirds_y):
                    continue

                # Create the bottom line of the machinery
                bottom_line = LineString([(mx1, my2), (mx2, my2)])

                # Create the circle at the bottom of the utility pole
                pole_circle = Point(circle_center).buffer(circle_radius)
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
