from __future__ import annotations

from shapely.geometry import Polygon
from sklearn.cluster import HDBSCAN

from .utils import Utils


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
                - 'detect_no_safety_vest_or_helmet': Detect if workers are not
                  wearing hardhats or safety vests.
                - 'detect_near_machinery_or_vehicle': Detect if workers are
                  dangerously close to machinery or vehicles.
                - 'detect_in_restricted_area': Detect if workers are entering
                  restricted areas.

        Raises:
            ValueError: If the detection_items is not a dictionary or if any
                of the keys are not strings or values are not booleans.

        Examples:
            >>> detector = DangerDetector({
            ...     'detect_no_safety_vest_or_helmet': True,
            ...     'detect_near_machinery_or_vehicle': True,
            ...     'detect_in_restricted_area': True,
            ... })
        """
        # Initialise the HDBSCAN clusterer
        self.clusterer = HDBSCAN(min_samples=3, min_cluster_size=2)

        # Define required keys
        required_keys = {
            'detect_no_safety_vest_or_helmet',
            'detect_near_machinery_or_vehicle',
            'detect_in_restricted_area',
        }

        # Validate detection_items type and content
        if isinstance(detection_items, dict) and all(
            isinstance(k, str) and isinstance(v, bool)
            for k, v in detection_items.items()
        ) and required_keys.issubset(detection_items.keys()):
            self.detection_items = detection_items
        else:
            self.detection_items = {}

    def detect_danger(
        self,
        datas: list[list[float]],
    ) -> tuple[list[str], list[Polygon]]:
        """
        Detects potential safety violations in a construction site.

        This function checks for two types of safety violations:
        1. Workers entering the controlled area.
        2. Workers not wearing hardhats or safety vests.
        3. Workers dangerously close to machinery or vehicles.

        Args:
            datas (List[List[float]]): A list of detections which includes
                bounding box coordinates, confidence score, and class label.

        Returns:
            Tuple[Set[str], List[Polygon]]: Warnings and polygons list.
        """
        # Initialise the list to store warning messages
        warnings: set[str] = set()

        # Normalise data
        datas = Utils.normalise_data(datas)

        polygons: list[Polygon] = []

        ############################################################
        # Check if detection is enabled or no specific detection items are set
        ############################################################
        if (
            not self.detection_items or
            self.detection_items.get('detect_in_restricted_area', False)
        ):
            self.check_restricted_area(datas, warnings, polygons)

        ############################################################
        # Classify detected objects into different categories
        ############################################################

        # Persons
        persons = [d for d in datas if d[5] == 5]

        # No hardhat
        hardhat_violations = [d for d in datas if d[5] == 2]

        # No safety vest
        safety_vest_violations = [
            d for d in datas if d[5] == 4
        ]

        # Machinery and vehicles
        machinery_vehicles = [
            d for d in datas if d[5]
            in [8, 9]
        ]

        # Filter out persons who are likely drivers
        if machinery_vehicles:
            non_drivers = [
                p for p in persons if not any(
                    Utils.is_driver(p[:4], mv[:4]) for mv in machinery_vehicles
                )
            ]
            persons = non_drivers

        ############################################################
        # Check if people are not wearing hardhats or safety vests
        ############################################################
        if (
            not self.detection_items or
                self.detection_items.get(
                    'detect_no_safety_vest_or_helmet', False,
                )
        ):
            self.check_safety_violations(
                persons, hardhat_violations,
                safety_vest_violations, warnings,
            )

        ############################################################
        # Check if people are dangerously close to machinery or vehicles
        ############################################################
        if (
            not self.detection_items or
            self.detection_items.get('detect_near_machinery_or_vehicle', False)
        ):
            self.check_proximity_violations(
                persons, machinery_vehicles, warnings,
            )

        return list(warnings), polygons

    def check_restricted_area(
        self,
        datas: list[list[float]],
        warnings: set[str],
        polygons: list[Polygon],
    ) -> None:
        """
        Check if people are entering the controlled area.

        Args:
            datas (List[List[float]]): A list of detections.
            warnings (set[str]): A set to store warning messages.
            polygons (list[Polygon]): A list to store detected polygons.
        """
        polygons.extend(Utils.detect_polygon_from_cones(datas, self.clusterer))
        people_count = Utils.calculate_people_in_controlled_area(
            polygons, datas,
        )
        if people_count > 0:
            warnings.add(
                f"Warning: {people_count} people have "
                'entered the controlled area!',
            )

    def check_safety_violations(
        self, persons: list[list[float]],
        hardhat_violations: list[list[float]],
        safety_vest_violations: list[list[float]],
        warnings: set[str],
    ) -> None:
        """
        Check for hardhat and safety vest violations.

        Args:
            persons (List[List[float]]): A list of person detections.
            hardhat_violations (List[List[float]]):
                A list of hardhat violations.
            safety_vest_violations (List[List[float]]):
                A list of safety vest violations.
            warnings (set[str]): A set to store warning messages.
        """
        for violation in hardhat_violations + safety_vest_violations:
            label = 'NO-Hardhat' if violation[5] == 2 else 'NO-Safety Vest'
            if not any(
                Utils.overlap_percentage(violation[:4], p[:4]) > 0.5
                for p in persons
            ):
                warning_msg = (
                    'Warning: Someone is not wearing a hardhat!'
                    if label == 'NO-Hardhat'
                    else 'Warning: Someone is not wearing a safety vest!'
                )
                warnings.add(warning_msg)

    def check_proximity_violations(
        self, persons: list[list[float]],
        machinery_vehicles: list[list[float]],
        warnings: set[str],
    ) -> None:
        """
        Check if anyone is dangerously close to machinery or vehicles.

        Args:
            persons (List[List[float]]): A list of person detections.
            machinery_vehicles (List[List[float]]): A list of machinery
                and vehicle detections.
            warnings (set[str]): A set to store warning messages.
        """
        for person in persons:
            for mv in machinery_vehicles:
                label = 'machinery' if mv[5] == 8 else 'vehicle'
                if Utils.is_dangerously_close(person[:4], mv[:4], label):
                    warning_msg = (
                        f"Warning: Someone is too close to {label}!"
                    )
                    warnings.add(warning_msg)
                    break


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
    ]

    warnings, polygons = detector.detect_danger(data)
    print(f"Warnings: {warnings}")
    print(f"Polygons: {polygons}")


if __name__ == '__main__':
    main()
