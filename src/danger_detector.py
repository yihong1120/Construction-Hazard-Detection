from __future__ import annotations

from typing import TypedDict

from shapely.geometry import MultiPoint
from shapely.geometry import Point
from shapely.geometry import Polygon


class InputData(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_label: int


class ResultData(TypedDict):
    warnings: set[str]


class DangerDetector:
    """
    A class to detect potential safety hazards based on the detection data.
    """

    def __init__(self):
        """
        Initialises the danger detector.
        """
        pass

    def calculate_people_in_controlled_area(
        self,
        datas: list[list[float]],
        polygon: Polygon | None,
    ) -> int:
        """
        Calculates the number of people within the safety cone area.

        Args:
            datas (List[List[float]]): The detection data.
            polygon (Optional[Polygon]): Polygon formed by the safety cones.

        Returns:
            int: Number of people detected within the controlled area.
        """
        # Check if there are any detections
        if not datas:
            return 0

        # Check if there is a valid polygon
        if not polygon or not isinstance(polygon, Polygon):
            return 0

        # Count the number of people within the controlled area
        people_count = 0
        for data in datas:
            if data[5] == 5:  # Check if it's a person
                x_center = (data[0] + data[2]) / 2
                y_center = (data[1] + data[3]) / 2
                if polygon.contains(Point(x_center, y_center)):
                    people_count += 1

        return people_count

    def detect_danger(
        self,
        datas: list[list[float]],
        polygon: Polygon | None,
    ) -> set[str]:
        """
        Detects potential safety violations in a construction site.

        This function checks for two types of safety violations:
        1. Workers entering the controlled area.
        2. Workers not wearing hardhats or safety vests.
        3. Workers dangerously close to machinery or vehicles.

        Args:
            datas (List[List[float]]): A list of detections which includes
                bounding box coordinates, confidence score, and class label.
            polygon (Optional[Polygon]): Polygon formed by the safety cones.

        Returns:
            Set[str]: A set of warning messages for safety violations.
        """
        warnings = set()  # Initialise the list to store warning messages

        # Check if people are entering the controlled area
        people_count = self.calculate_people_in_controlled_area(datas, polygon)
        if people_count > 0:
            warnings.add(f'警告: 有{people_count}個人進入受控區域!')

        # Classify detected objects into different categories
        persons = [d for d in datas if d[5] == 5]  # Persons
        hardhat_violations = [d for d in datas if d[5] == 2]  # No hardhat
        safety_vest_violations = [
            d for d in datas if d[5] == 4
        ]  # No safety vest
        machinery_vehicles = [
            d for d in datas if d[5] in [8, 9]
        ]  # Machinery and vehicles

        # Filter out persons who are likely drivers
        if machinery_vehicles:
            non_drivers = [
                p for p in persons if not any(
                    self.is_driver(p[:4], mv[:4]) for mv in machinery_vehicles
                )
            ]
            persons = non_drivers

        # Check for hardhat and safety vest violations
        for violation in hardhat_violations + safety_vest_violations:
            label = 'NO-Hardhat' if violation[5] == 2 else 'NO-Safety Vest'
            if not any(
                self.overlap_percentage(
                    violation[:4], p[:4],
                ) > 0.5 for p in persons
            ):
                warning_msg = (
                    '警告: 有人無配戴安全帽!' if label == 'NO-Hardhat'
                    else '警告: 有人無穿著安全背心!'
                )
                warnings.add(warning_msg)

        # Check if anyone is dangerously close to machinery or vehicles
        for person in persons:
            for mv in machinery_vehicles:
                label = '機具' if mv[5] == 8 else '車輛'
                if self.is_dangerously_close(person[:4], mv[:4], label):
                    warnings.add(f"警告: 有人過於靠近{label}!")
                    break

        return warnings

    @staticmethod
    def is_driver(
        person_bbox: list[float],
        vehicle_bbox: list[float],
    ) -> bool:
        """
        Check if a person is a driver based on position near a vehicle.

        Args:
            person_bbox (list[float]): Bounding box of person.
            vehicle_bbox (list[float]): Bounding box of vehicle.

        Returns:
            bool: True if the person is likely the driver, False otherwise.
        """
        # Extract coordinates and dimensions of person and vehicle boxes
        person_bottom_y = person_bbox[3]
        person_top_y = person_bbox[1]
        person_left_x = person_bbox[0]
        person_right_x = person_bbox[2]
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]

        vehicle_top_y = vehicle_bbox[1]
        vehicle_bottom_y = vehicle_bbox[3]
        vehicle_left_x = vehicle_bbox[0]
        vehicle_right_x = vehicle_bbox[2]
        vehicle_height = vehicle_bbox[3] - vehicle_bbox[1]

        # 1. Check vertical bottom position: person's bottom should be above
        #    the vehicle's bottom by at least half the person's height
        if not (
            person_bottom_y < vehicle_bottom_y
            and vehicle_bottom_y - person_bottom_y >= person_height / 2
        ):
            return False

        # 2. Check horizontal position: person's edges should not extend
        #    beyond half the width of the person from the vehicle's edges
        if not (
            person_left_x >= vehicle_left_x - person_width / 2
            and person_right_x <= vehicle_right_x + person_width / 2
        ):
            return False

        # 3. The person's top must be below the vehicle's top
        if not (person_top_y > vehicle_top_y):
            return False

        # 4. Person's height is less than or equal to half the vehicle's height
        if not (person_height <= vehicle_height / 2):
            return False

        return True

    @staticmethod
    def overlap_percentage(
        bbox1: list[float],
        bbox2: list[float],
    ) -> float:
        """
        Calculate the overlap percentage between two bounding boxes.

        Args:
            bbox1 (list[float]): The first bounding box.
            bbox2 (list[float]): The second bounding box.

        Returns:
            float: The overlap percentage.
        """
        # Calculate the coordinates of the intersection rectangle
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        # Calculate the area of the intersection rectangle
        overlap_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the area of both bounding boxes
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        # Calculate the overlap percentage
        return overlap_area / float(area1 + area2 - overlap_area)

    @staticmethod
    def is_dangerously_close(
        person_bbox: list[float],
        vehicle_bbox: list[float],
        label: str,
    ) -> bool:
        """
        Determine if a person is dangerously close to machinery or vehicles.

        Args:
            person_bbox (list[float]): Bounding box of person.
            vehicle_bbox (list[float]): Machine/vehicle box.
            label (str): Type of the second object ('machinery' or 'vehicle').

        Returns:
            bool: True if the person is dangerously close, False otherwise.
        """
        # Calculate dimensions of the person bounding box
        person_width = person_bbox[2] - person_bbox[0]
        person_height = person_bbox[3] - person_bbox[1]
        person_area = person_width * person_height

        # Calculate the area of the vehicle bounding box
        vehicle_area = (vehicle_bbox[2] - vehicle_bbox[0]) * (
            vehicle_bbox[3] - vehicle_bbox[1]
        )
        acceptable_ratio = 0.1 if label == '車輛' else 0.05

        # Check if person area ratio is acceptable compared to vehicle area
        if person_area / vehicle_area > acceptable_ratio:
            return False

        # Define danger distances
        danger_distance_horizontal = 5 * person_width
        danger_distance_vertical = 1.5 * person_height

        # Calculate min horizontal/vertical distance between person and vehicle
        horizontal_distance = min(
            abs(
                person_bbox[2] - vehicle_bbox[0],
            ),
            abs(person_bbox[0] - vehicle_bbox[2]),
        )
        vertical_distance = min(
            abs(
                person_bbox[3] - vehicle_bbox[1],
            ),
            abs(person_bbox[1] - vehicle_bbox[3]),
        )

        # Determine if the person is dangerously close
        return (
            horizontal_distance <= danger_distance_horizontal
            and vertical_distance <= danger_distance_vertical
        )


# Example usage
if __name__ == '__main__':
    detector = DangerDetector()
    data = [
        [706.87, 445.07, 976.32, 1073.6, 3, 0.91],
        [0.45513, 471.77, 662.03, 1071.4, 12, 0.75853],
        [1042.7, 638.5, 1077.5, 731.98, 18, 0.56060],
    ]

    polygon = MultiPoint([(0, 0), (0, 1000), (1000, 0)])

    warnings = detector.detect_danger(data, polygon)
    for warning in warnings:
        print(warning)
