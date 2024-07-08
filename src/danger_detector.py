from __future__ import annotations
from typing import TypedDict, List, Set

class BoundingBox(TypedDict):
    x1: float
    y1: float
    x2: float
    y2: float

class DetectionData(TypedDict):
    bbox: BoundingBox
    confidence: float
    class_label: float

class DangerDetector:
    """
    A class to detect potential safety hazards based on the detection data.
    """

    def __init__(self):
        """
        Initialises the danger detector.
        """
        pass

    def detect_danger(self, datas: List[DetectionData]) -> Set[str]:
        """
        Detects potential safety violations in a construction site.

        This function checks for two types of safety violations:
        1. Workers not wearing hardhats or safety vests.
        2. Workers dangerously close to machinery or vehicles.

        Args:
            datas (List[DetectionData]): A list of detections which includes
                bounding box coordinates, confidence score, and class label.

        Returns:
            Set[str]: A set of warning messages for safety violations.
        """
        warnings = set()  # Initialise the list to store warning messages

        # Classify detected objects into different categories
        persons = [d for d in datas if d['class_label'] == 5.0]  # Persons
        hardhat_violations = [d for d in datas if d['class_label'] == 2.0]  # No hardhat
        safety_vest_violations = [d for d in datas if d['class_label'] == 4.0]  # No safety vest
        machinery_vehicles = [d for d in datas if d['class_label'] in [8.0, 9.0]]  # Machinery and vehicles

        # Filter out persons who are likely drivers
        if machinery_vehicles:
            non_drivers = [
                p for p in persons if not any(
                    self.is_driver(p['bbox'], mv['bbox']) for mv in machinery_vehicles
                )
            ]
            persons = non_drivers

        # Check for hardhat and safety vest violations
        for violation in hardhat_violations + safety_vest_violations:
            label = 'NO-Hardhat' if violation['class_label'] == 2.0 else 'NO-Safety Vest'
            if not any(
                self.overlap_percentage(violation['bbox'], p['bbox']) > 0.5 for p in persons
            ):
                warning_msg = (
                    '警告: 有人無配戴安全帽!' if label == 'NO-Hardhat'
                    else '警告: 有人無穿著安全背心!'
                )
                warnings.add(warning_msg)

        # Check if anyone is dangerously close to machinery or vehicles
        for person in persons:
            for mv in machinery_vehicles:
                label = '機具' if mv['class_label'] == 8.0 else '車輛'
                if self.is_dangerously_close(person['bbox'], mv['bbox'], label):
                    warnings.add(f"警告: 有人過於靠近{label}!")
                    break

        return warnings

    @staticmethod
    def is_driver(person_bbox: BoundingBox, vehicle_bbox: BoundingBox) -> bool:
        """
        Check if a person is a driver based on position near a vehicle.

        Args:
            person_bbox (BoundingBox): Bounding box of person.
            vehicle_bbox (BoundingBox): Bounding box of vehicle.

        Returns:
            bool: True if the person is likely the driver, False otherwise.
        """
        # Extract coordinates and dimensions of person and vehicle boxes
        person_bottom_y = person_bbox['y2']
        person_top_y = person_bbox['y1']
        person_left_x = person_bbox['x1']
        person_right_x = person_bbox['x2']
        person_width = person_bbox['x2'] - person_bbox['x1']
        person_height = person_bbox['y2'] - person_bbox['y1']

        vehicle_top_y = vehicle_bbox['y1']
        vehicle_bottom_y = vehicle_bbox['y2']
        vehicle_left_x = vehicle_bbox['x1']
        vehicle_right_x = vehicle_bbox['x2']
        vehicle_height = vehicle_bbox['y2'] - vehicle_bbox['y1']

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
    def overlap_percentage(bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """
        Calculate the overlap percentage between two bounding boxes.

        Args:
            bbox1 (BoundingBox): The first bounding box.
            bbox2 (BoundingBox): The second bounding box.

        Returns:
            float: The overlap percentage.
        """
        # Calculate the coordinates of the intersection rectangle
        x1 = max(bbox1['x1'], bbox2['x1'])
        y1 = max(bbox1['y1'], bbox2['y1'])
        x2 = min(bbox1['x2'], bbox2['x2'])
        y2 = min(bbox1['y2'], bbox2['y2'])

        # Calculate the area of the intersection rectangle
        overlap_area = max(0, x2 - x1) * max(0, y2 - y1)

        # Calculate the area of both bounding boxes
        area1 = (bbox1['x2'] - bbox1['x1']) * (bbox1['y2'] - bbox1['y1'])
        area2 = (bbox2['x2'] - bbox2['x1']) * (bbox2['y2'] - bbox2['y1'])

        # Calculate the overlap percentage
        return overlap_area / float(area1 + area2 - overlap_area)

    @staticmethod
    def is_dangerously_close(person_bbox: BoundingBox, vehicle_bbox: BoundingBox, label: str) -> bool:
        """
        Determine if a person is dangerously close to machinery or vehicles.

        Args:
            person_bbox (BoundingBox): Bounding box of person.
            vehicle_bbox (BoundingBox): Machine/vehicle box.
            label (str): Type of the second object ('machinery' or 'vehicle').

        Returns:
            bool: True if the person is dangerously close, False otherwise.
        """
        # Calculate dimensions of the person bounding box
        person_width = person_bbox['x2'] - person_bbox['x1']
        person_height = person_bbox['y2'] - person_bbox['y1']
        person_area = person_width * person_height

        # Calculate the area of the vehicle bounding box
        vehicle_area = (vehicle_bbox['x2'] - vehicle_bbox['x1']) * (vehicle_bbox['y2'] - vehicle_bbox['y1'])
        acceptable_ratio = 0.1 if label == '車輛' else 0.05

        # Check if person area ratio is acceptable compared to vehicle area
        if person_area / vehicle_area > acceptable_ratio:
            return False

        # Define danger distances
        danger_distance_horizontal = 5 * person_width
        danger_distance_vertical = 1.5 * person_height

        # Calculate min horizontal/vertical distance between person and vehicle
        horizontal_distance = min(
            abs(person_bbox['x2'] - vehicle_bbox['x1']),
            abs(person_bbox['x1'] - vehicle_bbox['x2']),
        )
        vertical_distance = min(
            abs(person_bbox['y2'] - vehicle_bbox['y1']),
            abs(person_bbox['y1'] - vehicle_bbox['y2']),
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
        {'bbox': {'x1': 706.87, 'y1': 445.07, 'x2': 976.32, 'y2': 1073.6}, 'confidence': 0.91, 'class_label': 5.0},
        {'bbox': {'x1': 0.45513, 'y1': 471.77, 'x2': 662.03, 'y2': 1071.4}, 'confidence': 0.75853, 'class_label': 12.0},
        {'bbox': {'x1': 1042.7, 'y1': 638.5, 'x2': 1077.5, 'y2': 731.98}, 'confidence': 0.56060, 'class_label': 18.0},
    ]
    warnings = detector.detect_danger(data)
    for warning in warnings:
        print(warning)
