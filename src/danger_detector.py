from typing import List, Dict, Tuple
import time

class DangerDetector:
    """
    A class to detect potential safety hazards based on the detection data and timestamps.
    """

    def __init__(self):
        """
        Initialises the danger detector.
        """
        # Dictionary to track the last movement time of each vehicle or machinery
        self.last_movement_time: Dict[float, int] = {}  # Key as float since IDs are floats

    def detect_danger(self, timestamp: int, datas: List[List[float]]) -> List[str]:
        """
        Detects potential safety violations in a construction site.

        This function checks for two types of safety violations:
        1. Workers not wearing hardhats or safety vests.
        2. Workers dangerously close to machinery or vehicles.

        Args:
            timestamp (int): The current timestamp.
            datas (List[List[float]]): A list of detections. Each detection is a list that includes
                the bounding box coordinates, confidence score, and class label.

        Returns:
            List[str]: A list of warning messages for detected safety violations.
        """
        warnings = []  # Store all warning messages

        # Classify detected objects
        persons = [d for d in datas if d[5] == 5.0]  # Persons
        hardhat_violations = [d for d in datas if d[5] == 2.0]  # No hardhat
        safety_vest_violations = [d for d in datas if d[5] == 4.0]  # No safety vest
        machinery_vehicles = [d for d in datas if d[5] in [8.0, 9.0]]  # Machinery and vehicles

        # Check for hardhat and safety vest violations
        for violation in hardhat_violations + safety_vest_violations:
            # Determine the type of violation based on the class label
            label = 'NO-Hardhat' if violation[5] == 2.0 else 'NO-Safety Vest'
            # Check if there is no significant overlap with any person detected
            if not any(self.overlap_percentage(violation[:4], p[:4]) > 0.5 for p in persons):
                if label == 'NO-Hardhat':
                    warnings.append(f"Warning: Someone is not wearing a Hardhat!")
                else:  # label == 'NO-Safety Vest'
                    warnings.append(f"Warning: Someone is not wearing a Safety Vest!")

        # Check if anyone is dangerously close to machinery or vehicles
        for person in persons:
            for mv in machinery_vehicles:
                # Determine the label based on the class_label
                label = 'machinery' if mv[5] == 8.0 else 'vehicle'  # mv[5] contains the class label
                
                if self.is_dangerously_close(person[:4], mv[:4], label):  # Pass label as the third argument
                    warnings.append(f"Warning: There is a person dangerously close to the {label}!")
                    break  # Stop checking if one warning is already added for this person

        return warnings

    @staticmethod
    def is_driver(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
        """
        Check if a detected person is likely to be a driver based on their position relative to a vehicle.

        Args:
            bbox1 (Tuple[int, int, int, int]): Bounding box of the person.
            bbox2 (Tuple[int, int, int, int]): Bounding box of the vehicle.

        Returns:
            bool: True if the person is likely to be the driver, False otherwise.
        """
        person_centre_y = (bbox1[1] + bbox1[3]) / 2
        vehicle_bottom_y = bbox2[3]
        person_height = bbox1[3] - bbox1[1]

        return person_centre_y > vehicle_bottom_y and abs(person_centre_y - vehicle_bottom_y) >= person_height

    @staticmethod
    def overlap_percentage(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate the overlap percentage between two bounding boxes.

        Args:
            bbox1 (Tuple[int, int, int, int]): The first bounding box.
            bbox2 (Tuple[int, int, int, int]): The second bounding box.

        Returns:
            float: The overlap percentage.
        """
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        overlap_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        area1 = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
        area2 = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)

        return overlap_area / float(area1 + area2 - overlap_area)

    @staticmethod
    def is_dangerously_close(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int], label2: str) -> bool:
        """
        Determine whether a person is dangerously close to machinery or vehicles.

        Args:
            bbox1 (Tuple[int, int, int, int]): Bounding box of the person.
            bbox2 (Tuple[int, int, int, int]): Bounding box of the machinery or vehicle.
            label2 (str): Type of the second object ('machinery' or 'vehicle').

        Returns:
            bool: True if the person is dangerously close, False otherwise.
        """
        person_width = bbox1[2] - bbox1[0]
        person_height = bbox1[3] - bbox1[1]
        person_area = (person_width + 1) * (person_height + 1)

        machinery_vehicle_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
        acceptable_ratio = 1/10 if label2 == 'vehicle' else 1/20

        if person_area / machinery_vehicle_area > acceptable_ratio:
            return False

        danger_distance_horizontal = 5 * person_width
        danger_distance_vertical = 1.5 * person_height

        horizontal_distance = min(abs(bbox1[2] - bbox2[0]), abs(bbox1[0] - bbox2[2]))
        vertical_distance = min(abs(bbox1[3] - bbox2[1]), abs(bbox1[1] - bbox2[3]))

        return horizontal_distance <= danger_distance_horizontal and vertical_distance <= danger_distance_vertical

# Example usage
if __name__ == "__main__":
    detector = DangerDetector()
    timestamp = 1582123456
    data = [
        [706.87, 445.07, 976.32, 1073.6, 3, 0.91, 0],
        [0.45513, 471.77, 662.03, 1071.4, 12, 0.75853, 7],
        [1042.7, 638.5, 1077.5, 731.98, 18, 0.56060, 0]
    ]
    warnings = detector.detect_danger(timestamp, data)
    for warning in warnings:
        print(warning)