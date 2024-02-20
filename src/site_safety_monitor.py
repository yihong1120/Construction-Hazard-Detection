from typing import List, Dict, Tuple
import time

# Dictionary to track the last movement time of each vehicle or machinery
last_movement_time = {}

def detect_danger(timestamp: int, ids: List[int], data: List[Tuple], last_movement_time: Dict[int, int]) -> List[str]:
    warnings = []  # Store all warning information
    current_time = timestamp  # Current timestamp

    # Update last movement time for each detected vehicle or machinery
    for i, detection in enumerate(data):
        id = ids[i]
        if detection[-1] in [0, 7]:  # Assuming 0 and 7 are the labels for vehicle and machinery
            if id not in last_movement_time or current_time - last_movement_time[id] > 1800:
                last_movement_time[id] = current_time  # Update last movement time

    # Classify detections into different categories
    persons = [d for d in data if d[-2] == 1]  # Assuming 1 is the label for Person
    hardhat_violations = [d for d in data if d[-2] == 2]  # Assuming 2 is the label for NO-Hardhat
    safety_vest_violations = [d for d in data if d[-2] == 3]  # Assuming 3 is the label for NO-Safety Vest
    machinery_vehicles = [d for d in data if d[-2] in [0, 7]]  # Assuming 0 and 7 are the labels for vehicle and machinery

    # Check for hardhat and safety vest violations
    for violation in hardhat_violations + safety_vest_violations:
        if not any(overlap_percentage(violation[:4], p[:4]) > 0.7 for p in persons):
            warnings.append(f"Warning: Someone is not wearing a {violation[-2]}. Location: {violation[:4]}")

    # Check for persons that are drivers
    drivers = []
    for person in persons:
        for mv in machinery_vehicles:
            if overlap_percentage(person[:4], mv[:4]) > 0.7 and is_driver(person[:4], mv[:4]):
                drivers.append(person)
                break

    # Check for persons dangerously close to machinery or vehicles
    for person in persons:
        if person not in drivers:
            for mv in machinery_vehicles:
                id = ids[data.index(mv)]
                if current_time - last_movement_time[id] <= 1800:  # Check if the vehicle/machinery has moved in the last 30 minutes
                    if is_dangerously_close(person[:4], mv[:4], mv[-2]):
                        warnings.append(f"Warning: There is a person dangerously close to the machinery or vehicle! Location: {person[:4]}")
                        break

    return warnings

def is_driver(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
    person_center_y = (bbox1[1] + bbox1[3]) / 2
    vehicle_bottom_y = bbox2[3]
    person_height = bbox1[3] - bbox1[1]

    return person_center_y > vehicle_bottom_y and abs(person_center_y - vehicle_bottom_y) >= person_height

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

    overlap_percentage = overlap_area / float(area1 + area2 - overlap_area)

    return overlap_percentage

def is_dangerously_close(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int], label2: str) -> bool:
    """
    Determine whether a person is dangerously close to machinery or vehicles, 
    considering both the horizontal distance, vertical distance, and the area ratio conditions.

    Args:
        bbox1 (Tuple[int, int, int, int]): The bounding box of the person.
        bbox2 (Tuple[int, int, int, int]): The bounding box of the machinery or vehicle.
        label2 (str): The label of the second bounding box (either 'machinery' or 'vehicle').

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

    # Danger distance horizontally and vertically
    danger_distance_horizontal = 5 * person_width
    danger_distance_vertical = 1.5 * person_height

    # Calculate horizontal and vertical distance from person to machinery/vehicle
    horizontal_distance = min(abs(bbox1[2] - bbox2[0]), abs(bbox1[0] - bbox2[2]))
    vertical_distance = min(abs(bbox1[3] - bbox2[1]), abs(bbox1[1] - bbox2[3]))

    # Determine if the person is within the danger distance of the machinery/vehicle
    is_close = horizontal_distance <= danger_distance_horizontal and vertical_distance <= danger_distance_vertical
    
    return is_close

# Main execution block
if __name__ == '__main__':
    # Example usage with timestamp, ids, and data
    timestamp = 1582123456
    ids = [3, 12, 18]
    data = [
        (706.87, 445.07, 976.32, 1073.6, 3, 0.91, 0),
        (0.45513, 471.77, 662.03, 1071.4, 12, 0.75853, 7),
        (1042.7, 638.5, 1077.5, 731.98, 18, 0.56060, 0)
    ]
    
    # Use the detection function to check for dangers
    warnings = detect_danger(timestamp, ids, data, last_movement_time)
    
    # Print out any warnings
    for warning in warnings:
        print(warning)
