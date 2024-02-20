from typing import List, Dict, Tuple

# Example data structure, this needs to be replaced with actual detection results
detections = [
    {"label": "NO-Hardhat", "bbox": [100, 100, 200, 200]},
    {"label": "Person", "bbox": [150, 150, 250, 250]},
    {"label": "machinery", "bbox": [300, 300, 400, 400]},
    # ...Other detection results
]

def detect_danger(detections: List[Dict]) -> List[str]:
    warnings = []  # Store all warning information

    # Classify detections into different categories
    persons = [d for d in detections if d["label"] == "Person"]
    hardhat_violations = [d for d in detections if d["label"] == "NO-Hardhat"]
    safety_vest_violations = [d for d in detections if d["label"] == "NO-Safety Vest"]
    machinery_vehicles = [d for d in detections if d["label"] in ["machinery", "vehicle"]]

    # Check for hardhat and safety vest violations
    for violation in hardhat_violations + safety_vest_violations:
        if not any(overlap_percentage(violation["bbox"], p["bbox"]) > 0.7 for p in persons):
            warnings.append(f"Warning: Someone is not wearing a {violation['label'][3:]}. Location: {violation['bbox']}")

    # Check for persons that are drivers
    drivers = []
    for person in persons:
        for mv in machinery_vehicles:
            if overlap_percentage(person["bbox"], mv["bbox"]) > 0.7 and is_driver(person["bbox"], mv["bbox"]):
                drivers.append(person)
                break

    # Check for persons dangerously close to machinery or vehicles
    for person in persons:
        if person not in drivers:
            for mv in machinery_vehicles:
                if is_dangerously_close(person["bbox"], mv["bbox"], mv["label"]):
                    warnings.append(f"Warning: There is a person dangerously close to the machinery or vehicle! Location: {person['bbox']}")
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
    # Use the detection function to check for dangers
    warnings = detect_danger(detections)
    
    # Print out any warnings
    for warning in warnings:
        print(warning)
