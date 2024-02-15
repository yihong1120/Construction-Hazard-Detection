from typing import List, Dict, Tuple

# Example data structure, this needs to be replaced with actual detection results
detections = [
    {"label": "NO-Hardhat", "bbox": [100, 100, 200, 200]},
    {"label": "Person", "bbox": [150, 150, 250, 250]},
    {"label": "machinery", "bbox": [300, 300, 400, 400]},
    {"label": "Driver Seat", "bbox": [100, 100, 200, 200]},
    # ...Other detection results
]

def detect_danger(detections: List[Dict]) -> List[str]:
    """
    Detects potential dangers based on the provided detections.

    This function works by classifying the detections into different categories (persons, hardhat violations, safety vest violations, machinery/vehicles, and driver seats). 
    It then checks for hardhat and safety vest violations by seeing if there is any person detection that overlaps significantly with the violation detection. 
    If there is, a warning is added to the list of warnings.
    It also checks for persons that are dangerously close to machinery or vehicles by seeing if there is any person detection that is dangerously close to a machinery/vehicle detection and not significantly overlapping with a driver seat detection. 
    If there is, a warning is added to the list of warnings.

    Args:
        detections (List[Dict]): A list of detections. Each detection is a dictionary with a 'label' and 'bbox'.

    Returns:
        List[str]: A list of warnings.
    """
    warnings = []  # Store all warning information

    # Classify detections into different categories
    persons = [d for d in detections if d["label"] == "Person"]  # All person detections
    hardhat_violations = [d for d in detections if d["label"] == "NO-Hardhat"]  # All hardhat violation detections
    safety_vest_violations = [d for d in detections if d["label"] == "NO-Safety Vest"]  # All safety vest violation detections
    machinery_vehicles = [d for d in detections if d["label"] in ["machinery", "vehicle"]]  # All machinery and vehicle detections
    driver_seats = [d for d in detections if d["label"] == "Driver Seat"]  # All driver seat detections

    # Check for hardhat and safety vest violations
    for violation in hardhat_violations + safety_vest_violations:
        # If there is no person detection that overlaps significantly with the violation detection, add a warning
        if not any(overlap_percentage(violation["bbox"], p["bbox"]) > 0.7 for p in persons):
            warnings.append(f"Warning: Someone is not wearing a {violation['label'][3:]}. Location: {violation['bbox']}")

    # Check for persons dangerously close to machinery or vehicles
    for person in persons:
        # If there is no driver seat detection that overlaps significantly with the person detection and there is a machinery/vehicle detection that the person detection is dangerously close to, add a warning
        if not any(overlap_percentage(person["bbox"], ds["bbox"]) > 0.7 for ds in driver_seats):
            if any(is_dangerously_close(person["bbox"], mv["bbox"]) for mv in machinery_vehicles):
                warnings.append(f"Warning: There is a person dangerously close to the machinery or vehicle! Location: {person['bbox']}")

    return warnings

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

def is_dangerously_close(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
    """
    Determine whether a person is dangerously close to machinery or vehicles.

    This checks if the person's bounding box is within a distance of five times its width
    from the bounding box of machinery or vehicles.

    Args:
        bbox1 (Tuple[int, int, int, int]): The bounding box of the person.
        bbox2 (Tuple[int, int, int, int]): The bounding box of the machinery or vehicle.

    Returns:
        bool: True if the person is dangerously close, False otherwise.
    """
    # Calculate the width of the person's bounding box
    person_width = bbox1[2] - bbox1[0]
    danger_distance = 5 * person_width

    # Check if the person is within the danger distance of the machinery/vehicle
    # This includes checking all sides: left, right, top, and bottom
    is_close = (
        abs((bbox1[0] + bbox1[2]) / 2 - (bbox2[0] + bbox2[2]) / 2) < danger_distance and
        abs((bbox1[1] + bbox1[3]) / 2 - (bbox2[1] + bbox2[3]) / 2) < danger_distance
    )
    return is_close

# Main execution block
if __name__ == '__main__':
    # Use the detection function to check for dangers
    warnings = detect_danger(detections)
    
    # Print out any warnings
    for warning in warnings:
        print(warning)