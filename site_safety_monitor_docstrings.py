from typing import Dict, List, Tuple

from site_safety_monitor import detections


def detect_danger(detections: List[Dict]) -> List[str]:
    """
    Determine whether a person is in a dangerous situation based on the given detections.

    Args:
        detections (List[Dict]): List of detection results.

    Returns:
        List[str]: List of warning messages.

    """
    warnings = [] # Store all warning information

    # Traverse all detected targets
    for detection in detections:
        label = detection["label"]
        bbox = detection["bbox"]

        # Helmet Rules: Check if anyone is not wearing a helmet
        if label == "NO-Hardhat":
            warnings.append(f"Warning: Someone is not wearing a helmet! Location: {bbox}")

        # Safety Vest Rule: Check if anyone is not wearing a safety vest
        if label == "NO-Safety Vest":
            warnings.append(f"Warning: Someone is not wearing a safety vest! Location: {bbox}")

        # MACHINERY AND VEHICLE RULES: Check if persons are near machinery or vehicles
        if label == "Person":
            if any(d["label"] in ["machinery", "vehicle"] and is_too_close(d["bbox"], bbox) for d in detections):
                warnings.append(f"Warning: There is a person approaching the machinery or vehicle! Location: {bbox}")

        # Other rules can be added as needed
        #...

    return warnings

def is_too_close(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
    """
    Determine whether a person is too close to a machine or vehicle based on their bounding box coordinates.

    Args:
        bbox1 (Tuple[int, int, int, int]): Bounding box coordinates of the first object.
        bbox2 (Tuple[int, int, int, int]): Bounding box coordinates of the second object.

    Returns:
        bool: True if the objects are too close, False otherwise.

    """
    # Calculate the center point of the two bounding boxes
    center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    
    # Calculate the distance between center points
    distance = ((center1[0] - center2[0]) ** 2 + (center1[1] - center2[1]) ** 2) ** 0.5
    
    # Define a threshold. If the distance is less than this threshold, it is considered too close.
    threshold = 50 # This threshold needs to be set according to the actual situation
    return distance < threshold

# Use detection function
warnings = detect_danger(detections)
for warning in warnings:
    print(warning)
