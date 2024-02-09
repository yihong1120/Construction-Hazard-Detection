from typing import List, Dict, Tuple

# Example data structure, this needs to be replaced with actual detection results
detections = [
    {"label": "NO-Hardhat", "bbox": [100, 100, 200, 200]},
    {"label": "Person", "bbox": [150, 150, 250, 250]},
    {"label": "machinery", "bbox": [300, 300, 400, 400]},
    # ...Other detection results
]

def detect_danger(detections: List[Dict]) -> List[str]:
    """
    Detect potential safety hazards based on object detection results.

    This function processes a list of detection results and generates warnings
    for any detected safety hazards, such as persons not wearing hardhats or
    safety vests, or persons being too close to machinery.

    Args:
        detections (List[Dict]): A list of detection results, where each detection
                                 is a dictionary with a 'label' and 'bbox' (bounding box).

    Returns:
        List[str]: A list of warning messages indicating the detected safety hazards.

    Example:
        >>> detections = [{"label": "NO-Hardhat", "bbox": [100, 100, 200, 200]}]
        >>> warnings = detect_danger(detections)
        >>> for warning in warnings:
        ...     print(warning)
        Warning: Someone is not wearing a helmet! Location: [100, 100, 200, 200]
    """
    warnings = []  # Store all warning information

    # Traverse all detected objects
    for detection in detections:
        label = detection["label"]
        bbox = detection["bbox"]

        # Helmet Rule: Check if anyone is not wearing a hardhat
        if label == "NO-Hardhat":
            warnings.append(f"Warning: Someone is not wearing a helmet! Location: {bbox}")

        # Safety Vest Rule: Check if anyone is not wearing a safety vest
        if label == "NO-Safety Vest":
            warnings.append(f"Warning: Someone is not wearing a safety vest! Location: {bbox}")

        # Machinery and Vehicle Rule: Check if persons are near machinery or vehicles
        if label == "Person":
            for d in detections:
                if d["label"] in ["machinery", "vehicle"]:
                    if is_dangerously_close(bbox, d["bbox"]):
                        warnings.append(f"Warning: There is a person dangerously close to the machinery or vehicle! Location: {bbox}")
                        break  # Once a person is found close to one machinery/vehicle, no need to check for others

    return warnings

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