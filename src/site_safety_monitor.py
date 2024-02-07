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
            if any(d["label"] in ["machinery", "vehicle"] and is_too_close(d["bbox"], bbox) for d in detections):
                warnings.append(f"Warning: There is a person approaching the machinery or vehicle! Location: {bbox}")

        # Other rules can be added as needed
        # ...

    return warnings

def is_too_close(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> bool:
    """
    Determine whether two bounding boxes are too close to each other.

    This function calculates the Euclidean distance between the centres of two bounding boxes
    and compares it to a predefined threshold to determine if they are too close.

    Args:
        bbox1 (Tuple[int, int, int, int]): The bounding box of the first object.
        bbox2 (Tuple[int, int, int, int]): The bounding box of the second object.

    Returns:
        bool: True if the objects are too close, False otherwise.

    Example:
        >>> bbox1 = (100, 100, 200, 200)
        >>> bbox2 = (150, 150, 250, 250)
        >>> is_too_close(bbox1, bbox2)
        True
    """
    # Calculate the centre points of the two bounding boxes
    centre1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
    centre2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
    
    # Calculate the distance between the centre points
    distance = ((centre1[0] - centre2[0]) ** 2 + (centre1[1] - centre2[1]) ** 2) ** 0.5
    
    # Define a threshold for being 'too close'
    threshold = 50  # This threshold needs to be set according to the actual situation
    return distance < threshold

# Main execution block
if __name__ == '__main__':
    # Use the detection function to check for dangers
    warnings = detect_danger(detections)
    
    # Print out any warnings
    for warning in warnings:
        print(warning)