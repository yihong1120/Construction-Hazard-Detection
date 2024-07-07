---
marp: true
theme: default
class:
  - lead
  - invert
paginate: true
---

# YOLO to COCO Conversion and Model Evaluation

---

## Introduction

- **Objective**: Convert YOLO annotations to COCO format and evaluate object detection models.
- **Scripts**:
  - `convert_yolo_to_coco.py`
  - `evaluate_sahi_yolov8.py`
  - `evaluate_yolov8.py`
- **Usage**: Useful for preparing datasets and assessing model performance.

---

## convert_yolo_to_coco.py Overview

- **Purpose**: Convert YOLO format annotations to COCO format for compatibility with different tools and frameworks.
- **Features**:
  - Parses YOLO annotations and corresponding images.
  - Generates a COCO-format JSON file.

---

## Running convert_yolo_to_coco.py

1. Provide the directories for YOLO labels and images.
2. Specify the output file for COCO formatted annotations.
3. Command:
   ```bash
   python convert_yolo_to_coco.py --labels_dir <labels_directory> --images_dir <images_directory> --output <output_json_file>
   ```

---

## evaluate_sahi_yolov8.py Overview

- **Purpose**: Evaluate object detection models on COCO metrics using the SAHI framework.
- **Features**:
  - Sliced prediction for handling large images.
  - Computes COCO metrics for model evaluation.

---

## Using evaluate_sahi_yolov8.py

1. Specify model path, COCO JSON, and image directory.
2. Set optional parameters for slicing and confidence threshold.
3. Command:
   ```bash
   python evaluate_sahi_yolov8.py --model_path <model_file> --coco_json <coco_json_file> --image_dir <image_directory>
   ```

---

## evaluate_yolov8.py Overview

- **Purpose**: Evaluate YOLOv8 models using the Ultralytics framework.
- **Features**:
  - Direct integration with YOLOv8.
  - Provides standard evaluation metrics.

---

## Running evaluate_yolov8.py

1. Input the model and dataset configuration paths.
2. Execute the script to get evaluation metrics.
3. Command:
   ```bash
   python evaluate_yolov8.py --model_path <model_file> --data_path <data_yaml_file>
   ```

---

## Conclusion

- **Data Preparation**: Convert annotations to ensure compatibility with various frameworks.
- **Model Evaluation**: Assess model performance using standard COCO metrics.
- **Optimisation**: Utilise these tools for iterative improvement of object detection models.

Thank you for attending this presentation.
