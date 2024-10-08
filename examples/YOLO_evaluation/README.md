🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Evaluation Example

This repository provides examples and scripts for evaluating YOLO object detection models, particularly focusing on assessing the model's accuracy and performance on specific datasets.

## Usage

### Converting YOLO to COCO Format

To convert annotations from YOLO format to COCO format, use the `convert_yolo_to_coco.py` script. Specify the directories for YOLO labels and images, and the output path for the COCO formatted JSON:

```bash
python convert_yolo_to_coco.py --labels_dir dataset/valid/labels --images_dir dataset/valid/images --output dataset/coco_annotations.json
```

### Evaluating Models with SAHI

To evaluate a YOLO model using the SAHI library, run the `evaluate_sahi_yolo.py` script. Provide the paths to the model, COCO JSON, and image directory:

```bash
python evaluate_sahi_yolo.py --model_path "../../models/best_yolov8n.pt" --coco_json "dataset/coco_annotations.json" --image_dir "dataset/valid/images"
```

This script will output evaluation metrics such as Average Precision and Recall across different IoU thresholds.

### Evaluating Models with Ultralytics YOLO

For evaluation using the Ultralytics framework, execute the `evaluate_yolo.py` script. Again, specify the model and data configuration file paths:

```bash
python evaluate_yolo.py --model_path "../../models/best_yolov8n.pt" --data_path "dataset/data.yaml"
```

### Features

- **YOLO to COCO Conversion**: Convert your dataset annotations from YOLO format to COCO format for compatibility with various evaluation tools.
- **Model Evaluation**: Utilise the SAHI library or Ultralytics framework to evaluate your YOLO models.
- **Performance Metrics**: Gain insights into model performance with metrics like mAP, Precision, and Recall.
- **Customisable Evaluation**: Scripts allow for flexible evaluation setups tailored to your specific dataset and model.

## Configuration

Adjust the evaluation parameters, such as the confidence threshold and IoU metrics, to align with your specific dataset requirements. Modify the paths to the model, dataset, and annotation files as needed to fit your project's directory structure.
