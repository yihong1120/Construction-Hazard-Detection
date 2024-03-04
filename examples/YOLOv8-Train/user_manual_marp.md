---
marp: true
theme: default
class:
  - lead
  - invert
paginate: true
---

# YOLO Model Handling and Training

---

## Introduction

- **Objective**: Understand the workflow of training, validating, predicting, and exporting YOLO models.
- **Scripts**:
  - `train.py`: Handles YOLO model operations.
  - `train_yolo_models.sh`: Batch processes for model training.

---

## train.py Overview

- **Purpose**: Facilitate YOLO model operations including training, validation, prediction, and exporting.
- **Capabilities**:
  - Load and train YOLO models.
  - Validate models on a dataset.
  - Predict using trained models.
  - Export models to ONNX format.
  - Implement SAHI for sliced predictions.

---

## Using train.py

1. Set parameters for model handling:
   - Model path, data configuration, epochs, etc.
2. Execute the script for desired operations:
   ```bash
   python train.py --model_name <model_file> --data_config <data_config_file> --epochs <number_of_epochs>
   ```
3. Review the output for training metrics, predictions, and export paths.

---

## train_yolo_models.sh Overview

- **Purpose**: Automate the training process for multiple YOLO models.
- **Features**:
  - Checks and downloads base models.
  - Utilises pre-trained models if available.
  - Trains models sequentially.

---

## Executing train_yolo_models.sh

1. Ensure script permissions:
   `chmod +x train_yolo_models.sh`
2. Run the script:
   `./train_yolo_models.sh`
3. Monitor the training process for each model listed in the script.

---

## YOLOModelHandler Class Functions

- **Load Model**: Loads a YOLO model for operations.
- **Train Model**: Trains the model using specified dataset and epochs.
- **Validate Model**: Validates model performance on a dataset.
- **Predict Image**: Generates predictions for a given image.
- **Export Model**: Exports the trained model to different formats.

---

## SAHI Prediction with YOLO

- Utilise SAHI for improved prediction accuracy on large images.
- Slices large images into smaller segments for detailed analysis.
- Combines predictions from all segments for a comprehensive result.

---

## Conclusion

- Efficient and structured approach to handling YOLO models.
- Automates training and evaluation for multiple models.
- Employs SAHI for advanced image prediction capabilities.

Thank you for attending this presentation.