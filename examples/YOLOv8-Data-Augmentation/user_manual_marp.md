---
marp: true
theme: default
class:
  - lead
  - invert
paginate: true
---

# Data Augmentation and Bounding Box Visualisation

---

## Introduction

- **Objective**: Enhance image datasets for machine learning model training.
- **Components**:
  - Data Augmentation
  - File Renaming and Organisation
  - Bounding Box Visualisation

---

## DataAugmentation.py Overview

- **Purpose**: Augment image data for better model generalisation.
- **Features**:
  - Randomly applies a series of image transformations.
  - Supports bounding box adjustments for object detection tasks.
- **Usage**: Customisable for different dataset sizes and augmentation needs.

---

## Running the Data Augmentation Script

1. Set the dataset path and augmentation count:
   `--train_path './dataset_aug/train' --num_augmentations 30`
2. Execute the script:
    ```bash
    python data_augmentation.py
    ```
3. The script generates augmented images and corresponding label files.

---

## run_augmentation_and_move.sh Overview

- **Purpose**: Automate augmentation process and integrate new images into the dataset.
- **Features**:
  - Executes:
    ```bash
    data_augmentation.py
    ```
  - Renames and moves files to maintain dataset organisation.
- **Usage**: Ensures seamless augmentation and dataset expansion.

---

## Using run_augmentation_and_move.sh

1. Navigate to the script's directory.
2. Make the script executable:
    ```bash
    chmod +x run_augmentation_and_move.sh
    ```
3. Run the script:
    ```bash
    bash ./run_augmentation_and_move.sh
    ```
4. Augmented images are now part of the main dataset.

---

## visualise_bounding_boxes.py Overview

- **Purpose**: Visualise bounding boxes on images for verification.
- **Features**:
  - Draws bounding boxes based on label files.
  - Options to save or display the annotated images.
- **Usage**: Helps in verifying the correctness of label data.

---

## Visualising Bounding Boxes

1. Specify image and label paths:
   `--image 'path/to/image.jpg' --label 'path/to/label.txt'`
2. Choose to save or display the result:
   `--save` to save or omit to display.
3. Run the script:
    ```bash
    python visualise_bounding_boxes.py --image '...' --label '...' [--save]
    ```

---

## Conclusion

- **Efficient Data Augmentation**: Enhances training datasets for improved model performance.
- **Seamless Integration**: Automated scripts for easy augmentation and dataset management.
- **Quality Assurance**: Visualisation tools for bounding box verification.

Thank you for attending this presentation.

