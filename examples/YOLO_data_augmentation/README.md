🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLO Data Augmentation Example

This repository contains examples and scripts for performing data augmentation on image datasets, particularly aimed at enhancing the performance of YOLO object detection models. It is particularly useful for machine learning projects requiring enriched datasets for improved model training.

## Usage

### Data Augmentation

To perform data augmentation on your dataset, use the `data_augmentation.py` script. Specify the path to your training data and the number of augmentations per image:

```bash
python data_augmentation.py --train_path '../path/to/your/data' --num_augmentations 30
```

### Moving and Renaming Augmented Files

After augmentation, you can use the `run_augmentation_and_move.sh` script to move and rename augmented images and label files to the main dataset directory:

```bash
bash run_augmentation_and_move.sh
```

This script ensures that the augmented files are correctly sequenced and stored with the existing dataset files.

### Visualising Bounding Boxes

To visualise the bounding boxes on an image, use the `visualise_bounding_boxes.py` script. Provide the paths to the image and the corresponding label file:

```bash
python src/visualise_bounding_boxes.py --image 'path/to/image.jpg' --label 'path/to/label.txt'
```

You can choose to save the visualised image by using the `--save` flag and specifying the `--output` path.

## Features

- **Data Augmentation**: Enhance your image datasets by applying a series of randomised transformations.
- **Parallel Processing**: Utilise multi-threading for faster augmentation processing.
- **Bounding Box Preservation**: Ensure that augmented images maintain accurate bounding box annotations.
- **Shuffling**: Randomise the order of the dataset to prevent model overfitting.
- **Visualisation**: Easily visualise bounding boxes on your images for verification purposes.

## Configuration

The data augmentation process can be customised by modifying the augmentation sequences within the `data_augmentation.py` script. You can add or remove transformations as per your dataset requirements.
