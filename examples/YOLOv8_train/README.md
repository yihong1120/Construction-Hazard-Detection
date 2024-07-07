🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# YOLOv8 Train Example

This repository contains examples and scripts for training YOLOv8 object detection models. It is particularly aimed at enhancing the model's performance for specific datasets or applications.

## Usage

### Model Training

To train a YOLO model, use the `train.py` script. Specify the model name, the number of training epochs, and the data configuration:

```bash
python train.py --model_name 'yolov8n.pt' --epochs 100 --data_config 'dataset/data.yaml'
```

### Training YOLO Models with Script

You can automate the training process for different YOLO models using the `train_yolo_models.sh` script:

```bash
bash train_yolo_models.sh
```

This script will download base models if they are not present and start the training process for each.

### Model Exporting

After training, you can export your YOLO model to different formats, such as ONNX, using the `train.py` script:

```bash
python train.py --model_name 'yolov8n.pt' --export_format 'onnx' --onnx_path 'yolov8n.onnx'
```

### Model Prediction

To predict using a YOLO model, specify the image path for prediction:

```bash
python train.py --model_name 'yolov8n.pt' --predict_image 'path/to/image.jpg'
```

## Features

- **Training**: Improve your model's performance on specific datasets.
- **Model Training**: Train YOLO models from scratch or using pre-trained weights.
- **Model Exporting**: Export your trained models to different formats for deployment.
- **Model Prediction**: Easily predict using your trained models on new images.
- **Batch Training**: Use the provided shell script to train multiple YOLO models.

## Configuration

Modify the training parameters, such as epochs and data configuration, to suit your dataset requirements. You can also adjust the augmentation and hyperparameters within the YOLO configuration files.
