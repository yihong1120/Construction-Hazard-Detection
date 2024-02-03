🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# Construction-Hazard-Detection

"Construction-Hazard-Detection" is an AI-driven tool aimed at enhancing safety at construction sites. Utilising the YOLOv8 model for object detection, this system identifies potential hazards like overhead heavy loads and steel pipes. Post-processing is applied to the trained model for improved accuracy. The system is designed for deployment in real-time environments, providing instant analysis and warnings for any detected hazards.

## Dataset Information
The primary dataset for training this model is the [Construction Site Safety Image Dataset from Roboflow](https://www.kaggle.com/datasets/snehilsanyal/construction-site-safety-image-dataset-roboflow/data). We have enriched this dataset with additional annotations and made it openly accessible on Roboflow. The enhanced dataset can be found here: [Construction Hazard Detection on Roboflow](https://universe.roboflow.com/side-projects/construction-hazard-detection). This dataset includes the following labels:

- `0: 'Hardhat'`
- `1: 'Mask'`
- `2: 'NO-Hardhat'`
- `3: 'NO-Mask'`
- `4: 'NO-Safety Vest'`
- `5: 'Person'`
- `6: 'Safety Cone'`
- `7: 'Safety Vest'`
- `8: 'Machinery'`
- `9: 'Vehicle'`

Our comprehensive dataset ensures that the model is well-equipped to identify a wide range of potential hazards commonly found in construction environments.

## Installation Guide
To set up this project, follow these steps:
1. Clone the repository:
   ```
   git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
   ```
2. Navigate to the project directory:
   ```
   cd Construction-Hazard-Detection
   ```
3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

This system is designed for real-time detection and alerting of hazards at construction sites. Follow these detailed steps to utilise the system effectively:

### Preparing the Environment
1. **Setting Up the Hardware**: Ensure you have a computer with sufficient processing power and a high-quality camera for capturing live footage at the construction site.

2. **Camera Installation**: Position the camera strategically to cover high-risk areas where heavy loads and steel pipes are handled.

### Training the Model
1. **Gathering Data**: Collect images or videos of construction sites, focusing on various types of hazards such as heavy loads, steel pipes, and human presence.

2. **Data Annotation**: Annotate the collected data to identify and label the hazards and human figures accurately.

3. **Training YOLOv8**: Use the annotated dataset to train the YOLOv8 model. This can be done using the following command:
   ```
   python train.py --img 640 --batch 16 --epochs 50 --data dataset.yaml --weights yolov8n.pt
   ```
   Adjust the parameters based on your dataset and hardware capabilities.

### Post-processing and Deployment
1. **Applying Post-processing**: After training, apply post-processing techniques to enhance the model's accuracy in differentiating between hazardous and non-hazardous conditions.

2. **Model Integration**: Integrate the trained model with a software that can process the live feed from the camera.

3. **Running the System**: Start the system using the following command:
   ```
   python detect.py --source 0 --weights best.pt --conf-thres 0.4
   ```
   This command will initiate the detection process using your camera feed (`--source 0` refers to the default camera; change it if necessary). The `--weights` option should point to your trained model, and `--conf-thres` sets the confidence threshold for detection.

### Real-time Monitoring and Alerting
1. **Monitoring**: The system will continuously analyse the live feed from the construction site, detecting any potential hazards.

2. **Alerting**: When the system detects a human under a hazardous condition, it will trigger an alert. Ensure to have a mechanism (like a connected alarm or notification system) to notify the site personnel immediately.

## Contributing
We welcome contributions to this project. Please follow these steps:
1. Fork the repository.
2. Make your changes.
3. Submit a pull request with a clear description of your improvements.

## Development Roadmap
- [x] Data collection and preprocessing.
- [ ] Training YOLOv8 model with construction site data.
- [ ] Developing post-processing techniques for enhanced accuracy.
- [ ] Implementing real-time analysis and alert system.
- [ ] Testing and validation in simulated environments.
- [ ] Deployment in actual construction sites for field testing.
- [ ] Ongoing maintenance and updates based on user feedback.

## License
This project is licensed under the [AGPL-3.0 License](LICENSE.md).
