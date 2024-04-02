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

#### Gathering Data
Collect images or videos of construction sites, focusing on various hazards such as heavy loads, steel pipes, and human presence. For examples of data augmentation techniques, visit [YOLOv8 Data Augmentation Examples](examples/YOLOv8-Data-Augmentation).

#### Data Annotation
Annotate your data to accurately identify and label hazards and human figures. Detailed annotation guidelines are available in the dataset documentation.

#### Training YOLOv8
Use the annotated dataset to train the YOLOv8 model. Execute the following command, adjusting parameters based on your dataset and hardware capabilities:
```
python train.py --model_name 'yolov8n.pt' --epochs 100 --data_config 'dataset/data.yaml'
```
For training guidelines and advanced training options, refer to [YOLOv8 Train Examples](examples/YOLOv8-Train).

### Post-processing and Deployment

#### Applying Post-processing
After training, enhance the model's accuracy with post-processing techniques. For post-processing methods and evaluation strategies, see our detailed guide in [YOLOv8 Evaluation Examples](examples/YOLOv8-Evaluation).

#### Evaluate YOLOv8
Evaluating the performance of your YOLOv8 model is crucial for ensuring its effectiveness in real-world scenarios. We provide two main evaluation strategies: direct YOLOv8 model evaluation and combined SAHI+YOLOv8 evaluation for improved detection in complex scenes. To understand how to apply these methods and interpret the results, visit [YOLOv8 Evaluation Examples](examples/YOLOv8-Evaluation).

#### Model Integration and System Running
Integrate the trained model with software that can process the live feed from the camera. Start the system with the following command, which initiates the detection process using your camera feed:
```
python src/demo.py
```

### Real-time Monitoring and Alerting
1. **Monitoring**: The system will continuously analyse the live feed from the construction site, detecting any potential hazards.

2. **Alerting**: When the system detects a human under a hazardous condition, it will trigger an alert. Ensure to have a mechanism (like a connected alarm or notification system) to notify the site personnel immediately.

## Deployment Guide

To deploy the "Construction-Hazard-Detection" system using Docker, follow these steps:

### Building the Docker Image
1. Ensure Docker Desktop is installed and running on your machine.
2. Open a terminal and navigate to the root directory of the cloned repository.
3. Build the Docker image with the following command:
   ```
   docker build -t construction-hazard-detection .
   ```

### Running the Docker Container
1. Once the image is built, you can run the container using the following command:
   ```
   docker run -p 8080:8080 -e LINE_TOKEN=your_actual_token construction-hazard-detection
   ```
   Replace `your_actual_token` with your actual LINE Notify token.

   This command will start the container and expose port 8080 for the application, allowing you to access it from your host machine at `http://localhost:8080`.

### Notes
- Ensure that the `Dockerfile` is present in the root directory of the project and is properly configured as per your application's requirements.
- The `-e LINE_TOKEN=your_actual_token` flag sets the `LINE_TOKEN` environment variable inside the container, which is necessary for the application to send notifications. If you have other environment variables, you can add them in a similar manner.
- The `-p 8080:8080` flag maps port 8080 of the container to port 8080 on your host machine, allowing you to access the application via the host's IP address and port number.

For more information on Docker usage and commands, refer to the [Docker documentation](https://docs.docker.com/).

## Contributing
We welcome contributions to this project. Please follow these steps:
1. Fork the repository.
2. Make your changes.
3. Submit a pull request with a clear description of your improvements.

## Development Roadmap
- [x] Data collection and preprocessing.
- [x] Training YOLOv8 model with construction site data.
- [x] Developing post-processing techniques for enhanced accuracy.
- [x] Implementing real-time analysis and alert system.
- [x] Testing and validation in simulated environments.
- [x] Deployment in actual construction sites for field testing.
- [x] Ongoing maintenance and updates based on user feedback.

## License
This project is licensed under the [AGPL-3.0 License](LICENSE.md).
