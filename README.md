🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

<img width="100%"  src="./assets/images/project_graphics/banner.gif" alt="AI-Driven Construction Safety Banner">

<br>

<div align="center">
    <a href="examples
/YOLOv8_server_api">YOLOv8-Server-API</a> | <a href="examples
/streaming_web">Streaming-Web</a> | <a href="examples
/user_management">User-Management</a> | <a href="examples
/YOLOv8_data_augmentation">YOLOv8-Data-Augmentation</a> | <a href="examples
/YOLOv8_evaluation">YOLOv8-Evaluation</a> | <a href="examples
/YOLOv8_train">YOLOv8-Train</a>
</div>

<div align="center">
    <a href="https://github.com/pre-commit/pre-commit"><img src="https://img.shields.io/badge/pre--commit-3.7.1-blue?logo=pre-commit" alt="Pre-commit 3.7.1"></a>
    <a href="https://www.python.org/downloads/release/python-3124/"><img src="https://img.shields.io/badge/python-3.12.4-blue?logo=python" alt="Python 3.12.4"></a>
    <a href="https://github.com/ultralytics/yolov8"><img src="https://img.shields.io/badge/YOLOv8-ultralytics-blue?logo=yolo" alt="YOLOv8"></a>
    <a href="https://flask.palletsprojects.com/en/3.0.x/"><img src="https://img.shields.io/badge/flask-3.0.3-blue?logo=flask" alt="Flask 3.0.3"></a>
</div>

<br>

"Construction-Hazard-Detection" is an AI-driven tool aimed at enhancing safety at construction sites. Utilising the YOLOv8 model for object detection, this system identifies potential hazards like overhead heavy loads and steel pipes. Post-processing is applied to the trained model for improved accuracy. The system is designed for deployment in real-time environments, providing instant analysis and warnings for any detected hazards.

## Configuration

Before running the application, you need to configure the system by specifying the details of the video streams and other parameters in a YAML configuration file. An example configuration file `configuration.yaml` should look like this:

```yaml
# This is a list of video configurations
- video_url: "rtsp://example1.com/stream"  # URL of the video
  image_name: "cam1"  # Name of the image
  label: "label1"  # Label of the video
  model_key: "yolov8x"  # Model key for the video
  line_token: "token1"  # Line token for notification
  run_local: True  # Run objection detection locally
- video_url: "rtsp://example2.com/stream"
  image_name: "cam2"
  label: "label2"
  model_key: "yolov8x"
  line_token: "token2"
  run_local: True
```

Each object in the array represents a video stream configuration with the following fields:

- `video_url`: The URL of the live video stream.
- `model_key`: The key identifier for the machine learning model to use.
- `line_token`: The LINE messaging API token for sending notifications.

## Usage for Docker

To run the hazard detection system, you need to have Docker and Docker Compose installed on your machine. Follow these steps to get the system up and running:

1. Clone the repository to your local machine.
   ```
   git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
   ```
2. Navigate to the cloned directory.
   ```
   cd Construction-Hazard-Detection
   ```
3. Build and run the services using Docker Compose:

   ```bash
   docker-compose up --build
   ```

4. To run the main application with a specific configuration file, use the following command:

   ```bash
   docker-compose run main-application python main.py --config /path/in/container/configuration.yaml
   ```

   Replace `/path/in/container/configuration.yaml` with the actual path to your configuration file inside the container.

5. To stop the services, use the following command:

   ```bash
   docker-compose down
   ```

## Usage for Python

To run the hazard detection system with terminal, you need to have Python installed on your machine. Follow these steps to get the system up and running:

1. Clone the repository to your local machine.
   ```bash
   git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
   ```

2. Navigate to the cloned directory.
   ```bash
   cd Construction-Hazard-Detection
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Install and launch MySQL service:
   ```bash
   sudo apt install mysql-server
   sudo systemctl start mysql.service
   ```

5. Set up user accounts and passcodes. Use the following command to start the user management API:
   ```bash
   gunicorn -w 1 -b 0.0.0.0:8000 "examples.User-Management.app:user-managements-app"
   ```
   It is suggested to use Postman app to interact with the app.

6. To run the object-detection API, use the following command:
   ```bash
   gunicorn -w 1 -b 0.0.0.0:8001 "examples.Model-Server.app:app"
   ```

7. To run the main application with a specific configuration file, use the following command:
   ```bash
   python3 main.py --config /path/to/your/configuration.yaml
   ```
   Replace `/path/to/your/configuration.yaml` with the actual path to your configuration file.

8. To start the streaming web service, execute the following command:
   ```bash
   gunicorn -w 1 -k eventlet -b 127.0.0.1:8002 "examples.Stream-Web.app:streaming-web-app"
   ```

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

## Additional Information

- The system logs are available within the Docker container and can be accessed for debugging purposes.
- The output images with detections (if enabled) will be saved to the specified output path.
- Notifications will be sent through LINE messaging API during the specified hours if hazards are detected.

### Notes
- Ensure that the `Dockerfile` is present in the root directory of the project and is properly configured as per your application's requirements.
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
