🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

<img width="100%" src="./assets/images/project_graphics/banner.gif" alt="AI-Driven Construction Safety Banner">

<div align="center">
   <a href="examples/YOLOv8_server_api">Server-API</a> |
   <a href="examples/streaming_web">Streaming-Web</a> |
   <a href="examples/user_management">User-Management</a> |
   <a href="examples/YOLOv8_data_augmentation">Data-Augmentation</a> |
   <a href="examples/YOLOv8_evaluation">Evaluation</a> |
   <a href="examples/YOLOv8_train">Train</a>
</div>

<br>

<div align="center">
   <a href="https://github.com/pre-commit/pre-commit">
      <img src="https://img.shields.io/badge/pre--commit-3.7.1-blue?logo=pre-commit" alt="Pre-commit 3.7.1">
   </a>
   <a href="https://www.python.org/downloads/release/python-3124/">
      <img src="https://img.shields.io/badge/python-3.12.4-blue?logo=python" alt="Python 3.12.4">
   </a>
   <a href="https://github.com/ultralytics/ultralytics">
      <img src="https://img.shields.io/badge/YOLOv8-ultralytics-blue?logo=yolo" alt="YOLOv8">
   </a>
   <a href="https://flask.palletsprojects.com/en/3.0.x/">
      <img src="https://img.shields.io/badge/flask-3.0.3-blue?logo=flask" alt="Flask 3.0.3">
   </a>
   <a href="https://docs.pytest.org/en/8.2.x/">
      <img src="https://img.shields.io/badge/pytest-8.2.2-blue?logo=pytest" alt="pytest 8.2.2">
   </a>
</div>

<br>

"Construction-Hazard-Detection" is an AI-driven tool aimed at enhancing safety on construction sites. Utilising the YOLOv8 model for object detection, this system identifies potential hazards such as workers without helmets, workers without safety vests, workers in close proximity to machinery, and workers near vehicles. Post-processing algorithms are employed to enhance the accuracy of the detections. The system is designed for deployment in real-time environments, providing immediate analysis and alerts for detected hazards.

<br>
<br>

<div align="center">
   <img src="./assets/images/hazard-detection.png" alt="diagram" style="width: 100%;">
</div>

<br>


## Usage

Before running the application, you need to configure the system by specifying the details of the video streams and other parameters in a YAML configuration file. An example configuration file `configuration.yaml` should look like this:

```yaml
# This is a list of video configurations
- video_url: "rtsp://example1.com/stream"  # URL of the video
   image_name: "cam1"  # Name of the image
   label: "label1"  # Label of the video
   model_key: "yolov8n"  # Model key for the video
   line_token: "token1"  # Line token for notification
   run_local: True  # Run object detection locally
- video_url: "rtsp://example2.com/stream"
   image_name: "cam2"
   label: "label2"
   model_key: "yolov8n"
   line_token: "token2"
   run_local: True
```

Each object in the array represents a video stream configuration with the following fields:

- `video_url`: The URL of the live video stream. This can include:
   - Surveillance streams
   - RTSP streams
   - Secondary streams
   - YouTube videos or live streams
   - Discord streams
- `image_name`: The name assigned to the image or camera.
- `label`: The label assigned to the video stream.
- `model_key`: The key identifier for the machine learning model to use.
- `line_token`: The LINE messaging API token for sending notifications.
- `run_local`: Boolean value indicating whether to run object detection locally.

<br>

Now, you could launch the hazard-detection system in Docker or Python env:

<details>
   <summary>Docker</summary>

   ### Usage for Docker

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

</details>

<details>
   <summary>Python</summary>

   ### Usage for Python

   To run the hazard detection system with Python, follow these steps:

   1. Clone the repository to your local machine:
      ```bash
      git clone https://github.com/yihong1120/Construction-Hazard-Detection.git
      ```

   2. Navigate to the cloned directory:
      ```bash
      cd Construction-Hazard-Detection
      ```

   3. Install required packages:
      ```bash
      pip install -r requirements.txt
      ```

   4. Install and launch MySQL service (if required):
      ```bash
      sudo apt install mysql-server
      sudo systemctl start mysql.service
      ```

   5. Start user management API:
      ```bash
      gunicorn -w 1 -b 0.0.0.0:8000 "examples.User-Management.app:user-managements-app"
      ```

   6. Run object detection API:
      ```bash
      gunicorn -w 1 -b 0.0.0.0:8001 "examples.Model-Server.app:app"
      ```

   7. Run the main application with a specific configuration file:
      ```bash
      python3 main.py --config /path/to/your/configuration.yaml
      ```
      Replace `/path/to/your/configuration.yaml` with the actual path to your configuration file.

   8. Start the streaming web service:
      ```bash
      gunicorn -w 1 -k eventlet -b 127.0.0.1:8002 "examples.Stream-Web.app:streaming-web-app"
      ```

</details>

## Additional Information

- The system logs are available within the Docker container and can be accessed for debugging purposes.
- The output images with detections (if enabled) will be saved to the specified output path.
- Notifications will be sent through LINE messaging API during the specified hours if hazards are detected.

### Notes

- Ensure that the `Dockerfile` is present in the root directory of the project and is properly configured as per your application's requirements.
- The `-p 8080:8080` flag maps port 8080 of the container to port 8080 on your host machine, allowing you to access the application via the host's IP address and port number.

For more information on Docker usage and commands, refer to the [Docker documentation](https://docs.docker.com/).

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

<details>
   <summary>Models for detection</summary>

   | Model   | size<br><sup>(pixels) | mAP<sup>val<br>50 | mAP<sup>val<br>50-95 | params<br><sup>(M) | FLOPs<br><sup>(B) |
   | ------- | --------------------- | ------------------ | ------------------ | ----------------- | ----------------- |
   | YOLOv8n | 640                   | //                 | //                 | 3.2               | 8.7               |
   | YOLOv8s | 640                   | //                 | //                 | 11.2              | 28.6              |
   | YOLOv8m | 640                   | //                 | //                 | 25.9              | 78.9              |
   | YOLOv8l | 640                   | //                 | //                 | 43.7              | 165.2             |
   | YOLOv8x | 640                   | 82.9               | 60.9               | 68.2              | 257.8             |

</details>

<br>

Our comprehensive dataset ensures that the model is well-equipped to identify a wide range of potential hazards commonly found in construction environments.

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
