---
marp: true
theme: default
class:
  - lead
  - invert
paginate: true
---

# Live Stream Hazard Detection and Notification

---

## Introduction

- **Purpose**: Automate hazard detection in live streams using YOLOv8.
- **Functionality**:
  - Detects hazards in real-time video streams.
  - Sends notifications with relevant information.
  - Logs events and alerts.

---

## Setup and Configuration

1. **Environment Variables**:
   - `VIDEO_URL`: URL of the live video stream.
   - `MODEL_PATH`: File path of the YOLOv8 detection model.
   - `IMAGE_PATH`: Path for images to send with notifications (optional).

2. **Logging**:
   - Utilises custom logging configurations for monitoring and debugging.

---

## Implementation Details

- **LiveStreamDetector**:
  - Monitors live video streams for potential hazards.
- **LineNotifier**:
  - Sends notifications via LINE messaging service.
- **DangerDetector**:
  - Analyses detections to identify potential hazards.

---

## Main Execution Flow

1. Initialise components: Live stream detector, LINE notifier, and danger detector.
2. Process detections in real-time:
   - Convert timestamps, detect dangers, and compile warnings.
3. Send notifications:
   - If new warnings and a sufficient time has passed since the last notification.

---

## Notification Logic

- **Frequency Control**: Ensures a 5-minute interval between notifications.
- **Unique Warnings**: Filters and sends distinct warnings only.
- **Feedback**:
   - Success or failure of each notification attempt is logged.

---

## Resource Management

- Proper release of video stream and other resources after processing.
- Ensures efficient memory usage and avoids resource leaks.

---

## Running the Script

- Load environment variables from `.env` or system defaults.
- Input prompts for missing configuration details.
- Execute with:
  ```bash
  python demo.py
  ```

---

## Conclusion

- Streamlines the process of monitoring live streams for hazards.
- Integrates notification systems for immediate alerting.
- Facilitates logging and alert management for enhanced safety measures.

Thank you for attending this presentation.