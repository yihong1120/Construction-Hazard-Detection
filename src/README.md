🇬🇧 [English](./src/README.md) | 🇹🇼 [繁體中文](./src/README-zh-tw.md)

## Overview

This repository contains a collection of Python scripts designed for various functionalities including live stream detection, drawing management, model fetching, logging, and sending notifications through different platforms. The project is structured to facilitate easy integration and usage of these functionalities.

## Directory Structure

```
src
├── danger_detector.py
├── drawing_manager.py
├── __init__.py
├── lang_config.py
├── live_stream_detection.py
├── live_stream_tracker.py
├── model_fetcher.py
├── monitor_logger.py
├── notifiers
│   ├── broadcast_notifier.py
│   ├── __init__.py
│   ├── line_notifier.py
│   ├── messenger_notifier.py
│   ├── telegram_notifier.py
│   └── wechat_notifier.py
├── stream_capture.py
└── stream_viewer.py
```

## File Descriptions

### Main Modules

- **danger_detector.py**: Contains the [`DangerDetector`](./src/danger_detector.py) class for detecting potential safety hazards based on detection data.
- **drawing_manager.py**: Contains the [`DrawingManager`](./src/drawing_manager.py) class for drawing detections on frames and saving them.
- **lang_config.py**: Configuration file for language settings.
- **live_stream_detection.py**: Contains the [`LiveStreamDetector`](./src/live_stream_detection.py) class for performing live stream detection and tracking using YOLOv8 with SAHI.
- **live_stream_tracker.py**: Contains the [`LiveStreamDetector`](./src/live_stream_tracker.py) class for performing live stream detection and tracking using YOLOv8.
- **model_fetcher.py**: Contains functions to download model files if they do not already exist.
- **monitor_logger.py**: Contains the [`LoggerConfig`](./src/monitor_logger.py) class for setting up application logging with console and file handlers.
- **stream_capture.py**: Contains the [`StreamCapture`](./src/stream_capture.py) class for capturing frames from a video stream.
- **stream_viewer.py**: Contains the [`StreamViewer`](./src/stream_viewer.py) class for viewing video streams.

### Notifiers

- **notifiers/broadcast_notifier.py**: Contains the [`BroadcastNotifier`](./src/notifiers/broadcast_notifier.py) class for sending messages to a broadcast system.
- **notifiers/line_notifier.py**: Contains the [`LineNotifier`](./src/notifiers/line_notifier.py) class for sending notifications via LINE Notify.
- **notifiers/messenger_notifier.py**: Contains the [`MessengerNotifier`](./src/notifiers/messenger_notifier.py) class for sending notifications via Facebook Messenger.
- **notifiers/telegram_notifier.py**: Contains the [`TelegramNotifier`](./src/notifiers/telegram_notifier.py) class for sending notifications via Telegram.
- **notifiers/wechat_notifier.py**: Contains the [`WeChatNotifier`](./src/notifiers/wechat_notifier.py) class for sending notifications via WeChat Work.

## Usage

### Setting Up Environment Variables

Ensure you have a `.env` file in the root directory with the necessary environment variables for the notifiers, such as:

```
WECHAT_CORP_ID=your_wechat_corp_id
WECHAT_CORP_SECRET=your_wechat_corp_secret
WECHAT_AGENT_ID=your_wechat_agent_id
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
FACEBOOK_PAGE_ACCESS_TOKEN=your_facebook_page_access_token
LINE_NOTIFY_TOKEN=your_line_notify_token
```

### Running the Scripts

Each script can be run individually. For example, to run the `live_stream_tracker.py` script:

```bash
python live_stream_tracker.py --url <your_stream_url> --model <path_to_yolo_model>
```

### Example Usage

Refer to the `main` function in each script for example usage.
