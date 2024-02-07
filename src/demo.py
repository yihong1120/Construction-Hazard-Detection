from site_safety_monitor import detect_danger
from line_notifier import send_line_notification
from monitor_logger import setup_logging
from datetime import datetime

# 假設這是從你的檢測系統獲得的檢測結果
detections = [
    # ...Your detection results
]

def main(logger):
    # 使用 site_safety_monitor.py 中的檢測功能
    warnings = detect_danger(detections)
    
    # 如果有警告，則通過 LINE Chatbot 發送並記錄
    if warnings:
        line_token = 'YOUR_LINE_NOTIFY_TOKEN'  # 你的 LINE Notify 的 token
        for warning in warnings:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            message = f'[{current_time}] {warning}'
            send_line_notification(line_token, message)
            logger.warning(message)  # 記錄警告

if __name__ == '__main__':
    logger = setup_logging()  # 設定日誌
    main(logger)