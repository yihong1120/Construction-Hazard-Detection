# Use the shared base image
FROM yihong123/base:latest

# 設定工作目錄
WORKDIR /app

# Copy the application code into the container
COPY examples/local_notification_server /app

# 安裝依賴
RUN pip install --no-cache-dir -r /app/requirements.txt

# 複製 local_notification_server 目錄內容到容器
# COPY . /app/examples/local_notification_server

# 複製 Firebase 憑證檔案（如有需要，可依實際情況調整）
COPY construction-harzard-detection-firebase-adminsdk-fbsvc-ca9d30aff7.json /app/examples/local_notification_server/

# Define environment variables for Redis configuration (can be overridden at runtime)
ENV REDIS_HOST=redis
ENV REDIS_PORT=6379
ENV REDIS_PASSWORD='password'

# Expose the application port
EXPOSE 8003

# 預設啟動 FastAPI 服務
CMD ["uvicorn", "examples.local_notification_server.app:app", "--host", "0.0.0.0", "--port", "8003"]
