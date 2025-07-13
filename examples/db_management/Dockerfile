# Use the shared base image
FROM yihong123/base:latest

# 設定工作目錄
WORKDIR /app

# Copy the application code into the container
COPY examples/db_management /app

# 安裝依賴
RUN pip install --no-cache-dir -r /app/requirements.txt

# 複製 db_management 目錄內容到容器
# COPY . /app/examples/db_management

# Expose the application port
EXPOSE 8005

# 設定啟動指令
CMD ["uvicorn", "examples.db_management.app:app", "--host", "0.0.0.0", "--port", "8005"]
