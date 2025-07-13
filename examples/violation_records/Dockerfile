# Use the shared base image
FROM yihong123/base:latest

WORKDIR /app

COPY . /app

RUN pip install --no-cache-dir -r /app/requirements.txt

EXPOSE 8081

CMD ["uvicorn", "examples.violation_records.app:app", "--host", "0.0.0.0", "--port", "8081"]
