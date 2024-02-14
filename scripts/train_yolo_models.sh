#!/bin/bash

# Define models and their download links
declare -A models=(
    ["yolov8n.pt"]="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n.pt"
    ["yolov8s.pt"]="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt"
    ["yolov8m.pt"]="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8m.pt"
    ["yolov8l.pt"]="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8l.pt"
    ["yolov8x.pt"]="https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8x.pt"
)

# Ensure the models directory exists
mkdir -p models

# Check and download models
for model in "${!models[@]}"; do
    if [ ! -f "models/$model" ]; then
        echo "Model $model does not exist, downloading..."
        wget -O "models/$model" "${models[$model]}"
    fi
done

# Train each model in turn
for model in "${!models[@]}"; do
    echo "Starting training $model ..."
    python ../src/train.py --model_name "../models/$model" --epochs 100 --pt_path "../models/trained_$model"
    echo "$model training completed."
done

echo "All model training completed."
