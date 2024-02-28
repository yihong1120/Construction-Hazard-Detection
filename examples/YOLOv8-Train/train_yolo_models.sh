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
mkdir -p ../models

# Check and use pre-trained models if they exist
for model in "${!models[@]}"; do
    trained_model_path="../models/trained_$model"
    if [ -f "$trained_model_path" ]; then
        echo "Pre-trained model $trained_model_path exists, will be used for training."
        model_path="$trained_model_path"  # Use the pre-trained model for training
    else
        echo "Model $model does not exist or hasn't been trained, checking for base model..."
        if [ ! -f "../models/$model" ]; then
            echo "Base model $model does not exist, downloading..."
            wget -O "../models/$model" "${models[$model]}"
        fi
        model_path="../models/$model"
    fi

    # Start training
    echo "Starting training $model with $model_path ..."
    python ../src/train.py --model_name "$model_path" --epochs 100 --pt_path "$trained_model_path"
    echo "$model training completed."
done

echo "All model training completed."