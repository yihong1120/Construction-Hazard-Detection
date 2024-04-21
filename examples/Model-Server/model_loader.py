from sahi import AutoDetectionModel

def load_models():
    MODELS = {
        # 可根据需要解除注释并加载不同的模型
        # 'yolov8n': AutoDetectionModel.from_pretrained("yolov8", model_path='models/best_yolov8n.pt'),
        # 'yolov8s': AutoDetectionModel.from_pretrained("yolov8", model_path='models/best_yolov8s.pt'),
        # 'yolov8m': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8m.pt', device="cuda:0"),
        'yolov8l': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8l.pt', device="cuda:0"),
        'yolov8x': AutoDetectionModel.from_pretrained("yolov8", model_path='models/pt/best_yolov8x.pt', device="cuda:0")
    }
    return MODELS

# 使模型在全局范围内可用
MODELS = load_models()
