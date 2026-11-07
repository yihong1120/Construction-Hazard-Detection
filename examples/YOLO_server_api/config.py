from __future__ import annotations

import logging
import warnings

lazy import torch

from src.settings.inference import ApiInferenceSettings

# Configuration parameters for YOLO Server API
logger = logging.getLogger(__name__)

settings = ApiInferenceSettings()
USE_TENSORRT = settings.use_tensorrt
USE_SAHI = settings.use_sahi
MAX_LOADED_MODELS = settings.max_loaded_models
YOLO_INFERENCE_DEVICE = settings.device


def _is_cuda_available() -> bool:
    """Return whether PyTorch can initialise CUDA on this host."""

    return torch.cuda.is_available()


def get_inference_device() -> str:
    """Return a safe Ultralytics/SAHI device string for this host."""
    if YOLO_INFERENCE_DEVICE and YOLO_INFERENCE_DEVICE != 'auto':
        return YOLO_INFERENCE_DEVICE
    try:
        return 'cuda:0' if _is_cuda_available() else 'cpu'
    except Exception as exc:
        warnings.warn(
            f"CUDA availability check failed, falling back to CPU: {exc}",
            UserWarning,
            stacklevel=2,
        )
        return 'cpu'


# Configuration validation: SAHI mode enforces .pt file usage and is
# incompatible with TensorRT
if USE_SAHI and USE_TENSORRT:
    warnings.warn(
        'USE_SAHI=True forces .pt model usage, overriding USE_TENSORRT=True',
        UserWarning,
        stacklevel=2,
    )

# Display current configuration
_CONFIG_INFO: str = f"""
🔧 YOLO Server API Configuration:
   • USE_TENSORRT: {USE_TENSORRT}
   • USE_SAHI: {USE_SAHI}
   • Model file format: {".pt" if USE_SAHI or not USE_TENSORRT else ".engine"}
   • Inference method: {
    "SAHI slicing"
    if USE_SAHI
    else "TensorRT"
    if USE_TENSORRT
    else "Standard YOLO"
}
   • Inference device: {YOLO_INFERENCE_DEVICE}
   • Published model capacity: {MAX_LOADED_MODELS}
   • Inference workers: {settings.concurrency}
   • Admitted requests: {settings.queue_capacity}
"""


def log_configuration() -> None:
    """Emit the active non-sensitive YOLO configuration at app startup."""
    logger.info('%s', _CONFIG_INFO)
