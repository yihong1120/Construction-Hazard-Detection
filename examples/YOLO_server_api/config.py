from __future__ import annotations

import os
import warnings

# Configuration parameters for YOLO Server API

# Whether to use TensorRT for inference
USE_TENSORRT: bool = os.getenv('USE_TENSORRT', 'false').lower() == 'true'

# Whether to use SAHI for sliced inference (only works with .pt models)
USE_SAHI: bool = os.getenv('USE_SAHI', 'false').lower() == 'true'

# Base model variants
MODEL_VARIANTS_ENV: str = os.getenv(
    'MODEL_VARIANTS', 'yolo26x,yolo26l,yolo26m,yolo26s,yolo26n',
)
MODEL_VARIANTS: list[str] = [
    v.strip() for v in MODEL_VARIANTS_ENV.split(',') if v.strip()
]

# Whether to enable lazy loading of models:
# True means models are loaded only when first used
LAZY_LOAD_MODELS: bool = (
    os.getenv('LAZY_LOAD_MODELS', 'true').lower() == 'true'
)

# Maximum number of models allowed in memory simultaneously in lazy loading
# mode (LRU eviction)
MAX_LOADED_MODELS: int = int(os.getenv('MAX_LOADED_MODELS', '3'))

# Whether to preload the smallest model at startup (only in lazy mode)
PRELOAD_SMALLEST: bool = (
    os.getenv('PRELOAD_SMALLEST', 'true').lower() == 'true'
)

# Whether to explicitly call torch.cuda.empty_cache() when releasing/evicting
# models
EXPLICIT_CUDA_CLEANUP: bool = (
    os.getenv('EXPLICIT_CUDA_CLEANUP', 'true').lower() == 'true'
)

# Inference device. Use "auto" to prefer CUDA only when PyTorch can initialise
# it successfully; this avoids reconnect loops on hosts where CUDA reports
# availability errors such as error 804.
YOLO_INFERENCE_DEVICE: str = os.getenv(
    'YOLO_INFERENCE_DEVICE',
    os.getenv('YOLO_DEVICE', 'auto'),
).strip().lower()


def get_inference_device() -> str:
    """Return a safe Ultralytics/SAHI device string for this host."""
    if YOLO_INFERENCE_DEVICE and YOLO_INFERENCE_DEVICE != 'auto':
        return YOLO_INFERENCE_DEVICE
    try:
        import torch

        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    except Exception as exc:
        warnings.warn(
            f'CUDA availability check failed, falling back to CPU: {exc}',
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
   • Model file format: {'.pt' if USE_SAHI or not USE_TENSORRT else '.engine'}
   • Inference method: {
    'SAHI slicing' if USE_SAHI else
    'TensorRT' if USE_TENSORRT else
    'Standard YOLO'
}
   • Inference device: {YOLO_INFERENCE_DEVICE}
   • Model variants: {', '.join(MODEL_VARIANTS)}
"""

print(_CONFIG_INFO)

# Type hints and docstrings added for clarity and maintainability


def get_model_variants() -> list[str]:
    """Retrieve the list of model variants from the environment variable.

    Returns:
        list[str]: A list of model variant names.
    """
    return MODEL_VARIANTS


def is_lazy_loading_enabled() -> bool:
    """Check if lazy loading of models is enabled.

    Returns:
        bool: True if lazy loading is enabled, False otherwise.
    """
    return LAZY_LOAD_MODELS


def get_max_loaded_models() -> int:
    """Get the maximum number of models allowed in memory simultaneously.

    Returns:
        int: The maximum number of models.
    """
    return MAX_LOADED_MODELS


def should_preload_smallest_model() -> bool:
    """Determine if the smallest model should be preloaded at startup.

    Returns:
        bool: True if the smallest model should be preloaded, False otherwise.
    """
    return PRELOAD_SMALLEST


def should_cleanup_cuda_cache() -> bool:
    """Check if explicit CUDA cache cleanup is enabled.

    Returns:
        bool: True if CUDA cache cleanup is enabled, False otherwise.
    """
    return EXPLICIT_CUDA_CLEANUP
