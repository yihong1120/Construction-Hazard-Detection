from __future__ import annotations

import datetime
import hashlib
from pathlib import Path
from typing import Final

import torch


VALID_MODEL_FILES: Final[dict[str, str]] = {
    'yolo26n': 'best_yolo26n.pt',
    'yolo26s': 'best_yolo26s.pt',
    'yolo26m': 'best_yolo26m.pt',
    'yolo26l': 'best_yolo26l.pt',
    'yolo26x': 'best_yolo26x.pt',
}


def _model_destination_path(model: str) -> Path:
    """Return the destination path for a managed model file."""
    if model not in VALID_MODEL_FILES:
        raise ValueError(
            f"Invalid model key: {model}. "
            f"Must be one of {list(VALID_MODEL_FILES.keys())}.",
        )

    base_dir = Path('models/pt').resolve()
    destination_path = (base_dir / VALID_MODEL_FILES[model]).resolve()
    try:
        destination_path.relative_to(base_dir)
    except ValueError:
        raise ValueError('Attempted path traversal in destination path.')
    return destination_path


async def update_model_file(model: str, model_file: Path) -> None:
    """
    Update the model file for a specified model.

    Args:
        model (str): The model key (e.g., 'yolo26n', 'yolo26s').
        model_file (Path): The path to the new `.pt` model file.
    """
    destination_path = _model_destination_path(model)

    if not model_file.is_file() or model_file.suffix != '.pt':
        raise ValueError(
            f"Invalid file: {model_file}. Must be a valid `.pt` file.",
        )

    try:
        torch.jit.load(str(model_file))
    except Exception as e:
        raise ValueError(f"Invalid PyTorch model file: {e}")

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Move the model file to the destination path
        model_file.rename(destination_path)
    except Exception as e:
        raise OSError(f"Failed to update model file: {e}")


async def get_new_model_path(
    model: str, last_update_time: datetime.datetime,
) -> Path | None:
    """Return the updated model path without loading its bytes into memory.

    Args:
        model (str): The model key (e.g., 'yolo26n', 'yolo26s').
        last_update_time (datetime.datetime): The last update time
            provided by the user.

    Returns:
        Path | None: Updated model path if available, else None.
    """
    destination_path = _model_destination_path(model)

    if not destination_path.is_file():
        return None

    file_mod_time = datetime.datetime.fromtimestamp(
        destination_path.stat().st_mtime,
    )
    if file_mod_time > last_update_time:
        return destination_path
    return None


def model_file_etag(model_path: Path) -> str:
    """Return the strong ETag derived from the model file's SHA-256 digest."""
    return f'"{model_file_checksum(model_path)}"'


def model_file_checksum(model_path: Path) -> str:
    """Calculate a streaming SHA-256 checksum without loading the model."""
    digest = hashlib.sha256()
    with model_path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
