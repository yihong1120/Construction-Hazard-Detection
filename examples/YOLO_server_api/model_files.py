from __future__ import annotations

from pathlib import Path
import torch
import datetime
import hashlib

async def compute_file_hash(file_path: Path) -> str:
    """
    Compute the SHA-256 hash of a file.

    Args:
        file_path (Path): The path to the file.

    Returns:
        str: The SHA-256 hash of the file.
    """
    sha256 = hashlib.sha256()
    with file_path.open('rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

async def update_model_file(model: str, model_file: Path) -> None:
    """
    Update the model file for a specified model.

    Args:
        model (str): The model key (e.g., 'yolo11n', 'yolo11s').
        model_file (Path): The path to the new `.pt` model file.
    """
    valid_models = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']
    if model not in valid_models:
        raise ValueError(f"Invalid model key: {model}. Must be one of {valid_models}.")

    if not model_file.is_file() or model_file.suffix != '.pt':
        raise ValueError(f"Invalid file: {model_file}. Must be a valid `.pt` file.")

    # Compute the hash of the uploaded model file
    uploaded_file_hash = await compute_file_hash(model_file)

    # Retrieve the stored hash for the model
    stored_hashes = {
        'yolo11n': 'stored_hash_for_yolo11n',
        'yolo11s': 'stored_hash_for_yolo11s',
        'yolo11m': 'stored_hash_for_yolo11m',
        'yolo11l': 'stored_hash_for_yolo11l',
        'yolo11x': 'stored_hash_for_yolo11x',
    }
    if uploaded_file_hash != stored_hashes.get(model):
        raise ValueError("The uploaded model file's hash does not match the stored hash.")

    try:
        # Validate the model file by loading it with torch.jit.load
        torch.jit.load(model_file)
    except Exception as e:
        raise ValueError(f"Invalid PyTorch model file: {e}")

    destination_path = Path(f'models/pt/best_{model}.pt')
    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        model_file.rename(destination_path)
    except Exception as e:
        raise IOError(f"Failed to update model file: {e}")


async def get_new_model_file(model: str, last_update_time: datetime.datetime) -> bytes | None:
    """
    Retrieve the new model file if updated since the provided time.

    Args:
        model (str): The model key (e.g., 'yolo11n', 'yolo11s').
        last_update_time (datetime): The last update time provided by the user.

    Returns:
        bytes | None: Model file content if updated, else None.
    """
    valid_models = ['yolo11n', 'yolo11s', 'yolo11m', 'yolo11l', 'yolo11x']
    if model not in valid_models:
        raise ValueError(f"Invalid model key: {model}. Must be one of {valid_models}.")

    destination_path = Path(f'models/pt/best_{model}.pt')
    if not destination_path.is_file():
        return None

    file_mod_time = datetime.datetime.fromtimestamp(destination_path.stat().st_mtime)
    if file_mod_time > last_update_time:
        try:
            with destination_path.open('rb') as f:
                return f.read()
        except Exception as e:
            raise IOError(f"Failed to read model file: {e}")
    return None
