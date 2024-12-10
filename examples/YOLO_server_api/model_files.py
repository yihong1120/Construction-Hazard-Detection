from __future__ import annotations

import datetime
from pathlib import Path

import torch


async def update_model_file(model: str, model_file: Path) -> None:
    """
    Update the model file for a specified model.

    Args:
        model (str): The model key (e.g., 'yolo11n', 'yolo11s').
        model_file (Path): The path to the new `.pt` model file.
    """
    # Define valid models and their corresponding filenames
    valid_models = {
        'yolo11n': 'best_yolo11n.pt',
        'yolo11s': 'best_yolo11s.pt',
        'yolo11m': 'best_yolo11m.pt',
        'yolo11l': 'best_yolo11l.pt',
        'yolo11x': 'best_yolo11x.pt',
    }
    if model not in valid_models:
        raise ValueError(
            f"Invalid model key: {model}. "
            f"Must be one of {list(valid_models.keys())}.",
        )

    if not model_file.is_file() or model_file.suffix != '.pt':
        raise ValueError(
            f"Invalid file: {model_file}. Must be a valid `.pt` file.",
        )

    try:
        torch.jit.load(str(model_file))
    except Exception as e:
        raise ValueError(f"Invalid PyTorch model file: {e}")

    # Use a base directory and construct the destination path safely
    base_dir = Path('models/pt').resolve()
    destination_filename = valid_models[model]
    destination_path = base_dir / destination_filename

    # Ensure the destination path is within the base directory
    destination_path = destination_path.resolve()
    if not str(destination_path).startswith(str(base_dir)):
        raise ValueError('Attempted path traversal in destination path.')

    destination_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        # Move the model file to the destination path
        model_file.rename(destination_path)
    except Exception as e:
        raise OSError(f"Failed to update model file: {e}")


async def get_new_model_file(
    model: str, last_update_time: datetime.datetime,
) -> bytes | None:
    """
    Retrieve the new model file if updated since the provided time.

    Args:
        model (str): The model key (e.g., 'yolo11n', 'yolo11s').
        last_update_time (datetime.datetime): The last update time
            provided by the user.

    Returns:
        bytes | None: Model file content if updated, else None.
    """
    # Define valid models and their corresponding filenames
    valid_models = {
        'yolo11n': 'best_yolo11n.pt',
        'yolo11s': 'best_yolo11s.pt',
        'yolo11m': 'best_yolo11m.pt',
        'yolo11l': 'best_yolo11l.pt',
        'yolo11x': 'best_yolo11x.pt',
    }

    if model not in valid_models:
        raise ValueError(
            f"Invalid model key: {model}. "
            f"Must be one of {list(valid_models.keys())}.",
        )

    # Use a base directory and construct the destination path safely
    base_dir = Path('models/pt').resolve()
    destination_filename = valid_models[model]
    destination_path = base_dir / destination_filename

    # Ensure the destination path is within the base directory
    destination_path = destination_path.resolve()
    if not str(destination_path).startswith(str(base_dir)):
        raise ValueError('Attempted path traversal in destination path.')

    if not destination_path.is_file():
        return None

    file_mod_time = datetime.datetime.fromtimestamp(
        destination_path.stat().st_mtime,
    )
    if file_mod_time > last_update_time:
        try:
            with destination_path.open('rb') as f:
                return f.read()
        except Exception as e:
            raise OSError(f"Failed to read model file: {e}")
    return None
