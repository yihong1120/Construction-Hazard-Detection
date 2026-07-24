from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from examples.YOLO_train import train


def _handler() -> train.YOLOModelHandler:
    """Create a handler without constructing a real Ultralytics model."""
    handler = train.YOLOModelHandler.__new__(train.YOLOModelHandler)
    handler.model = MagicMock()
    handler.model_name = 'models.pt'
    handler.batch_size = 1
    return handler


def test_train_on_all_data_requires_model_and_images_directory(
    tmp_path: Path,
) -> None:
    """All-data training stops safely without a model or source images."""
    handler = _handler()
    handler.model = None
    with pytest.raises(RuntimeError, match='not loaded'):
        handler.train_on_all_data(str(tmp_path / 'data.yaml'), 1, 'auto')

    config = tmp_path / 'data.yaml'
    config.write_text('names: [worker]\n')
    handler = _handler()
    with patch('builtins.print') as output:
        handler.train_on_all_data(str(config), 1, 'auto')

    assert 'images' in output.call_args.args[0]


def test_train_on_all_data_writes_filtered_temporary_config(
    tmp_path: Path,
) -> None:
    """All-data training preserves classes while replacing dataset splits."""
    images = tmp_path / 'images'
    images.mkdir()
    config = tmp_path / 'data.yaml'
    config.write_text(
        'path: old-root\ntrain: old-train\nval: old-val\ntest: old-test\n'
        'names: [worker, cone]\nnc: 2\n',
    )
    handler = _handler()
    handler.load_model = MagicMock()
    observed: list[str] = []

    def train_model(data_config: str, **_kwargs: object) -> None:
        observed.append(Path(data_config).read_text())

    handler.train_model = MagicMock(side_effect=train_model)

    handler.train_on_all_data(str(config), 3, 'SGD')

    handler.load_model.assert_called_once()
    handler.train_model.assert_called_once_with(
        data_config=str(tmp_path / 'all_data_train.yaml'),
        epochs=3,
        optimizer='SGD',
    )
    assert observed == [
        f'path: {tmp_path}\ntrain: images\nval: images\n'
        'names: [worker, cone]\nnc: 2',
    ]
    assert not (tmp_path / 'all_data_train.yaml').exists()


def test_main_supports_standalone_full_data_training() -> None:
    """The CLI invokes all-data training when the flag is selected."""
    args = argparse.Namespace(
        data_config='dataset/data.yaml',
        epochs=8,
        model_name='models.pt',
        export_format='onnx',
        onnx_path='model.onnx',
        pt_path='model.pt',
        batch_size=2,
        optimizer='AdamW',
        cross_validate=False,
        n_splits=5,
        full_data_training=True,
    )
    handler = MagicMock()

    with (
        patch.object(
            train.argparse.ArgumentParser,
            'parse_args',
            return_value=args,
        ),
        patch.object(train, 'YOLOModelHandler', return_value=handler),
    ):
        train.main()

    handler.train_on_all_data.assert_called_once_with(
        data_config='dataset/data.yaml',
        epochs=8,
        optimizer='AdamW',
    )
    handler.train_model.assert_not_called()
    handler.save_model.assert_called_once_with('model.pt')
