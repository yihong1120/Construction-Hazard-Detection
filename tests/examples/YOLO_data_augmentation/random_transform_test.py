from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from examples.YOLO_data_augmentation import (
    data_augmentation_albumentations as subject,
)


class StubTransform:
    """Record a transform construction without running image operations."""

    def __init__(
        self,
        name: str,
        args: tuple[object, ...],
        kwargs: dict[str, object],
    ) -> None:
        self.name = name
        self.args = args
        self.kwargs = kwargs


class AlbumentationsStub:
    """Expose the flexible transform factory API used by the builder."""

    def __init__(self) -> None:
        self.created: list[StubTransform] = []

    def __getattr__(self, name: str) -> Any:
        def factory(*args: object, **kwargs: object) -> StubTransform:
            transform = StubTransform(name, args, kwargs)
            self.created.append(transform)
            return transform

        return factory


class LimitedAlbumentationsStub(AlbumentationsStub):
    """Expose only selected transforms to exercise compatibility fallbacks."""

    def __init__(self, available: set[str]) -> None:
        super().__init__()
        self.available = available

    def __getattr__(self, name: str) -> Any:
        if name not in self.available:
            raise AttributeError(name)
        return super().__getattr__(name)


class FlipFallbackAlbumentationsStub(AlbumentationsStub):
    """Model an older release that only exposes the combined flip transform."""

    def __getattr__(self, name: str) -> Any:
        if name in {'HorizontalFlip', 'VerticalFlip'}:
            raise AttributeError(name)
        return super().__getattr__(name)


@pytest.mark.parametrize(
    ('has_bboxes', 'bboxes'),
    [
        (False, None),
        (True, [[0.2, 0.2, 0.1, 0.1]]),
    ],
)
def test_random_transform_builds_a_bbox_aware_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    has_bboxes: bool,
    bboxes: list[list[float]] | None,
) -> None:
    """Every public pipeline branch builds compatible transform objects."""
    albumentations = AlbumentationsStub()
    monkeypatch.setattr(subject, 'A', albumentations)
    augmenter = subject.DataAugmentation(str(tmp_path))

    pipeline = augmenter.random_transform(
        has_bboxes=has_bboxes,
        image_shape=(480, 640),
        bboxes=bboxes,
    )

    assert pipeline.name == 'Compose'
    assert pipeline.kwargs['bbox_params'].name == 'BboxParams'
    chosen = pipeline.args[0]
    assert 4 <= len(chosen) <= 7
    assert any(
        item.name == 'RandomBrightnessContrast'
        for item in albumentations.created
    )
    assert any(
        item.name == 'RandomGridShuffle'
        for item in albumentations.created
    )


def test_optional_transforms_degrade_gracefully_on_older_versions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Optional Albumentations APIs have safe fallbacks or no-op behaviour."""
    albumentations = LimitedAlbumentationsStub({'RandomCrop'})
    monkeypatch.setattr(subject, 'A', albumentations)
    augmenter = subject.DataAugmentation(str(tmp_path))

    crop = augmenter.random_bbox_safe_crop_transform((480, 640))
    assert crop.name == 'RandomCrop'
    assert augmenter.at_least_one_bbox_crop_transform((480, 640)) is None
    assert augmenter.symmetry_transform() is None
    assert augmenter.safe_rotate_transform() is None
    assert augmenter.random_scale_transform() is None
    assert augmenter.pad_if_needed_transform() is None
    assert augmenter.normalize_bboxes_with_albumentations(
        np.zeros((2, 2, 3), dtype=np.uint8),
        [[0.5, 0.5, 0.2, 0.2]],
        [1],
    ) == ([[0.5, 0.5, 0.2, 0.2]], [1])

    bbox_safe = LimitedAlbumentationsStub({'BBoxSafeRandomCrop'})
    monkeypatch.setattr(subject, 'A', bbox_safe)
    assert augmenter.random_bbox_safe_crop_transform((480, 640)).name == (
        'BBoxSafeRandomCrop'
    )

    square_symmetry = LimitedAlbumentationsStub({'SquareSymmetry'})
    monkeypatch.setattr(subject, 'A', square_symmetry)
    assert augmenter.symmetry_transform().name == 'SquareSymmetry'

    def legacy_factory(*, format: str) -> str:
        return format

    assert subject.DataAugmentation._create_transform(
        legacy_factory,
        coord_format='yolo',
    ) == 'yolo'
    assert augmenter._choose_bbox_transforms([], []) == []
    assert not augmenter._bbox_stays_in_single_grid_cell(
        [0.5, 0.5, 0.0, 0.2],
        (2, 2),
    )


def test_reference_image_helpers_handle_empty_and_unreadable_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """FDA source selection skips absent images and reports unreadable files.

    """
    augmenter = subject.DataAugmentation(str(tmp_path))
    assert augmenter.get_fda_reference_images() == []

    image = tmp_path / 'images' / 'reference.jpg'
    image.parent.mkdir()
    image.write_bytes(b'not-an-image')
    monkeypatch.setattr(subject.cv2, 'imread', lambda _path: None)
    assert augmenter.get_fda_reference_images() == []
    with pytest.raises(ValueError, match='could not be loaded'):
        augmenter.get_random_target_image()

    rgb = np.zeros((3, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(subject.cv2, 'imread', lambda _path: rgb)
    monkeypatch.setattr(subject.cv2, 'cvtColor', lambda value, _code: value)
    assert augmenter.get_random_target_image() is rgb


def test_mask_dropout_uses_the_post_transform_pipeline(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Positive detections receive optional mask-dropout post-processing."""
    image = np.zeros((24, 32, 3), dtype=np.uint8)
    bboxes = [[0.5, 0.5, 0.2, 0.2]]
    labels = [1]
    primary = MagicMock(
        return_value={
            'image': image,
            'bboxes': bboxes,
            'class_labels': labels,
        },
    )
    post = MagicMock(
        return_value={
            'image': image,
            'bboxes': bboxes,
            'class_labels': labels,
        },
    )

    def transform_factory(**kwargs: object) -> StubTransform:
        return StubTransform('MaskDropout', (), kwargs)

    def compose_factory(*_args: object, **_kwargs: object) -> MagicMock:
        return post

    albumentations = type(
        'MaskAlbumentations',
        (),
        {
            'Compose': staticmethod(compose_factory),
            'MaskDropout': staticmethod(transform_factory),
            'BboxParams': staticmethod(transform_factory),
        },
    )()
    monkeypatch.setattr(subject, 'A', albumentations)
    augmenter = subject.DataAugmentation(str(tmp_path))

    with (
        patch.object(augmenter, 'random_transform', return_value=primary),
        patch.object(
            augmenter,
            'get_fda_reference_images',
            return_value=[image],
        ),
        patch.object(
            augmenter, 'generate_random_mask',
            return_value=image[:, :, 0],
        ),
        patch.object(
            augmenter,
            'normalize_bboxes_with_albumentations',
            side_effect=[(bboxes, labels), (bboxes, labels), (bboxes, labels)],
        ),
        patch.object(subject.random, 'random', return_value=0.0),
    ):
        result = augmenter.process_image(image, bboxes, labels)

    assert result['bboxes'] == bboxes
    assert primary.call_args.kwargs['fda_metadata'] == [image]
    post.assert_called_once()


def test_generate_mask_and_flip_fallbacks_cover_all_shape_variants(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Mask generation handles each supported shape and legacy flip APIs."""
    augmenter = subject.DataAugmentation(str(tmp_path))
    with patch.object(
        subject.random,
        'choice',
        side_effect=['circle', 'rect', 'ellipse'],
    ):
        mask = augmenter.generate_random_mask(
            (80, 100),
            min_objects=3,
            max_objects=3,
        )
    assert mask.shape == (80, 100)
    assert mask.any()

    albumentations = FlipFallbackAlbumentationsStub()
    monkeypatch.setattr(subject, 'A', albumentations)
    augmenter.random_transform(has_bboxes=False)
    assert sum(item.name == 'Flip' for item in albumentations.created) >= 2


def test_augment_image_skips_unreadable_or_invalid_outputs(
    tmp_path: Path,
) -> None:
    """Invalid inputs do not write partial images or invalid label records."""
    augmenter = subject.DataAugmentation(str(tmp_path), num_augmentations=1)
    image_path = tmp_path / 'source.jpg'
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    labels = [1]
    bboxes = [[0.5, 0.5, 0.2, 0.2]]

    with (
        patch.object(subject.cv2, 'imread', return_value=None),
        patch('builtins.print') as output,
    ):
        augmenter.augment_image(image_path)
    output.assert_called_once_with('Error processing image: None')

    with (
        patch.object(subject.cv2, 'imread', return_value=image),
        patch.object(subject.cv2, 'cvtColor', return_value=image),
        patch.object(
            augmenter, 'read_label_file',
            return_value=(labels, bboxes),
        ),
        patch.object(
            augmenter,
            'normalize_bboxes_with_albumentations',
            return_value=([], []),
        ),
        patch('builtins.print') as output,
    ):
        augmenter.augment_image(image_path)
    output.assert_called_once()

    with (
        patch.object(subject.cv2, 'imread', return_value=image),
        patch.object(subject.cv2, 'cvtColor', return_value=image),
        patch.object(
            augmenter, 'read_label_file',
            return_value=(labels, bboxes),
        ),
        patch.object(
            augmenter,
            'normalize_bboxes_with_albumentations',
            side_effect=[(bboxes, labels), ([], [])],
        ),
        patch.object(
            augmenter,
            'resize_image_and_bboxes',
            return_value=(image, bboxes),
        ),
        patch('builtins.print') as output,
    ):
        augmenter.augment_image(image_path)
    assert any(
        'Skipping augmentation' in str(call)
        for call in output.call_args_list
    )

    transformed = {
        'image': image.astype(np.float32),
        'bboxes': [],
        'class_labels': [],
    }
    with (
        patch.object(subject.cv2, 'imread', return_value=image),
        patch.object(subject.cv2, 'cvtColor', return_value=image),
        patch.object(
            augmenter, 'read_label_file',
            return_value=(labels, bboxes),
        ),
        patch.object(
            augmenter,
            'normalize_bboxes_with_albumentations',
            side_effect=[(bboxes, labels), (bboxes, labels), ([], [])],
        ),
        patch.object(
            augmenter,
            'resize_image_and_bboxes',
            return_value=(image, bboxes),
        ),
        patch.object(augmenter, 'process_image', return_value=transformed),
        patch.object(subject.cv2, 'imwrite') as write_image,
    ):
        augmenter.augment_image(image_path)
    write_image.assert_not_called()

    transformed = {
        'image': image.astype(np.float32),
        'bboxes': [],
        'class_labels': [],
    }
    with (
        patch.object(subject.cv2, 'imread', return_value=image),
        patch.object(subject.cv2, 'cvtColor', return_value=image),
        patch.object(augmenter, 'read_label_file', return_value=([], [])),
        patch.object(
            augmenter,
            'normalize_bboxes_with_albumentations',
            return_value=([], []),
        ),
        patch.object(
            augmenter,
            'resize_image_and_bboxes',
            return_value=(image, []),
        ),
        patch.object(augmenter, 'process_image', return_value=transformed),
        patch.object(subject.cv2, 'imwrite') as write_image,
        patch.object(augmenter, 'write_label_file') as write_labels,
    ):
        augmenter.augment_image(image_path)
    write_image.assert_called_once()
    write_labels.assert_called_once()
