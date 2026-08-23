from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np

from examples.YOLO_data_augmentation.data_augmentation_albumentations import A
from examples.YOLO_data_augmentation.data_augmentation_albumentations import (
    DataAugmentation,
)
from examples.YOLO_data_augmentation.data_augmentation_albumentations import (
    main,
)

_requires_albumentations = unittest.skipUnless(
    hasattr(A, 'Compose'),
    'Albumentations transforms are unavailable.',
)


@_requires_albumentations
class TestDataAugmentation(unittest.TestCase):
    """Unit tests for the DataAugmentation class."""

    def setUp(self) -> None:
        """Set up the test environment."""
        self.train_path = 'tests/cv_dataset'
        self.num_augmentations = 2
        self.augmenter = DataAugmentation(
            self.train_path,
            self.num_augmentations,
        )

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imread',
    )
    @patch('builtins.print')
    def test_augment_image_exception(
        self,
        mock_print: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        # Mock image reading exception
        """Test augment image exception.

        Args:
            mock_print: Value used by this callable.
            mock_imread: Value used by this callable.
        """
        mock_imread.side_effect = Exception('Mocked exception')

        # Test augment_image
        self.augmenter.augment_image(
            Path('tests/cv_dataset/images/mock_image.jpg'),
        )

        # Check if the print method was called with the correct output
        mock_print.assert_any_call(
            'Error processing image: '
            'tests/cv_dataset/images/mock_image.jpg: Mocked exception',
        )

    @patch('builtins.print')
    def test_augment_image_none(self, mock_print: MagicMock) -> None:
        # Test when image is None
        """Test augment image none.

        Args:
            mock_print: Value used by this callable.
        """
        self.augmenter.augment_image(None)
        mock_print.assert_any_call('Error processing image: None')

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imread',
    )
    @patch(
        'examples.YOLO_data_augmentation.'
        'data_augmentation_albumentations.cv2.cvtColor',
    )
    def test_augment_image(
        self,
        mock_cvtColor: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        # Mock image and label data
        """Test augment image.

        Args:
            mock_cvtColor: Value used by this callable.
            mock_imread: Value used by this callable.
        """
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_bboxes = [[0.5, 0.5, 0.2, 0.2]]
        mock_class_labels = [1]

        mock_imread.return_value = mock_image
        mock_cvtColor.return_value = mock_image

        with patch.object(
            self.augmenter,
            'read_label_file',
            return_value=(mock_class_labels, mock_bboxes),
        ):
            transformed = {
                'image': mock_image,
                'bboxes': mock_bboxes,
                'class_labels': mock_class_labels,
            }
            with patch.object(
                self.augmenter,
                'process_image',
                return_value=transformed,
            ):
                with patch.object(
                    self.augmenter,
                    'write_label_file',
                ) as mock_write_label_file:
                    with patch(
                        'examples.YOLO_data_augmentation.'
                        'data_augmentation_albumentations.cv2.imwrite',
                    ) as mock_imwrite:
                        self.augmenter.augment_image(Path('image.jpg'))
                        self.assertTrue(mock_imread.called)
                        self.assertTrue(mock_cvtColor.called)
                        self.assertTrue(mock_imwrite.called)
                        self.assertTrue(mock_write_label_file.called)

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imread',
    )
    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.cvtColor',
    )
    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imwrite',
    )
    def test_augment_image_with_alpha_channel(
        self,
        mock_imwrite: MagicMock,
        mock_cvtColor: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        # Mock image with alpha channel
        """Test augment image with alpha channel.

        Args:
            mock_imwrite: Value used by this callable.
            mock_cvtColor: Value used by this callable.
            mock_imread: Value used by this callable.
        """
        mock_image = np.random.randint(0, 255, (100, 100, 4), dtype=np.uint8)
        mock_imread.return_value = mock_image

        mock_bboxes = [[0.5, 0.5, 0.2, 0.2]]
        mock_class_labels = [1]

        with patch.object(
            self.augmenter,
            'read_label_file',
            return_value=(mock_class_labels, mock_bboxes),
        ):
            # Simulate cv2.cvtColor behaviour to remove alpha channel
            mock_cvtColor.side_effect = lambda img, code: (
                img[
                    :,
                    :,
                    :3,
                ]
                if img.shape[2] == 4
                else img
            )

            transformed = {
                'image': mock_image[:, :, :3],
                'bboxes': mock_bboxes,
                'class_labels': mock_class_labels,
            }
            with patch.object(
                self.augmenter,
                'process_image',
                return_value=transformed,
            ):
                self.augmenter.augment_image(Path('image_with_alpha.jpg'))
                self.assertTrue(mock_imread.called)
                self.assertTrue(mock_cvtColor.called)

                # Ensure the image has 3 channels
                # after removing the alpha channel
                processed_image = mock_cvtColor(mock_image, None)
                self.assertEqual(processed_image.shape[2], 3)

                self.assertTrue(mock_imwrite.called)

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imread',
    )
    @patch(
        'examples.YOLO_data_augmentation.'
        'data_augmentation_albumentations.cv2.cvtColor',
    )
    def test_augment_image_with_empty_labels(
        self,
        mock_cvtColor: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        """Test augment_image keeps intentionally object-free images."""
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_imread.return_value = mock_image
        mock_cvtColor.return_value = mock_image

        with patch.object(
            self.augmenter,
            'read_label_file',
            return_value=([], []),
        ):
            transformed = {
                'image': mock_image,
                'bboxes': [],
                'class_labels': [],
            }
            with patch.object(
                self.augmenter,
                'process_image',
                return_value=transformed,
            ):
                with patch.object(
                    self.augmenter,
                    'write_label_file',
                ) as mock_write_label_file:
                    with patch(
                        'examples.YOLO_data_augmentation.'
                        'data_augmentation_albumentations.cv2.imwrite',
                    ) as mock_imwrite:
                        self.augmenter.augment_image(Path('empty.jpg'))

        self.assertTrue(mock_imwrite.called)
        mock_write_label_file.assert_called()
        for call in mock_write_label_file.call_args_list:
            self.assertEqual(call.args[0], [])
            self.assertEqual(call.args[1], [])

    @_requires_albumentations
    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'A.Compose',
    )
    def test_resize_small_image(self, mock_compose: MagicMock) -> None:
        """Test resize_image_and_bboxes method with a small image."""
        mock_image = np.random.randint(0, 255, (20, 20, 3), dtype=np.uint8)
        mock_bboxes = [[0.5, 0.5, 0.2, 0.2]]
        class_labels = [1]
        image_path = Path('small_image.jpg')

        # Mock the transformation result
        mock_transformed = {
            'image': np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8),
            'bboxes': [[0.5, 0.5, 0.2, 0.2]],
        }
        mock_transform = MagicMock()
        mock_transform.return_value = mock_transformed
        mock_compose.return_value = mock_transform

        resized_image, resized_bboxes = self.augmenter.resize_image_and_bboxes(
            mock_image,
            mock_bboxes,
            class_labels,
            image_path,
        )

        # Verify that the Compose and BboxParams were called correctly
        mock_compose.assert_called()
        mock_transform.assert_called_once_with(
            image=mock_image,
            bboxes=mock_bboxes,
            class_labels=class_labels,
        )

        # Verify the transformation result
        self.assertEqual(resized_image.shape, (64, 64, 3))
        self.assertEqual(resized_bboxes, [[0.5, 0.5, 0.2, 0.2]])

    @_requires_albumentations
    def test_resize_large_image(self) -> None:
        """Test resize_image_and_bboxes method with a large image."""
        mock_image = np.random.randint(0, 255, (2000, 2000, 3), dtype=np.uint8)
        mock_bboxes = [[0.5, 0.5, 0.2, 0.2]]
        mock_class_labels = [1]
        image_path = Path('large_image.jpg')

        with patch('builtins.print') as mock_print:
            resized_image, resized_bboxes = (
                self.augmenter.resize_image_and_bboxes(
                    mock_image,
                    mock_bboxes,
                    mock_class_labels,
                    image_path,
                )
            )
            self.assertEqual(resized_image.shape, (1920, 1920, 3))
            mock_print.assert_called_with(
                f"Resize {image_path} due to large size: {mock_image.shape}",
            )

    def test_random_bbox_safe_crop_transform_uses_bbox_aware_crop(
        self,
    ) -> None:
        """Test random crop augmentation uses an Albumentations bbox
        transform."""
        mock_crop = MagicMock(return_value='bbox_safe_crop')
        with patch.object(
            A,
            'RandomSizedBBoxSafeCrop',
            mock_crop,
            create=True,
        ):
            with patch(
                'examples.YOLO_data_augmentation.'
                'data_augmentation_albumentations.random.randint',
                side_effect=[512, 640],
            ):
                transform = self.augmenter.random_bbox_safe_crop_transform(
                    (720, 1280),
                )

        self.assertEqual(transform, 'bbox_safe_crop')
        mock_crop.assert_called_once_with(height=512, width=640, p=1)

    def test_at_least_one_bbox_crop_transform_uses_fixed_transform(
        self,
    ) -> None:
        """Test AtLeastOneBBoxRandomCrop is used."""
        mock_crop = MagicMock(return_value='at_least_one_crop')
        with patch.object(
            A,
            'AtLeastOneBBoxRandomCrop',
            mock_crop,
            create=True,
        ):
            with patch(
                'examples.YOLO_data_augmentation.'
                'data_augmentation_albumentations.random.randint',
                side_effect=[512, 640],
            ):
                transform = self.augmenter.at_least_one_bbox_crop_transform(
                    (720, 1280),
                )

        self.assertEqual(transform, 'at_least_one_crop')
        mock_crop.assert_called_once_with(
            height=512,
            width=640,
            erosion_factor=0.2,
            p=1,
        )

    def test_bbox_crop_transform_clamps_to_image_size(self) -> None:
        """Test random bbox crop never exceeds the current image dimensions."""
        mock_crop = MagicMock(return_value='bbox_safe_crop')
        with patch.object(
            A,
            'RandomSizedBBoxSafeCrop',
            mock_crop,
            create=True,
        ):
            with patch(
                'examples.YOLO_data_augmentation.'
                'data_augmentation_albumentations.random.randint',
                side_effect=[528, 637],
            ):
                transform = self.augmenter.random_bbox_safe_crop_transform(
                    (480, 640),
                )

        self.assertEqual(transform, 'bbox_safe_crop')
        mock_crop.assert_called_once_with(height=480, width=637, p=1)

    def test_choose_bbox_transforms_keeps_crop_transform_single(self) -> None:
        """Test crop transforms are not combined with dimension-changing
        transforms."""
        with patch(
            'examples.YOLO_data_augmentation.'
            'data_augmentation_albumentations.random.random',
            return_value=0.0,
        ):
            with patch(
                'examples.YOLO_data_augmentation.'
                'data_augmentation_albumentations.random.choice',
                return_value='crop_b',
            ) as mock_choice:
                with patch(
                    'examples.YOLO_data_augmentation.'
                    'data_augmentation_albumentations.random.sample',
                ) as mock_sample:
                    transforms = self.augmenter._choose_bbox_transforms(
                        ['rotate', 'affine'],
                        ['crop_a', 'crop_b'],
                    )

        self.assertEqual(transforms, ['crop_b'])
        mock_choice.assert_called_once_with(['crop_a', 'crop_b'])
        mock_sample.assert_not_called()

    def test_choose_bbox_transforms_combines_non_crop_transforms(self) -> None:
        """Test non-crop transforms can still be randomly combined."""
        with patch(
            'examples.YOLO_data_augmentation.'
            'data_augmentation_albumentations.random.random',
            return_value=1.0,
        ):
            with patch(
                'examples.YOLO_data_augmentation.'
                'data_augmentation_albumentations.random.randint',
                return_value=2,
            ) as mock_randint:
                with patch(
                    'examples.YOLO_data_augmentation.'
                    'data_augmentation_albumentations.random.sample',
                    return_value=['rotate', 'affine'],
                ) as mock_sample:
                    transforms = self.augmenter._choose_bbox_transforms(
                        ['rotate', 'affine', 'perspective'],
                        ['crop'],
                    )

        self.assertEqual(transforms, ['rotate', 'affine'])
        mock_randint.assert_called_once_with(1, 2)
        mock_sample.assert_called_once_with(
            ['rotate', 'affine', 'perspective'],
            k=2,
        )

    def test_grid_shuffle_allows_boxes_inside_single_grid_cell(self) -> None:
        """Test RandomGridShuffle can be used when every bbox stays in one
        cell."""
        bboxes = [
            [0.16, 0.16, 0.20, 0.20],
            [0.50, 0.50, 0.20, 0.20],
        ]

        can_use = self.augmenter._can_use_random_grid_shuffle(
            bboxes,
            (3, 3),
        )

        self.assertTrue(can_use)

    def test_grid_shuffle_rejects_boxes_crossing_grid_cell(self) -> None:
        """Test RandomGridShuffle is skipped when any bbox crosses a grid
        cell."""
        bboxes = [
            [0.50, 0.50, 0.80, 0.20],
        ]

        can_use = self.augmenter._can_use_random_grid_shuffle(
            bboxes,
            (3, 3),
        )

        self.assertFalse(can_use)

    def test_grid_shuffle_allows_boxes_touching_grid_boundary(self) -> None:
        """Test a bbox ending exactly on a grid boundary is not treated as
        crossing."""
        bboxes = [
            [1 / 6, 1 / 6, 1 / 3, 1 / 3],
        ]

        can_use = self.augmenter._can_use_random_grid_shuffle(
            bboxes,
            (3, 3),
        )

        self.assertTrue(can_use)

    def test_safe_rotate_transform_uses_fixed_transform(self) -> None:
        """Test SafeRotate is created."""
        mock_rotate = MagicMock(return_value='safe_rotate')
        with patch.object(A, 'SafeRotate', mock_rotate, create=True):
            transform = self.augmenter.safe_rotate_transform()

        self.assertEqual(transform, 'safe_rotate')
        _, kwargs = mock_rotate.call_args
        self.assertEqual(kwargs['angle_range'], (-45, 45))
        self.assertEqual(kwargs['border_mode'], 4)
        self.assertEqual(kwargs['p'], 1)

    def test_symmetry_transform_uses_d4(self) -> None:
        """Test D4 square symmetry is used."""
        mock_d4 = MagicMock(return_value='d4')
        with patch.object(A, 'D4', mock_d4, create=True):
            transform = self.augmenter.symmetry_transform()

        self.assertEqual(transform, 'd4')
        mock_d4.assert_called_once_with(p=1)

    def test_random_scale_transform_uses_fixed_transform(self) -> None:
        """Test RandomScale is created."""
        mock_scale = MagicMock(return_value='random_scale')
        with patch.object(A, 'RandomScale', mock_scale, create=True):
            transform = self.augmenter.random_scale_transform()

        self.assertEqual(transform, 'random_scale')
        _, kwargs = mock_scale.call_args
        self.assertEqual(kwargs['scale_range'], (-0.25, 0.35))
        self.assertEqual(kwargs['p'], 1)

    def test_pad_if_needed_transform_uses_fixed_transform(self) -> None:
        """Test PadIfNeeded is created."""
        mock_pad = MagicMock(return_value='pad')
        with patch.object(A, 'PadIfNeeded', mock_pad, create=True):
            transform = self.augmenter.pad_if_needed_transform()

        self.assertEqual(transform, 'pad')
        _, kwargs = mock_pad.call_args
        self.assertEqual(kwargs['min_height'], 640)
        self.assertEqual(kwargs['min_width'], 640)
        self.assertEqual(kwargs['p'], 1)

    def test_create_bbox_params_filters_invalid_boxes(self) -> None:
        """Test bbox params request clipping and visibility filtering."""
        mock_params = MagicMock(return_value='bbox_params')
        with patch.object(A, 'BboxParams', mock_params, create=True):
            bbox_params = self.augmenter._create_bbox_params()

        self.assertEqual(bbox_params, 'bbox_params')
        _, kwargs = mock_params.call_args
        self.assertEqual(kwargs['format'], 'yolo')
        self.assertTrue(kwargs['clip'])
        self.assertTrue(kwargs['filter_invalid_bboxes'])
        self.assertEqual(
            kwargs['min_visibility'],
            self.augmenter.min_bbox_visibility,
        )

    def test_random_resized_crop_transform_uses_fixed_size_argument(
        self,
    ) -> None:
        """Test RandomResizedCrop is created with the fixed API."""

        def fake_crop(
            *,
            size: tuple[int, int],
            scale: tuple[float, float],
            p: int,
        ) -> tuple:
            """Perform fake crop.

            Args:
                size: Value used by this callable.
                scale: Value used by this callable.
                p: Value used by this callable.

            Returns:
                The callable result.
            """
            return size, scale, p

        with patch.object(A, 'RandomResizedCrop', fake_crop, create=True):
            transform = self.augmenter.random_resized_crop_transform()

        self.assertEqual(transform, ((640, 640), (0.3, 1.0), 1))

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imread',
    )
    @patch(
        'examples.YOLO_data_augmentation.'
        'data_augmentation_albumentations.cv2.cvtColor',
    )
    def test_get_fda_reference_images_resizes_to_target_shape(
        self,
        mock_cvtColor: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        """Test FDA references are resized to match the augmented input
        image."""
        mock_reference = np.random.randint(
            0,
            255,
            (722, 640, 3),
            dtype=np.uint8,
        )
        mock_imread.return_value = mock_reference
        mock_cvtColor.return_value = mock_reference

        with patch('pathlib.Path.glob', return_value=[Path('ref.jpg')]):
            refs = self.augmenter.get_fda_reference_images(
                count=1,
                target_shape=(640, 640),
            )

        self.assertEqual(len(refs), 1)
        self.assertEqual(refs[0].shape, (640, 640, 3))

    def test_normalize_bboxes_uses_albumentations_bbox_params(self) -> None:
        """Test bbox normalization is delegated to Albumentations."""
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_transform = MagicMock(
            return_value={
                'image': mock_image,
                'bboxes': [[0.5, 0.5, 0.2, 0.2]],
                'class_labels': [1],
            },
        )
        mock_noop = MagicMock(return_value='noop')
        with patch.object(A, 'BboxParams', MagicMock(), create=True):
            with patch.object(A, 'NoOp', mock_noop, create=True):
                with patch.object(
                    A,
                    'Compose',
                    return_value=mock_transform,
                    create=True,
                ) as compose:
                    bboxes, class_labels = (
                        self.augmenter.normalize_bboxes_with_albumentations(
                            mock_image,
                            [[0.5, 0.5, 0.2, 0.2]],
                            [1],
                        )
                    )

        mock_noop.assert_called_once_with(p=1)
        compose.assert_called_once()
        self.assertEqual(compose.call_args.args[0], ['noop'])
        mock_transform.assert_called_once_with(
            image=mock_image,
            bboxes=[[0.5, 0.5, 0.2, 0.2]],
            class_labels=[1],
        )
        self.assertEqual(bboxes, [[0.5, 0.5, 0.2, 0.2]])
        self.assertEqual(class_labels, [1])

    def test_process_image_marks_positive_images_as_having_bboxes(
        self,
    ) -> None:
        """Test object images pass the bbox-presence flag to augmentation
        choice."""
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_transform = MagicMock(
            return_value={
                'image': mock_image,
                'bboxes': [[0.5, 0.5, 0.2, 0.2]],
                'class_labels': [1],
            },
        )
        with patch.object(
            self.augmenter,
            'normalize_bboxes_with_albumentations',
            side_effect=lambda image, bboxes, labels: (bboxes, labels),
        ):
            with patch.object(
                self.augmenter,
                'random_transform',
                return_value=mock_transform,
            ) as mock_random_transform:
                self.augmenter.process_image(
                    mock_image,
                    [[0.5, 0.5, 0.2, 0.2]],
                    [1],
                )

        mock_random_transform.assert_called_once_with(
            has_bboxes=True,
            image_shape=mock_image.shape[:2],
            bboxes=[[0.5, 0.5, 0.2, 0.2]],
        )

    def test_process_image_marks_empty_images_as_without_bboxes(self) -> None:
        """Test empty-label images pass the no-bbox flag to augmentation
        choice."""
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_transform = MagicMock(
            return_value={
                'image': mock_image,
                'bboxes': [],
                'class_labels': [],
            },
        )
        with patch.object(
            self.augmenter,
            'normalize_bboxes_with_albumentations',
            side_effect=lambda image, bboxes, labels: (bboxes, labels),
        ):
            with patch.object(
                self.augmenter,
                'random_transform',
                return_value=mock_transform,
            ) as mock_random_transform:
                self.augmenter.process_image(mock_image, [], [])

        mock_random_transform.assert_called_once_with(
            has_bboxes=False,
            image_shape=mock_image.shape[:2],
            bboxes=[],
        )

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'ProcessPoolExecutor',
    )
    def test_augment_data(self, mock_executor: MagicMock) -> None:
        """Test augment data.

        Args:
            mock_executor: Value used by this callable.
        """
        mock_executor.return_value.__enter__.return_value.map = MagicMock()
        self.augmenter.augment_data(batch_size=2)
        self.assertTrue(mock_executor.called)

    def test_read_label_file(self) -> None:
        """Test read_label_file method."""
        label_content = '0 0.5 0.5 0.2 0.2\n'
        label_path = Path('label.txt')
        with patch(
            'builtins.open',
            unittest.mock.mock_open(read_data=label_content),
        ):
            with patch('pathlib.Path.exists', return_value=True):
                class_labels, bboxes = self.augmenter.read_label_file(
                    label_path,
                )
            self.assertEqual(class_labels, [0])
            self.assertEqual(bboxes, [[0.5, 0.5, 0.2, 0.2]])

    def test_read_empty_label_file(self) -> None:
        """Test an empty label file is treated as an object-free image."""
        label_path = Path('empty_label.txt')
        with patch('pathlib.Path.exists', return_value=True):
            with patch(
                'builtins.open', unittest.mock.mock_open(read_data='\n'),
            ):
                class_labels, bboxes = self.augmenter.read_label_file(
                    label_path,
                )

        self.assertEqual(class_labels, [])
        self.assertEqual(bboxes, [])

    def test_read_missing_label_file(self) -> None:
        """Test a missing label file is treated as an object-free image."""
        class_labels, bboxes = self.augmenter.read_label_file(
            Path('missing_label.txt'),
        )

        self.assertEqual(class_labels, [])
        self.assertEqual(bboxes, [])

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'cv2.imread',
    )
    @patch(
        'examples.YOLO_data_augmentation.'
        'data_augmentation_albumentations.cv2.cvtColor',
    )
    def test_augment_image_writes_transformed_bboxes_without_quality_retry(
        self,
        mock_cvtColor: MagicMock,
        mock_imread: MagicMock,
    ) -> None:
        """Test augmentation writes the bbox returned by Albumentations."""
        mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        mock_bboxes = [[0.5, 0.5, 0.2, 0.2]]
        mock_class_labels = [1]
        mock_imread.return_value = mock_image
        mock_cvtColor.return_value = mock_image

        transformed = {
            'image': mock_image,
            'bboxes': [[0.5, 0.5, 0.9, 1.0]],
            'class_labels': mock_class_labels,
        }

        with patch.object(
            self.augmenter,
            'read_label_file',
            return_value=(mock_class_labels, mock_bboxes),
        ):
            with patch.object(
                self.augmenter,
                'process_image',
                return_value=transformed,
            ) as mock_process:
                with patch.object(
                    self.augmenter,
                    'write_label_file',
                ) as mock_write_label_file:
                    with patch(
                        'examples.YOLO_data_augmentation.'
                        'data_augmentation_albumentations.cv2.imwrite',
                    ) as mock_imwrite:
                        self.augmenter.augment_image(Path('image.jpg'))

        self.assertEqual(mock_process.call_count, self.num_augmentations)
        self.assertEqual(mock_imwrite.call_count, self.num_augmentations)
        for call in mock_write_label_file.call_args_list:
            np.testing.assert_allclose(
                call.args[0],
                [[0.5, 0.5, 0.9, 1.0]],
            )

    def test_write_label_file(self) -> None:
        """Test write_label_file method."""
        bboxes_aug = [(0.5, 0.5, 0.2, 0.2)]
        class_labels_aug = [0]
        label_path = Path('label_aug.txt')
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            self.augmenter.write_label_file(
                bboxes_aug,
                class_labels_aug,
                label_path,
            )
            mock_file().write.assert_called_with('0 0.5 0.5 0.2 0.2\n')

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'random.shuffle',
    )
    def test_shuffle_data(self, mock_shuffle: MagicMock) -> None:
        """Test shuffle data.

        Args:
            mock_shuffle: Value used by this callable.
        """
        image_dir = Path(self.train_path) / 'images'
        label_dir = Path(self.train_path) / 'labels'
        image_paths = [image_dir / f"image_{i}.jpg" for i in range(5)]
        label_paths = [label_dir / f"label_{i}.txt" for i in range(5)]

        with patch.object(
            Path,
            'glob',
            side_effect=[image_paths, label_paths],
        ):
            with patch.object(Path, 'rename') as mock_rename:
                self.augmenter.shuffle_data()
                self.assertTrue(mock_shuffle.called)
                self.assertTrue(mock_rename.called)

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'DataAugmentation',
    )
    @patch('argparse.ArgumentParser.parse_args')
    def test_main(
        self,
        mock_parse_args: MagicMock,
        MockDataAugmentation: MagicMock,
    ) -> None:
        # Mock command line arguments
        """Test main.

        Args:
            mock_parse_args: Value used by this callable.
            MockDataAugmentation: Value used by this callable.
        """
        mock_parse_args.return_value = argparse.Namespace(
            train_path='./dataset_aug/train',
            num_augmentations=10,
            batch_size=5,
        )

        # Mock DataAugmentation class
        mock_augmenter = MockDataAugmentation.return_value
        mock_augmenter.augment_data = MagicMock()
        mock_augmenter.shuffle_data = MagicMock()

        # Execute main function
        main()

        # Verify DataAugmentation class was correctly initialised
        MockDataAugmentation.assert_called_once_with('./dataset_aug/train', 10)

        # Verify augment_data and shuffle_data methods were called
        mock_augmenter.augment_data.assert_called_once_with(batch_size=5)
        mock_augmenter.shuffle_data.assert_called_once()

    @patch(
        'examples.YOLO_data_augmentation.data_augmentation_albumentations.'
        'DataAugmentation',
    )
    @patch('argparse.ArgumentParser.parse_args')
    def test_main_exception(
        self,
        mock_parse_args: MagicMock,
        MockDataAugmentation: MagicMock,
    ) -> None:
        # Mock command line arguments
        """Test main exception.

        Args:
            mock_parse_args: Value used by this callable.
            MockDataAugmentation: Value used by this callable.
        """
        mock_parse_args.return_value = argparse.Namespace(
            train_path='./dataset_aug/train',
            num_augmentations=10,
            batch_size=5,
        )

        # Mock DataAugmentation class to raise an exception
        mock_augmenter = MockDataAugmentation.return_value
        mock_augmenter.augment_data.side_effect = Exception('Test exception')

        with patch('builtins.print') as mock_print:
            # Execute main function
            main()

            # Verify print was called with the correct error message
            mock_print.assert_called_with('Error: Test exception')


if __name__ == '__main__':
    unittest.main()
