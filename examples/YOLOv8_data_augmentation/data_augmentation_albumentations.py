from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor

import os
import time
import cv2
import argparse
import gc
import random
import uuid
from pathlib import Path

import albumentations as A
import numpy as np
from tqdm import tqdm


class DataAugmentation:
    """
    A class to perform data augmentation for image datasets, especially useful
    for training machine learning models.
    """

    def __init__(self, train_path: str, num_augmentations: int = 1):
        """
        Initialise the DataAugmentation class.

        Args:
            train_path (str): The path to the training data.
            num_augmentations (int): Number of augmentations per image.
        """
        self.train_path = Path(train_path)
        self.num_augmentations = num_augmentations

    def resize_image_and_bboxes(self, image, bboxes, class_labels, image_path) -> tuple[np.ndarray, list]:  
        """
        Resize images and bounding boxes if they are too small or too large.

        Args:
            image (np.ndarray): The input image.
            bboxes (list): The bounding boxes.
            class_labels (list[int]): The class labels.
            image_path (Path): The path to the image file.

        Returns:
            tuple[np.ndarray, list]: The resized image and bounding boxes.
        """
        # Check and resize small images
        if image.shape[0] < 32 or image.shape[1] < 32:
            print(f"Resize {image_path} due to small size: {image.shape}")
            transform = A.Compose([
                A.SmallestMaxSize(max_size=64, interpolation=cv2.INTER_LINEAR),
                A.LongestMaxSize(max_size=64, interpolation=cv2.INTER_LINEAR)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True))
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image, bboxes = transformed['image'], transformed['bboxes']

        # Check and resize large images
        if image.shape[0] > 1920 or image.shape[1] > 1920:
            print(f"Resize {image_path} due to large size: {image.shape}")
            transform = A.Compose([
                A.LongestMaxSize(max_size=1920, interpolation=cv2.INTER_LINEAR),
                A.SmallestMaxSize(max_size=1920, interpolation=cv2.INTER_LINEAR)
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True))
            transformed = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image, bboxes = transformed['image'], transformed['bboxes']

        return image, bboxes

    def random_crop_with_random_size(self, image, **kwargs) -> np.ndarray:
        """
        Custom function to randomly crop the image at a random position with a random size.

        Args:
            image (np.ndarray): Input image.

        Returns:
            np.ndarray: Cropped image.
        """
        height, width = image.shape[:2]
        
        # Randomly generate the height and width of the crop box
        crop_height = random.randint(400, 800)
        crop_width = random.randint(400, 800)
        
        # Ensure the crop box does not exceed the original image size
        max_x = max(0, width - crop_width)
        max_y = max(0, height - crop_height)
        
        # Randomly generate the top-left coordinates of the crop box
        start_x = random.randint(0, max_x)
        start_y = random.randint(0, max_y)
        
        # Crop the image
        cropped_image = image[start_y:start_y + crop_height, start_x:start_x + crop_width]
        return cropped_image

    def random_transform(self) -> A.Compose:
        """
        Generate a random augmentation pipeline.

        Returns:
            A.Compose: The augmentation pipeline.
        """
        # Augmentations that affect bounding boxes
        bbox_augmentations = [
            A.HorizontalFlip(p=1),
            A.VerticalFlip(p=1),
            A.RandomRotate90(p=1),
            A.Rotate(limit=(-45, 45), p=1),
            A.Affine(
                scale=(0.9, 1.1),  # Scaling range
                translate_percent=(0.05, 0.1),  # Translation percentage range
                shear=(-15, 15),  # Shear angle range
                rotate=45,  # Rotation angle range
                p=1  # Probability
            ),
            A.Transpose(p=1),
            A.Perspective(scale=(0.05, 0.1), p=1),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=None, p=1),
            A.Lambda(image=self.random_crop_with_random_size, p=1),
            # A.RandomResizedCrop(height=640, width=640, scale=(0.3, 1.0), p=1),
            A.RandomResizedCrop(size=(640, 640), scale=(0.3, 1.0), p=1),
        ]

        # Augmentations that do not affect bounding boxes
        non_bbox_augmentations = [
            # Colour and brightness adjustments
            A.RandomBrightnessContrast(
                brightness_limit=(-0.03, 0.03),
                contrast_limit=(-0.03, 0.03),
                p=1.0
            ),
            A.RGBShift(r_shift_limit=3, g_shift_limit=3, b_shift_limit=3, p=1.0),
            A.HueSaturationValue(hue_shift_limit=3, sat_shift_limit=5, val_shift_limit=3, p=1.0),
            A.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02, p=1.0),
            A.PlanckianJitter(p=1),
            # Blur and noise
            A.MotionBlur(blur_limit=(3, 7), p=1.0),
            A.GaussianBlur(blur_limit=(3, 7), p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
            # Colour space and contrast adjustments
            A.CLAHE(clip_limit=2, p=1.0),
            A.Sharpen(alpha=(0.1, 0.3), lightness=(0.7, 1.0), p=1.0),
            A.CoarseDropout(num_holes_range=(1, 6), hole_height_range=(0, 6), hole_width_range=(0, 6), p=1.0),
            A.ToGray(p=1),
            A.Equalize(p=1),
            A.Posterize(num_bits=4, p=1.0),
            A.InvertImg(p=1),
            # Special effects
            A.RandomShadow(shadow_roi=(0.3, 0.3, 0.7, 0.7), num_shadows_limit = (1, 2), shadow_dimension=2, p=1.0),
            # A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.1, p=1.0),
            A.RandomSnow(snow_point_range=(0.05, 0.15), brightness_coeff=1.2, p=1.0),
            A.RandomRain(slant_range=(-5,5), drop_length=5, drop_width=1, blur_value=2, p=1.0),
            A.ChannelShuffle(p=1),
            A.RandomSunFlare(flare_roi=(0.02, 0.04, 0.08, 0.08), angle_range=(0, 1), p=1),
            A.Defocus(radius=(3, 5), p=1),
            # Other augmentations
            A.RandomToneCurve(scale=0.05, p=1.0),
            A.RandomGamma(gamma_limit=(90, 110), p=1.0),
            A.Superpixels(p_replace=0.1, n_segments=100, p=1),
            A.ImageCompression(quality_range = (50, 90), p=1.0),
            A.GlassBlur(sigma=0.5, max_delta=1, p=1.0),
            A.PixelDropout(dropout_prob=0.05, p=1),
            A.OpticalDistortion(distort_limit=(0.05, 0.1), shift_limit=(0.05, 0.1), p=1.0)
        ]

        # Randomly select 1 to 2 augmentations that affect bounding boxes
        num_bbox_transforms = random.randint(1, 2)
        chosen_bbox_transforms = random.sample(bbox_augmentations, k=num_bbox_transforms)

        # Randomly select 2 to 3 augmentations that do not affect bounding boxes
        num_non_bbox_transforms = random.randint(2, 3)
        chosen_non_bbox_transforms = random.sample(non_bbox_augmentations, k=num_non_bbox_transforms)

        chosen_transforms = chosen_bbox_transforms + chosen_non_bbox_transforms

        # Return the augmentation pipeline
        return A.Compose(chosen_transforms + [
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0)
        ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels'], clip=True))

    def process_image(self, image: np.ndarray, bboxes: list, class_labels: list[int]) -> dict:
        """
        Apply augmentations to the image and bounding boxes.

        Args:
            image (np.ndarray): The input image.
            bboxes (list): The bounding boxes.
            class_labels (list[int]): The class labels.

        Returns:
            dict: The transformed image and bounding boxes.
        """
        # Generate a random augmentation pipeline
        aug_transform = self.random_transform()
        
        # Apply the augmentation pipeline to the image and bounding boxes
        transformed = aug_transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        return transformed

    def augment_image(self, image_path: Path) -> None:
        """
        Processes and augments a single image.

        Args:
            image_path (Path): The path to the image file.
        """
        image = None
        bboxes = None
        try:
            # Read the image using OpenCV
            image = cv2.imread(str(image_path))

            if image is None:
                print('Error processing image: None')
                return

            # Remove the alpha channel if the image has 4 channels
            if image.shape[2] == 4:
                image = image[ :, :, :3]

            # Convert the BGR image to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = np.clip(image, 0, 255).astype(np.uint8)

            # Read the label file
            label_path = self.train_path / 'labels' / image_path.with_suffix('.txt').name
            class_labels, bboxes = self.read_label_file(label_path)

            # Resize the image and bounding boxes
            image, bboxes = self.resize_image_and_bboxes(image, bboxes, class_labels, image_path)

            # Ensure the coordinates are between 0 and 1
            bboxes = np.clip(bboxes, 0, 1)

            for i in range(self.num_augmentations):
                # Apply augmentations to the image and bounding boxes
                transformed = self.process_image(image=image, bboxes=bboxes, class_labels=class_labels)
                image_aug, bboxes_aug, class_labels_aug = transformed['image'], transformed['bboxes'], transformed['class_labels']

                # Ensure the coordinates are between 0 and 1
                bboxes_aug = np.clip(bboxes_aug, 0, 1)

                # Ensure the image data type is uint8
                image_aug = np.clip(image_aug * 255, 0, 255).astype(np.uint8)

                # Save the augmented image and label file
                aug_image_filename = f"{image_path.stem}_aug_{i}{image_path.suffix}"
                aug_label_filename = f"{image_path.stem}_aug_{i}.txt"

                image_aug_path = self.train_path / 'images' / aug_image_filename
                label_aug_path = self.train_path / 'labels' / aug_label_filename

                # Save the image using OpenCV
                cv2.imwrite(str(image_aug_path), cv2.cvtColor(image_aug * 255, cv2.COLOR_RGB2BGR))
                self.write_label_file(bboxes_aug, class_labels_aug, str(label_aug_path))

        except Exception as e:
            print(f"Error processing image: {image_path}: {e}")
        finally:
            if image is not None:
                del image
            if bboxes is not None:
                del bboxes
            gc.collect()  # 強制垃圾回收

    def augment_data(self, batch_size=10) -> None:
        """
        Processes images in parallel to save time.

        Args:
            batch_size (int): The number of images to process in each batch.
        """
        image_paths = list(self.train_path.glob('images/*.jpg'))
        num_workers = min(batch_size, os.cpu_count() - 1)

        print(f"Using {num_workers} parallel workers for data augmentation.")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            list(tqdm(executor.map(self.augment_image, image_paths), total=len(image_paths)))

    @staticmethod
    def read_label_file(label_path: Path) -> tuple[list[list[float]], list[int]]:
        """
        Reads a label file and converts it into a list of bounding boxes and class labels.

        Args:
            label_path (Path): The path to the label file.

        Returns:
            tuple[list[list[float]], list[int]]: The list of bounding boxes and class labels.
        """
        annotations = []
        with open(label_path, 'r') as f:
            for line in f:
                class_id, x_center, y_center, width, height = map(float, line.strip().split())
                annotations.append([class_id, x_center, y_center, width, height])


        # Prepare the bounding boxes and class labels
        bboxes = [ann[1:] for ann in annotations]
        class_labels = [ann[0] for ann in annotations]

        return class_labels, bboxes
        
    @staticmethod
    def write_label_file(bboxes_aug: list[tuple], class_labels_aug: list[int], label_path: Path) -> None:
        """
        Writes bounding boxes and class labels to a label file.

        Args:
            bboxes (list[tuple]): The bounding boxes to write.
            class_labels (list[int]): The class labels to write.
            label_path (Path): The path to the label file.
        """
        # Combine class labels with the transformed bounding boxes
        annotations = [[class_labels_aug[i]] + list(bboxes_aug[i]) for i in range(len(bboxes_aug))]

        with open(label_path, 'w') as f:
            for ann in annotations:
                class_id, x_center, y_center, width, height = ann
                f.write(f'{int(class_id)} {x_center:.6} {y_center:.6} {width:.6} {height:.6}\n')

    def shuffle_data(self) -> None:
        """
        Shuffles the augmented dataset to ensure randomness.

        This method pairs each image file with its corresponding label file,
        assigns a unique UUID to each pair, and then saves them with the new
        names into the original directories.
        """
        image_dir = self.train_path / 'images'
        label_dir = self.train_path / 'labels'

        # Retrieve paths for all image and label files
        image_paths = list(image_dir.glob('*'))
        label_paths = list(label_dir.glob('*'))

        # Ensure the count of images and labels matches
        assert len(image_paths) == len(label_paths), 'The counts of image and label files do not match!'

        # Sort paths to ensure matching between images and labels
        image_paths.sort()
        label_paths.sort()

        # Shuffle image and label paths together to maintain correspondence
        combined = list(zip(image_paths, label_paths))
        random.shuffle(combined)

        # Rename files with a new UUID
        for image_path, label_path in combined:
            unique_id = str(uuid.uuid4())
            new_image_name = unique_id + image_path.suffix
            new_label_name = unique_id + label_path.suffix

            new_image_path = image_dir / new_image_name
            new_label_path = label_dir / new_label_name

            image_path.rename(new_image_path)
            label_path.rename(new_label_path)


def main():
    """
    Main function to perform data augmentation on image datasets.
    """
    try:
        parser = argparse.ArgumentParser(
            description='Perform data augmentation on image datasets.',
        )
        parser.add_argument(
            '--train_path',
            type=str,
            default='./dataset_aug/train',
            help='Path to the training data directory.',
        )
        parser.add_argument(
            '--num_augmentations',
            type=int,
            default=10,
            help='Number of augmentations per image.',
        )
        parser.add_argument(
            '--batch_size',
            type=int,
            default=5,
            help='Number of images to process in each batch.',
        )
        args = parser.parse_args()

        augmenter = DataAugmentation(args.train_path, args.num_augmentations)
        augmenter.augment_data(batch_size=args.batch_size)

        print('Pausing for 5 seconds before shuffling data...')
        time.sleep(5)

        augmenter.shuffle_data()
        print('Data augmentation and shuffling complete.')

    except Exception as e:
        print(f"Error: {e}")


if __name__ == '__main__':
    main()
