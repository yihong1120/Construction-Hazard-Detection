from typing import List, Tuple
import imageio.v3 as imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import argparse
import random
from pathlib import Path
from tqdm import tqdm
import time
import uuid
import gc

class DataAugmentation:
    """ 
    A class to perform data augmentation for image datasets, especially useful for training machine learning models.
    
    Attributes:
        train_path (Path): The path to the training data.
        num_augmentations (int): The number of augmentations to perform per image.
        seq (iaa.Sequential): The sequence of augmentations to apply.
    """
    
    def __init__(self, train_path: str, num_augmentations: int = 1):
        """
        Initialise the DataAugmentation class.

        Args:
            train_path (str): The path to the training data.
            num_augmentations (int): The number of augmentations to perform per image.
        """
        self.train_path = Path(train_path)
        self.num_augmentations = num_augmentations
        self.seq = self._get_augmentation_sequence()

    def _get_augmentation_sequence(self) -> iaa.Sequential:
        """
        Define a sequence of augmentations with different probabilities for each augmentation.
    
        Returns:
            iaa.Sequential: The sequence of augmentations to apply.
        """
        augmentations = [
            iaa.Sometimes(0.5, iaa.Flipud()),  # 50% probability to flip upside down
            iaa.Sometimes(0.5, iaa.Fliplr()),  # 50% probability to flip left to right
            iaa.Sometimes(0.6, iaa.Affine(rotate=(-45, 45))),  # 50% probability to rotate
            iaa.Sometimes(0.6, iaa.Resize((0.7, 1.3))),  # 50% probability to resize
            iaa.Sometimes(0.4, iaa.Multiply((0.8, 1.2))),  # 30% probability to change brightness
            iaa.Sometimes(0.4, iaa.LinearContrast((0.8, 1.2))),  # 30% probability to change contrast
            iaa.Sometimes(0.2, iaa.GaussianBlur(sigma=(0, 0.5))),  # 20% probability to blur
            iaa.Sometimes(0.4, iaa.Crop(percent=(0, 0.5))),  # 40% probability to crop
            iaa.Sometimes(0.2, iaa.SaltAndPepper(0.02)),  # 10% probability for salt and pepper noise
            iaa.Sometimes(0.2, iaa.ElasticTransformation(alpha=(0, 30), sigma=10)),  # 20% probability for elastic transformation
            iaa.Sometimes(0.1, iaa.MotionBlur(k=15, angle=[-45, 45])),  # 10% probability to add motion blur to simulate water flow
            iaa.Sometimes(0.6, iaa.ShearX((-40, 40))),  # 20% probability to shear on X axis
            iaa.Sometimes(0.6, iaa.ShearY((-40, 40))),  # 20% probability to shear on Y axis
            iaa.Sometimes(0.4, iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2))),  # 20% probability to sharpen
            iaa.Sometimes(0.2, iaa.PiecewiseAffine(scale=(0.01, 0.03))),  # 10% probability for piecewise affine
            iaa.Sometimes(0.3, iaa.Grayscale(alpha=(0.0, 1.0))),  # 10% probability to grayscale
            iaa.Sometimes(0.3, iaa.AddToHueAndSaturation((-30, 30))),  # 20% probability to change hue and saturation
            iaa.Sometimes(0.3, iaa.GammaContrast((0.5, 1.5))),  # 20% probability to change gamma contrast
            iaa.Sometimes(0.3, iaa.ChangeColorTemperature((3300, 6500))),  # 20% probability to change color temperature
            iaa.Sometimes(0.3, iaa.PerspectiveTransform(scale=(0.01, 0.1))),  # 10% probability for perspective transform
            iaa.Sometimes(0.3, iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25))),  # 10% probability for coarse dropout
            iaa.Sometimes(0.1, iaa.Invert(0.3)),  # 10% probability to invert colors
            iaa.Sometimes(0.4, iaa.imgcorruptlike.Spatter(severity=1)),  # 40% probability to add watermarks
            # iaa.Sometimes(0.1, iaa.Fog()),  # 10% probability to add fog
            # iaa.Sometimes(0.1, iaa.Rain(speed=(0.1, 0.3))),  # 10% probability to add rain
            # iaa.Sometimes(0.1, iaa.Clouds()),  # 10% probability to add clouds
            # iaa.Sometimes(0.1, iaa.Snowflakes(flake_size=(0.2, 0.4))),  # 10% probability to add snowflakes
        ]
        return iaa.Sequential(augmentations, random_order=True)

    def augment_image(self, image_path: Path):
        """
        Processes and augments a single image.

        Args:
            image_path (Path): The path to the image file.
        """
        try:
            image = imageio.imread(image_path)
            if image.shape[2] == 4:  # Remove alpha channel if present
                image = image[:, :, :3]

            label_path = self.train_path / 'labels' / image_path.with_suffix('.txt').name
            image_shape = image.shape
            bbs = BoundingBoxesOnImage(self.read_label_file(label_path, image_shape), shape=image_shape)

            for i in range(self.num_augmentations):
                image_aug, bbs_aug = self.seq(image=image, bounding_boxes=bbs)
                bbs_aug = bbs_aug.clip_out_of_image()

                aug_image_filename = image_path.stem + f"_aug_{i}" + image_path.suffix
                aug_label_filename = image_path.stem + f"_aug_{i}.txt"

                image_aug_path = self.train_path / 'images' / aug_image_filename
                label_aug_path = self.train_path / 'labels' / aug_label_filename

                imageio.imwrite(image_aug_path, image_aug)
                self.write_label_file(bbs_aug, label_aug_path, image_aug.shape[1], image_aug.shape[0])

                # 显式释放大对象以释放内存
                del image_aug, bbs_aug
                gc.collect()
        except Exception as e:
            print(f"Error augmenting image: {image_path}")
            print(e)
        finally:
            # Delete the original image to free up memory
            del image  # Always delete 'image' since it's defined at the beginning
            # Conditionally delete other variables if they've been defined
            if 'image_aug' in locals():
                del image_aug
            if 'bbs' in locals():
                del bbs
            if 'bbs_aug' in locals():
                del bbs_aug
            gc.collect()  # Force garbage collection

    def augment_data(self, batch_size=10):
        """
        Processes images in batches to save memory.
        """
        image_paths = list(self.train_path.glob('images/*.jpg'))
        total_batches = (len(image_paths) + batch_size - 1) // batch_size  # Calculate total number of batches

        with tqdm(total=len(image_paths)) as progress:
            for batch_index in range(total_batches):
                start_index = batch_index * batch_size
                end_index = min((batch_index + 1) * batch_size, len(image_paths))
                batch_paths = image_paths[start_index:end_index]

                for image_path in batch_paths:
                    self.augment_image(image_path)
                    progress.update(1)
                
                # After processing a batch, force garbage collection
                gc.collect()

    @staticmethod
    def read_label_file(label_path: Path, image_shape: Tuple[int, int, int]) -> List[BoundingBox]:
        """Reads a label file and converts it into a list of bounding boxes."""
        bounding_boxes = []
        if label_path.exists():
            with label_path.open('r') as file:
                for line in file:
                    values = list(map(float, line.split()))
                    if len(values) == 5:
                        class_id, x_center, y_center, width, height = values
                        x1 = (x_center - width / 2) * image_shape[1]
                        y1 = (y_center - height / 2) * image_shape[0]
                        x2 = (x_center + width / 2) * image_shape[1]
                        y2 = (y_center + height / 2) * image_shape[0]
                        bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=int(class_id)))
        return bounding_boxes

    @staticmethod
    def write_label_file(bounding_boxes: BoundingBoxesOnImage, label_path: Path, image_width: int, image_height: int):
        """
        Writes bounding boxes to a label file.

        Args:
            bounding_boxes (BoundingBoxesOnImage): The bounding boxes to write.
            label_path (Path): The path to the label file.
            image_width (int): The width of the image.
            image_height (int): The height of the image.
        """
        with label_path.open('w') as f:
            for bb in bounding_boxes.bounding_boxes:  # Iterate through actual bounding boxes
                x_center = (bb.x1 + bb.x2) / 2 / image_width
                y_center = (bb.y1 + bb.y2) / 2 / image_height
                width = (bb.x2 - bb.x1) / image_width
                height = (bb.y2 - bb.y1) / image_height
                f.write(f"{bb.label} {x_center} {y_center} {width} {height}\n")

    def shuffle_data(self) -> None:
        """
        Shuffles the augmented dataset to ensure randomness.

        This method pairs each image file with its corresponding label file, assigns a unique
        UUID to each pair, and then saves them with the new names into the original directories.
        """
        image_dir = self.train_path / 'images'
        label_dir = self.train_path / 'labels'

        # Retrieve paths for all image and label files
        image_paths = list(image_dir.glob('*'))
        label_paths = list(label_dir.glob('*'))

        # Ensure the count of images and labels matches
        assert len(image_paths) == len(label_paths), "The counts of image and label files do not match!"

        # Sort paths to ensure matching between images and labels
        image_paths.sort()
        label_paths.sort()

        # Shuffle the paths of images and labels together to maintain correspondence
        combined = list(zip(image_paths, label_paths))
        random.shuffle(combined)

        # Rename files with a new UUID
        for image_path, label_path in combined:
            # Generate a unique identifier
            unique_id = str(uuid.uuid4())
            new_image_name = unique_id + image_path.suffix
            new_label_name = unique_id + label_path.suffix

            new_image_path = image_dir / new_image_name
            new_label_path = label_dir / new_label_name

            image_path.rename(new_image_path)
            label_path.rename(new_label_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform data augmentation on image datasets.')
    parser.add_argument('--train_path', type=str, default='./dataset_aug/train', help='Path to the training data directory.')
    parser.add_argument('--num_augmentations', type=int, default=35, help='Number of augmentations per image.')
    parser.add_argument('--batch_size', type=int, default=10, help='Number of images to process in each batch.')
    args = parser.parse_args()

    augmenter = DataAugmentation(args.train_path, args.num_augmentations)
    # augmenter.augment_data(batch_size=args.batch_size)  # 這裡傳遞 batch_size

    # Pause for 5 seconds before shuffling to allow for user inspection.
    # print("Pausing for 5 seconds before shuffling data...")
    # time.sleep(5)
    
    augmenter.shuffle_data()
    print("Data augmentation and shuffling complete.")
