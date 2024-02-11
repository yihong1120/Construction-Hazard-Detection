import os
from typing import List, Tuple
import imageio.v3 as imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import argparse
from tqdm import tqdm
import shutil
import random
from pathlib import Path
import glob

class DataAugmentation:
    """ 
    A class to perform data augmentation for image datasets, especially useful for training machine learning models.
    
    Attributes:
        train_path (str): The path to the training data.
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
        self.train_path = train_path
        self.num_augmentations = num_augmentations
        self.seq = self._get_augmentation_sequence()

    def _get_augmentation_sequence(self) -> iaa.Sequential:
        """Define a sequence of augmentations."""
        return iaa.Sequential([
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-15, 15)),
            iaa.Multiply((0.8, 1.2)),
            iaa.LinearContrast((0.8, 1.2)),
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.Resize((0.7, 1.3)),
            iaa.Crop(px=(0, 16)),
            iaa.SaltAndPepper(0.02),
            iaa.ElasticTransformation(alpha=50, sigma=5),
            iaa.ShearX((-20, 20)),
            iaa.ShearY((-20, 20)),
            iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2)),
            iaa.PiecewiseAffine(scale=(0.01, 0.03)),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            # Colour transformation augmenters
            iaa.AddToHueAndSaturation((-30, 30)),
            iaa.GammaContrast((0.5, 1.5)),
            iaa.AddToHueAndSaturation((-20, 20)),
        ], random_order=True)

    def augment_image(self, image_path: str):
        """
        Process and augment a single image.
        """
        image = imageio.imread(image_path)
        label_path = image_path.replace('images', 'labels').replace('.png', '.txt').replace('.jpg', '.txt').replace('.jpeg', '.txt')
        image_shape = image.shape
        bbs = BoundingBoxesOnImage(self.read_label_file(label_path, image_shape), shape=image_shape)

        for i in range(self.num_augmentations):
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image_aug, bbs_aug = self.seq(image=image, bounding_boxes=bbs)

            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            file_extension = os.path.splitext(image_path)[1]
            aug_image_filename = f"{base_filename}_aug_{i}{file_extension}"
            aug_label_filename = f"{base_filename}_aug_{i}.txt"

            image_aug_path = os.path.join(self.train_path, 'images', aug_image_filename)
            label_aug_path = os.path.join(self.train_path, 'labels', aug_label_filename)

            imageio.imwrite(image_aug_path, image_aug)
            self.write_label_file(bbs_aug, label_aug_path, image_aug.shape[1], image_aug.shape[0])

    def augment_data(self):
        """
        Perform data augmentation on all images in the dataset.
        """
        file_types = ['*.png', '*.jpg', '*.jpeg']
        image_paths = []
        for file_type in file_types:
            image_paths.extend(glob.glob(os.path.join(self.train_path, 'images', file_type)))

        for image_path in tqdm(image_paths):
            self.augment_image(image_path)

    @staticmethod
    def read_label_file(label_path: str, image_shape: Tuple[int, int, int]) -> List[BoundingBox]:
        bounding_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    x1 = (x_center - width / 2) * image_shape[1]
                    y1 = (y_center - height / 2) * image_shape[0]
                    x2 = (x_center + width / 2) * image_shape[1]
                    y2 = (y_center + height / 2) * image_shape[0]
                    bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=int(class_id)))
        return bounding_boxes

    @staticmethod
    def write_label_file(bounding_boxes: BoundingBoxesOnImage, label_path: str, image_width: int, image_height: int):
        with open(label_path, 'w') as f:
            for bb in bounding_boxes:
                x_center = ((bb.x1 + bb.x2) / 2) / image_width
                y_center = ((bb.y1 + bb.y2) / 2) / image_height
                width = (bb.x2 - bb.x1) / image_width
                height = (bb.y2 - bb.y1) / image_height
                x_center, y_center, width, height = [max(0, min(1, val)) for val in [x_center, y_center, width, height]]
                class_index = bb.label
                f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")

    def shuffle_data(self) -> None:
        """
        Shuffles the augmented dataset to ensure randomness.

        This method pairs each image file with its corresponding label file, shuffles these pairs,
        and then saves them into temporary directories. After shuffling, it renames the temporary
        directories to the original ones, effectively replacing them with the shuffled files.

        Raises:
            AssertionError: If the number of image files and label files do not match.
        """
        image_dir = os.path.join(self.train_path, 'images')
        label_dir = os.path.join(self.train_path, 'labels')

        # Retrieve paths for all image and label files
        image_paths: List[str] = glob.glob(os.path.join(image_dir, '*'))
        label_paths: List[str] = glob.glob(os.path.join(label_dir, '*'))

        # Ensure the count of images and labels matches
        assert len(image_paths) == len(label_paths), "The counts of image and label files do not match!"

        # Shuffle the paths of images and labels together to maintain correspondence
        combined: List[Tuple[str, str]] = list(zip(image_paths, label_paths))
        random.shuffle(combined)
        image_paths, label_paths = zip(*combined)

        # Create temporary directories for shuffled files
        temp_image_dir: str = os.path.join(self.train_path, 'temp_images')
        temp_label_dir: str = os.path.join(self.train_path, 'temp_labels')
        os.makedirs(temp_image_dir, exist_ok=True)
        os.makedirs(temp_label_dir, exist_ok=True)

        # Move shuffled files to the temporary directories
        for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
            new_image_path: str = os.path.join(temp_image_dir, f"{i:06d}" + os.path.splitext(image_path)[1])
            new_label_path: str = os.path.join(temp_label_dir, f"{i:06d}.txt")
            shutil.move(image_path, new_image_path)
            shutil.move(label_path, new_label_path)

        # Remove the original directories
        shutil.rmtree(image_dir)
        shutil.rmtree(label_dir)

        # Rename temporary directories to the original names
        os.rename(temp_image_dir, image_dir)
        os.rename(temp_label_dir, label_dir)

        print("Dataset shuffled successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform data augmentation on image datasets.')
    parser.add_argument('--train_path', type=str, default='../dataset_aug/train', help='Path to the training data')
    parser.add_argument('--num_augmentations', type=int, default=1, help='Number of augmentations per image')
    args = parser.parse_args()
    
    augmenter = DataAugmentation(args.train_path, args.num_augmentations)
    augmenter.augment_data()
    augmenter.shuffle_data()