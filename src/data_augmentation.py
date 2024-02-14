from typing import List, Tuple
import imageio.v3 as imageio
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import argparse
from concurrent.futures import ThreadPoolExecutor
import shutil
import random
from pathlib import Path

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
        Define a sequence of augmentations.

        Returns:
            iaa.Sequential: The sequence of augmentations to apply.
        """
        mandatory_augmentations = [
            iaa.Flipud(0.5),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-15, 15)),
            iaa.Resize((0.5, 1.3)),
        ]
        optional_augmentations = [
            iaa.Multiply((0.8, 1.2)),
            iaa.LinearContrast((0.8, 1.2)),
            iaa.GaussianBlur(sigma=(0, 0.5)),
            iaa.Crop(px=(0, 16)),
            iaa.SaltAndPepper(0.02),
            iaa.ElasticTransformation(alpha=(0, 30), sigma=10),
            iaa.ShearX((-20, 20)),
            iaa.ShearY((-20, 20)),
            iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2)),
            iaa.PiecewiseAffine(scale=(0.01, 0.03)),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.AddToHueAndSaturation((-30, 30)),
            iaa.GammaContrast((0.5, 1.5)),
            iaa.ChangeColorTemperature((3300, 6500)),
            iaa.PerspectiveTransform(scale=(0.01, 0.1)),
            iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
            iaa.Invert(0.3),
        ]
        num_augmentations = random.randint(1, len(optional_augmentations))  # Randomly choose the number of augmentations
        chosen_augmentations = random.sample(optional_augmentations, num_augmentations)
        final_augmentations_sequence = mandatory_augmentations + chosen_augmentations
        return iaa.Sequential(final_augmentations_sequence, random_order=True)

    def augment_image(self, image_path: Path):
        """
        Process and augment a single image.

        Args:
            image_path (Path): The path to the image to augment.
        """
        image = imageio.imread(image_path)
        label_path = image_path.with_suffix('.txt').parent.parent / 'labels' / image_path.with_suffix('.txt').name
        image_shape = image.shape
        bbs = BoundingBoxesOnImage(self.read_label_file(label_path, image_shape), shape=image_shape)

        for i in range(self.num_augmentations):
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image_aug, bbs_aug = self.seq(image=image, bounding_boxes=bbs)

            # Clip bounding boxes that fall out of the image
            bbs_aug = bbs_aug.clip_out_of_image()

            aug_image_filename = image_path.stem + f"_aug_{i}" + image_path.suffix
            aug_label_filename = image_path.stem + f"_aug_{i}.txt"

            image_aug_path = self.train_path / 'images' / aug_image_filename
            label_aug_path = self.train_path / 'labels' / aug_label_filename

            imageio.imwrite(image_aug_path, image_aug)
            self.write_label_file(bbs_aug, label_aug_path, image_aug.shape[1], image_aug.shape[0])

    def augment_data(self):
        """
        Perform data augmentation on all images in the dataset.
        """
        image_paths = (path for path in self.train_path.glob('images/*'))

        with ThreadPoolExecutor(max_workers=5) as executor:
            executor.map(self.augment_image, image_paths)

    @staticmethod
    def read_label_file(label_path: Path, image_shape: Tuple[int, int, int]) -> List[BoundingBox]:
        """
        Read a label file and convert it to bounding boxes.

        Args:
            label_path (Path): The path to the label file.
            image_shape (Tuple[int, int, int]): The shape of the image.

        Returns:
            List[BoundingBox]: A list of bounding boxes.
        """
        bounding_boxes = []
        if label_path.exists():
            with open(label_path, 'r') as file:
                for line in file:
                    class_id, x_center, y_center, width, height = map(float, line.split())
                    x1 = (x_center - width / 2) * image_shape[1]
                    y1 = (y_center - height / 2) * image_shape[0]
                    x2 = (x_center + width / 2) * image_shape[1]
                    y2 = (y_center + height / 2) * image_shape[0]
                    bounding_boxes.append(BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=int(class_id)))
        else:
            print(f"Label file {label_path} does not exist")
        return bounding_boxes

    @staticmethod
    def write_label_file(bounding_boxes: BoundingBoxesOnImage, label_path: Path, image_width: int, image_height: int):
        """
        Write bounding boxes to a label file.

        Args:
            bounding_boxes (BoundingBoxesOnImage): The bounding boxes to write.
            label_path (Path): The path to the label file.
            image_width (int): The width of the image.
            image_height (int): The height of the image.
        """
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
        image_dir = self.train_path / 'images'
        label_dir = self.train_path / 'labels'

        # Retrieve paths for all image and label files
        image_paths = list(image_dir.glob('*'))
        label_paths = list(label_dir.glob('*'))

        # Ensure the count of images and labels matches and that they correspond to each other
        assert len(image_paths) == len(label_paths), "The counts of image and label files do not match!"
        image_paths.sort()
        label_paths.sort()
        assert all(image_path.stem == label_path.stem for image_path, label_path in zip(image_paths, label_paths)), "Image and label files do not correspond to each other!"

        # Shuffle the paths of images and labels together to maintain correspondence
        combined = list(zip(image_paths, label_paths))
        random.shuffle(combined)

        # Create temporary directories for shuffled files
        temp_image_dir = self.train_path / 'temp_images'
        temp_label_dir = self.train_path / 'temp_labels'
        temp_image_dir.mkdir(exist_ok=True)
        temp_label_dir.mkdir(exist_ok=True)

        # Move shuffled files to the temporary directories
        for i, (image_path, label_path) in enumerate(combined):
            # Maintain the original file name but prepend with a sortable index
            new_image_name = f"{i:06d}_{image_path.name}"
            new_label_name = f"{i:06d}_{label_path.name}"
            new_image_path = temp_image_dir / new_image_name
            new_label_path = temp_label_dir / new_label_name

            image_path.rename(new_image_path)
            label_path.rename(new_label_path)

        # Remove the original directories
        shutil.rmtree(image_dir)
        shutil.rmtree(label_dir)

        # Rename temporary directories to the original names
        temp_image_dir.rename(image_dir)
        temp_label_dir.rename(label_dir)

        print("Dataset shuffled successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform data augmentation on image datasets.')
    parser.add_argument('--train_path', type=str, default='../dataset_aug/train', help='Path to the training data')
    parser.add_argument('--num_augmentations', type=int, default=30, help='Number of augmentations per image')
    args = parser.parse_args()
    
    augmenter = DataAugmentation(args.train_path, args.num_augmentations)
    augmenter.augment_data()
    augmenter.shuffle_data()