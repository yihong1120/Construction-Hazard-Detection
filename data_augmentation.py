import glob
import os
from typing import List, Tuple
import imageio.v2 as imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import argparse
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


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
                """
                Sharpen the image with variable alpha and lightness.

                Args:
                    alpha (Tuple[float, float]): Range of alpha values for sharpening.
                    lightness (Tuple[float, float]): Range of lightness values for sharpening.
                """,
            iaa.PiecewiseAffine(scale=(0.01, 0.03)),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            # Colour transformation augmenters
            iaa.AddToHueAndSaturation((-30, 30)),
            iaa.GammaContrast((0.5, 1.5)),
            iaa.AddToHueAndSaturation((-20, 20)),
        ], random_order=True)

    def augment_image(self, image_path: str, num_augmentations: int, seq: iaa.Sequential, file_extension: str):
        """
        Process and augment a single image.

        Args:
            image_path (str): The path to the input image.
            num_augmentations (int): The number of augmentations to perform.
            seq (iaa.Sequential): The sequence of augmentations to apply.
            file_extension (str): The file extension for the augmented images.
        """
        # Load the image
        image = imageio.imread(image_path)
        
        # Get the label file path
        label_path = image_path.replace('images', 'labels').replace('.png', '.txt')
        
        # Get the shape of the image
        image_shape = image.shape
        
        # Convert the annotations into BoundingBox objects
        bbs = BoundingBoxesOnImage(self.read_label_file(label_path, image_shape), shape=image_shape)
        
        # Perform augmentations
        for i in range(num_augmentations):
            # Remove the alpha channel if it exists
            if image.shape[2] == 4:
                image = image[:, :, :3]
            image_aug, bbs_aug = seq(image=image, bounding_boxes=bbs)

            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            aug_image_filename = f"{base_filename}_aug_{i}{file_extension}"
            aug_label_filename = f"{base_filename}_aug_{i}.txt"
            
            image_aug_path = os.path.join(self.train_path, 'images', aug_image_filename)
            label_aug_path = os.path.join(self.train_path, 'labels', aug_label_filename)

            imageio.imwrite(image_aug_path, image_aug)
            self.write_label_file(bbs_aug.remove_out_of_image().clip_out_of_image(), label_aug_path, image_shape[1], image_shape[0])

    def augment_data(self):
        """
        Perform data augmentation on all images in the dataset.
        """
        # Support file types
        file_types = ['*.png', '*.jpg', '*.jpeg']
        
        # Fetch all the file paths of matching files
        image_paths = []
        for file_type in file_types:
            image_paths.extend(glob.glob(os.path.join(self.train_path, 'images', file_type)))

        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            # Conduct correct parameters to augment_image
            futures = [executor.submit(self.augment_image, image_path, self.num_augmentations, self.seq, os.path.splitext(image_path)[1]) for image_path in image_paths]
            for future in tqdm(as_completed(futures), total=len(futures)):
                future.result()

    @staticmethod
    def read_label_file(label_path: str, image_shape: Tuple[int, int, int]) -> List[BoundingBox]:
        """
        Read a label file and convert annotations into BoundingBox objects.

        Args:
            label_path (str): The path to the label file.
            image_shape (Tuple[int, int, int]): The shape of the image.

        Returns:
            List[BoundingBox]: A list of BoundingBox objects representing the annotations.
        """
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
    def write_label_file(bounding_boxes: List[BoundingBox], label_path: str, image_width: int, image_height: int):
        """
        Write the augmented bounding box information back to a label file.

        Args:
            bounding_boxes (List[BoundingBox]): The list of augmented BoundingBox objects.
            label_path (str): The path where the label file is to be saved.
            image_width (int): The width of the image.
            image_height (int): The height of the image.
        """
        """
        Write the augmented bounding box information back to a label file.

        Args:
            bounding_boxes (List[BoundingBox]): The list of augmented BoundingBox objects.
            label_path (str): The path where the label file is to be saved.
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform data augmentation on image datasets.')
    parser.add_argument('--train_path', type=str, default='dataset_aug/train', help='Path to the training data')
    parser.add_argument('--num_augmentations', type=int, default=150, help='Number of augmentations per image')
    args = parser.parse_args()
    augmenter = DataAugmentation(args.train_path, args.num_augmentations)
    augmenter.augment_data()
