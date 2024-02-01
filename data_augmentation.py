import glob
import os
from typing import List, Tuple
import imageio.v2 as imageio
import imgaug as ia
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import imgaug.augmenters as iaa
import argparse
from tqdm import tqdm


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
        The constructor for DataAugmentation class.

        Parameters:
            train_path (str): The path to the training data.
            num_augmentations (int): The number of augmentations to perform per image.
        """
        self.train_path = train_path
        self.num_augmentations = num_augmentations
        # Define a sequence of augmentations
        self.seq = iaa.Sequential([
            iaa.Flipud(0.5), 
            iaa.Fliplr(0.5),  
            iaa.Affine(rotate=(-15, 15)),  # Adjust the range of rotation angles to -15 degrees to 15 degrees
            iaa.Multiply((0.8, 1.2)),  # Adjust the range of brightness variation
            iaa.LinearContrast((0.8, 1.2)),  # Adjust the range of contrast variation
            iaa.GaussianBlur(sigma=(0, 0.5)),  # Adjust the strength of Gaussian blur
            iaa.Resize((0.7, 1.3)),  # Adjust the range of random scaling
            iaa.Crop(px=(0, 16)),  # Adjust the range of random cropping
            iaa.SaltAndPepper(0.02),  # Adjust the strength of salt and pepper noise
            iaa.ElasticTransformation(alpha=50, sigma=5),  # Adjust the strength of elastic transformation
            iaa.ShearX((-20, 20)),  # Adjust the range of shear transformation for the x-axis
            iaa.ShearY((-20, 20)),  # Adjust the range of shear transformation for the y-axis
            iaa.Sharpen(alpha=(0, 0.5), lightness=(0.8, 1.2)),  # Adjust the strength of sharpening
            iaa.PiecewiseAffine(scale=(0.01, 0.03)),  # Adjust the strength of piecewise affine transformation
            iaa.Grayscale(alpha=(0.0, 1.0))  # Add random grayscale processing
        ], random_order=True)

    def augment_data(self):
        """ Performs the augmentation on the dataset. """
        image_paths = glob.glob(os.path.join(self.train_path, 'images', '*.png'))

        # Encapsulate the image_paths loop to display a progress bar
        for image_path in tqdm(image_paths, desc="Augmenting Images"):
            image = imageio.imread(image_path)
            # Replace 'images' with 'labels' in the path and change file extension to read labels
            label_path = image_path.replace('images', 'labels').replace('.png', '.txt')
            image_shape = image.shape
            # Read bounding box information and initialise them on the image
            bbs = BoundingBoxesOnImage(self.read_label_file(label_path, image_shape), shape=image_shape)

            for i in range(self.num_augmentations):
                # If the image has an alpha channel, remove it
                if image.shape[2] == 4:
                    image = image[:, :, :3]

                # Perform augmentation
                image_aug, bbs_aug = self.seq(image=image, bounding_boxes=bbs)

                # Prepare filenames for augmented images and labels
                base_filename = os.path.splitext(os.path.basename(image_path))[0]
                aug_image_filename = f"{base_filename}_aug_{i}.png"
                aug_label_filename = f"{base_filename}_aug_{i}.txt"
                
                # Define paths for saving the augmented images and labels
                image_aug_path = os.path.join(self.train_path, 'images', aug_image_filename)
                label_aug_path = os.path.join(self.train_path, 'labels', aug_label_filename)

                # Save the augmented image and label
                imageio.imwrite(image_aug_path, image_aug)
                self.write_label_file(bbs_aug.remove_out_of_image().clip_out_of_image(), label_aug_path, image_shape[1], image_shape[0])

    @staticmethod
    def read_label_file(label_path: str, image_shape: Tuple[int, int, int]) -> List[BoundingBox]:
        """
        Reads a label file and converts annotations into BoundingBox objects.

        Parameters:
            label_path (str): The path to the label file.
            image_shape (Tuple[int, int, int]): The shape of the image.

        Returns:
            List[BoundingBox]: A list of BoundingBox objects.
        """
        bounding_boxes = []
        if os.path.exists(label_path):
            with open(label_path, 'r') as file:
                for line in file:
                    # Parsing label information for bounding box creation
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
        Writes the augmented bounding box information back to a label file.

        Parameters:
            bounding_boxes (List[BoundingBox]): A list of augmented BoundingBox objects.
            label_path (str): The path where the label file is to be saved.
            image_width (int): The width of the image.
            image_height (int): The height of the image.
        """
        # Open the label file for writing
        with open(label_path, 'w') as f:
            # Iterate over the augmented bounding boxes and write them to the file
            for bb in bounding_boxes:
                # Calculate the center, width, and height for the YOLO format
                x_center = ((bb.x1 + bb.x2) / 2) / image_width
                y_center = ((bb.y1 + bb.y2) / 2) / image_height
                width = (bb.x2 - bb.x1) / image_width
                height = (bb.y2 - bb.y1) / image_height

                # Ensure all values are within [0, 1]
                x_center, y_center, width, height = [max(0, min(1, val)) for val in [x_center, y_center, width, height]]

                class_index = bb.label
                # Write the bounding box information in the YOLO format
                f.write(f"{class_index} {x_center} {y_center} {width} {height}\n")


if __name__ == '__main__':
    # Create a parser
    parser = argparse.ArgumentParser(description='Perform data augmentation on image datasets.')
    # Add train_path argument with a default value
    parser.add_argument('--train_path', type=str, default='dataset/train', help='Path to the training data')
    # Add num_augmentations argument with a default value
    parser.add_argument('--num_augmentations', type=int, default=5, help='Number of augmentations per image')

    # Parse the command line arguments
    args = parser.parse_args()

    # Use the parsed arguments
    train_path = args.train_path
    num_augmentations = args.num_augmentations

    # Initialise the DataAugmentation class
    augmenter = DataAugmentation(train_path, num_augmentations)
    # Perform the data augmentation
    augmenter.augment_data()
