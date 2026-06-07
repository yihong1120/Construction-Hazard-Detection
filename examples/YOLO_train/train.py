from __future__ import annotations

import argparse
import os
import shutil
from typing import Any

import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sklearn.model_selection import KFold
from ultralytics import YOLO


class YOLOModelHandler:
    """Handles loading, training, validating, and predicting with YOLO models.

    Attributes:
        model_name (str): The name of the model file to be loaded.
        model (YOLO, Optional): The loaded YOLO model object.
    """

    def __init__(self, model_name: str, batch_size: int = -1) -> None:
        """
        Initialises the YOLOModelHandler with a specified model.

        Args:
            model_name (str): The name of the model file (either .yaml or .pt).
            batch_size (int): The batch size for training and validation.

        Raises:
            ValueError: If the model format is not supported.
        """
        self.model_name: str = model_name
        self.model: YOLO | None = None
        self.batch_size: int = batch_size
        self.load_model()

    def load_model(self) -> None:
        """Loads the YOLO model specified by the model name."""
        if self.model_name.endswith('.yaml'):
            # Build a new model from scratch
            self.model = YOLO(self.model_name)
            # Set device to CPU by default for YAML models
            self.device = torch.device('cpu')
        elif self.model_name.endswith('.pt'):
            # Load a pre-trained model (recommended for training)
            self.model = YOLO(self.model_name)

            # Check and set the device
            if torch.backends.mps.is_available():
                # Use MPS if available
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                # Use CUDA if MPS is unavailable but CUDA is
                self.device = torch.device('cuda')
            else:
                # Use CPU if neither MPS nor CUDA is available
                self.device = torch.device('cpu')

        # Load the model onto the specified device
        if self.model:
            self.model.to(self.device)
        else:
            raise ValueError("Unsupported model format. Use '.yaml' or '.pt'")

    def train_model(
        self, data_config: str, epochs: int, optimizer: str,
    ) -> None:
        """
        Trains the YOLO model with the given data config and number of epochs.

        Args:
            data_config (str): The path to the data configuration file.
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training and validation.
            optimizer (str): The type of optimizer to use.

        Raises:
            RuntimeError: If the model is not loaded properly before training.
        """
        if self.model is None:
            raise RuntimeError('The model is not loaded properly.')
        # Train the model
        self.model.train(
            data=data_config,
            epochs=epochs,
            batch=self.batch_size,
            optimizer=optimizer,
        )

    def validate_model(self) -> Any:
        """
        Validates the YOLO model on the validation dataset.

        Args:
            batch_size (int): The batch size for training and validation.

        Returns:
            The validation results.

        Raises:
            RuntimeError: If model is not loaded properly before validation.
        """
        if self.model is None:
            raise RuntimeError('The model is not loaded properly.')
        # Evaluate model performance on the validation set
        return self.model.val(batch=self.batch_size)

    def predict_image(self, image_path: str) -> Any:
        """
        Makes a prediction using the YOLO model on the specified image.

        Args:
            image_path (str): The path to the image file for prediction.

        Returns:
            The prediction results.

        Raises:
            RuntimeError: If  model is not loaded properly before prediction.
        """
        if self.model is None:
            raise RuntimeError('The model is not loaded properly.')
        # Predict on an image
        return self.model(image_path)

    @staticmethod
    def predict_image_sahi(yolo_model_path: str, image_path: str) -> Any:
        """
        Makes a prediction using the YOLO model on the specified image
        with SAHI post-processing.

        Args:
            yolo_model_path (str): The path to the YOLO model file.
            image_path (str): The path to the image file for prediction.

        Returns:
            The prediction results with SAHI post-processing.

        Raises:
            RuntimeError: If model is not loaded properly before prediction.
        """
        if not yolo_model_path:
            raise RuntimeError('The model path is not provided.')

        # Convert YOLO model to SAHI format; adjust for your YOLO version
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=yolo_model_path,
            confidence_threshold=0.3,
            # device="cpu", or 'cuda:0'
        )

        # With an image path, get the sliced prediction
        result = get_sliced_prediction(
            image_path,
            sahi_model,
            slice_height=640,
            slice_width=640,
            overlap_height_ratio=0.2,
            overlap_width_ratio=0.2,
        )

        # Visualise the prediction results
        result.export_visuals(export_dir='./')

        # Access the object prediction list
        object_prediction_list = result.object_prediction_list

        # Return the SAHI formatted results
        return object_prediction_list

    def export_model(self, export_format: str = 'onnx') -> str:
        """
        Exports the YOLO model to the specified format.

        Args:
            export_format (str): The format to export the model to.

        Returns:
            The path to the exported model file.

        Raises:
            RuntimeError: If the model is not loaded properly before exporting.
        """
        if self.model is None:
            raise RuntimeError('The model is not loaded properly.')
        # Export the model to the desired format
        return self.model.export(format=export_format)

    def save_model(self, save_path: str) -> None:
        """
        Saves the YOLO model to a .pt file.

        Args:
            save_path (str): The path to save the .pt model file.
        """
        if self.model is None:
            raise RuntimeError('The model is not loaded properly.')
        # Save the model to the specified path
        torch.save(self.model.state_dict(), save_path)

    def cross_validate_model(
        self,
        data_config: str,
        epochs: int,
        optimizer: str,
        n_splits: int = 5,
    ) -> None:
        """
        Performs k-fold cross-validation on the YOLO model.

        Args:
            data_config (str): The path to the data configuration file.
            epochs (int): The number of training epochs.
            optimizer (str): The type of optimizer to use.
            n_splits (int): Number of folds for cross-validation.

        Raises:
            RuntimeError: If the model is not loaded properly before training.
        """
        if self.model is None:
            raise RuntimeError('The model is not loaded properly.')

        # Load the data
        dataset_path = os.path.join(os.path.dirname(data_config))
        images_path = os.path.join(dataset_path, 'images')
        labels_path = os.path.join(dataset_path, 'labels')

        # List all image files
        image_files = [
            f
            for f in os.listdir(
                images_path,
            )
            if os.path.isfile(os.path.join(images_path, f))
        ]
        kf = KFold(n_splits=n_splits)

        fold = 1
        for train_index, val_index in kf.split(image_files):
            train_images = [image_files[i] for i in train_index]
            val_images = [image_files[i] for i in val_index]

            # Create temporary directories for training and validation sets
            temp_train_dir = os.path.join(dataset_path, 'train')
            temp_val_dir = os.path.join(dataset_path, 'val')

            os.makedirs(temp_train_dir, exist_ok=True)
            os.makedirs(temp_val_dir, exist_ok=True)

            os.makedirs(os.path.join(temp_train_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(temp_train_dir, 'labels'), exist_ok=True)
            os.makedirs(os.path.join(temp_val_dir, 'images'), exist_ok=True)
            os.makedirs(os.path.join(temp_val_dir, 'labels'), exist_ok=True)

            # Copy files to the temporary directories
            for image in train_images:
                shutil.copy(
                    os.path.join(images_path, image),
                    os.path.join(temp_train_dir, 'images', image),
                )
                shutil.copy(
                    os.path.join(
                        labels_path,
                        image.replace(
                            '.jpg',
                            '.txt',
                        ).replace('.png', '.txt'),
                    ),
                    os.path.join(
                        temp_train_dir,
                        'labels',
                        image.replace(
                            '.jpg',
                            '.txt',
                        ).replace('.png', '.txt'),
                    ),
                )

            for image in val_images:
                shutil.copy(
                    os.path.join(images_path, image),
                    os.path.join(temp_val_dir, 'images', image),
                )
                shutil.copy(
                    os.path.join(
                        labels_path,
                        image.replace(
                            '.jpg',
                            '.txt',
                        ).replace('.png', '.txt'),
                    ),
                    os.path.join(
                        temp_val_dir,
                        'labels',
                        image.replace(
                            '.jpg',
                            '.txt',
                        ).replace('.png', '.txt'),
                    ),
                )

            # Update data_config file for this fold
            with open(data_config) as file:
                data_yaml = file.read()

            data_yaml = data_yaml.replace(
                'dataset/train/images',
                temp_train_dir + '/images',
            )
            data_yaml = data_yaml.replace(
                'dataset/valid/images',
                temp_val_dir + '/images',
            )

            temp_data_config = os.path.join(
                dataset_path,
                f"data_fold{fold}.yaml",
            )
            with open(temp_data_config, 'w') as file:
                file.write(data_yaml)

            # Reload the model to ensure a fresh start for each fold
            self.load_model()

            print(f"Training fold {fold}/{n_splits}")
            self.train_model(
                data_config=temp_data_config,
                epochs=epochs,
                optimizer=optimizer,
            )
            metrics = self.validate_model()
            print(f"Validation metrics for fold {fold}:", metrics)

            # Clean up temporary directories
            shutil.rmtree(temp_train_dir)
            shutil.rmtree(temp_val_dir)
            os.remove(temp_data_config)

            fold += 1

    def train_on_all_data(
        self,
        data_config: str,
        epochs: int,
        optimizer: str,
    ) -> None:
        """
        Trains the model on the entire dataset (all images) without splitting.
        This provides the strongest model that has seen all available data.

        Args:
            data_config (str): The path to the data configuration file.
            epochs (int): The number of training epochs.
            optimizer (str): The type of optimizer to use.
        """
        if self.model is None:
            raise RuntimeError('The model is not loaded properly.')

        print('Preparing to train on all data...')

        # Load the data paths
        dataset_path = os.path.dirname(data_config)
        images_path = os.path.join(dataset_path, 'images')

        # Verify the all-images directory exists (as expected by the CV logic)
        if not os.path.exists(images_path) or not os.path.isdir(images_path):
            print(
                (
                    f"Warning: 'images' folder not found at {images_path}. "
                    'Cannot perform all-data training.'
                ),
            )
            return

        # Create a temporary config for all-data training
        # We read the original config to preserve class names
        with open(data_config) as file:
            original_yaml = file.read()

        # Create a new YAML focusing on the full images directory
        # We update train and val to point to the absolute path
        # of the images folder.
        abs_images_path = os.path.abspath(images_path)

        # We try to preserve the 'names' and 'nc' parts while replacing paths
        # A simple append/override strategy:
        # We write a new file that defines path/train/val at the top.
        # If 'names' is a dict in the file, it will be parsed.

        temp_all_data_config = os.path.join(
            dataset_path, 'all_data_train.yaml',
        )

        with open(temp_all_data_config, 'w') as f:
            # We override the path settings.
            # Note: If the original file has 'train:' lines,
            # this prepend might conflict if not handled,
            # but usually last key wins or we can comment out old lines.
            # To be safe, we'll try to use the dictionary mode if
            # supported or just rely on the fact that we're pointing
            # to the same dataset structure but unified.

            # Better approach: Read lines, exclude path/train/val
            # lines, write new ones.
            lines = original_yaml.splitlines()
            filtered_lines = [
                line
                for line in lines
                if not any(
                    line.strip().startswith(k)
                    for k in ['path:', 'train:', 'val:', 'test:']
                )
            ]

            f.write(f"path: {os.path.dirname(abs_images_path)}\n")
            f.write('train: images\n')
            # Validate on training data just to enable the process
            f.write('val: images\n')
            f.write('\n'.join(filtered_lines))

        # Reload model to start fresh
        self.load_model()

        print(f"Training on all data using config: {temp_all_data_config}")
        self.train_model(
            data_config=temp_all_data_config,
            epochs=epochs,
            optimizer=optimizer,
        )

        # Cleanup
        if os.path.exists(temp_all_data_config):
            os.remove(temp_all_data_config)


def main() -> None:
    """Run the YOLO command-line training workflow."""
    parser = argparse.ArgumentParser(
        description='YOLO training, validation, prediction, and export.',
    )

    parser.add_argument(
        '--data_config',
        type=str,
        default='dataset/data.yaml',
        help='Path to the data configuration file',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs',
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default='./../../models/pt/best_yolo26x.pt',
        help='Name or path of the YOLO model file',
    )
    parser.add_argument(
        '--export_format',
        type=str,
        default='onnx',
        help='Format to export the model to',
    )
    parser.add_argument(
        '--onnx_path',
        type=str,
        default=None,
        help='Path to save the exported ONNX model',
    )
    parser.add_argument(
        '--pt_path',
        type=str,
        default='model.pt',
        help='Path to save the trained model in .pt format',
    )
    parser.add_argument(
        '--sahi_image_path',
        type=str,
        default='../../assets/IMG_1091.PNG',
        help='Path to the image file for SAHI prediction',
    )

    parser.add_argument(
        '--batch_size',
        type=int,
        default=-1,
        help='Batch size for training and validation',
    )

    parser.add_argument(
        '--optimizer',
        type=str,
        default='auto',
        help='Type of optimizer to use',
    )

    parser.add_argument(
        '--cross_validate',
        action='store_true',
        help='Perform cross-validation',
    )

    parser.add_argument(
        '--n_splits',
        type=int,
        default=5,
        help='Number of folds for cross-validation',
    )

    parser.add_argument(
        '--full_data_training',
        action='store_true',
        help=(
            'Train on all available data (images folder) '
            'without validation split'
        ),
    )

    args = parser.parse_args()

    handler = YOLOModelHandler(args.model_name, args.batch_size)

    try:
        # 1. Cross Validation
        if args.cross_validate:
            print('--- Starting Cross-Validation ---')
            handler.cross_validate_model(
                data_config=args.data_config,
                epochs=args.epochs,
                optimizer=args.optimizer,
                n_splits=args.n_splits,
            )
            print('--- Cross-Validation Complete ---')

        # 2. Final Training on All Data (Optional or Standalone)
        if args.full_data_training:
            print('--- Starting Final Training on All Data ---')
            handler.train_on_all_data(
                data_config=args.data_config,
                epochs=args.epochs,
                optimizer=args.optimizer,
            )

        # 3. Standard Training (Default if no special modes selected)
        if not args.cross_validate and not args.full_data_training:
            print('--- Starting Standard Training (Train/Val Split) ---')
            handler.train_model(
                data_config=args.data_config,
                epochs=args.epochs,
                optimizer=args.optimizer,
            )
            metrics = handler.validate_model()
            print('Validation metrics:', metrics)

        export_path = (
            handler.export_model(export_format=args.export_format)
            if args.onnx_path is None
            else args.onnx_path
        )
        handler.save_model(args.pt_path)
    except Exception as e:
        print(f"Error occurred: {e}")
        exit(1)

    print(f"{args.export_format.upper()} model exported to:", export_path)
    print(f"Model saved to: {args.pt_path}")


if __name__ == '__main__':
    main()

    # Predict on an image
    # results = handler.predict_image("https://ultralytics.com/images/bus.jpg")

    # SAHI Prediction
    # sahi_result = handler.predict_image_sahi(
    #     args.model_name, args.sahi_image_path
    # )
    # print("SAHI Prediction Results:", sahi_result)

    # Example command to run the script
    # python train.py \
    #     --data_config=cv_dataset/data.yaml \
    #     --epochs=2 \
    #     --model_name=../../models/pt/yolo26n.pt \
    #     --batch_size=16 \
    #     --optimizer=auto \
    #     --cross_validate \
    #     --n_splits=5

    # python train.py \
    #     --data_config=cv_dataset/data.yaml \
    #     --model_name=../../models/pt/best_yolo26l.pt \
    #     --epochs=100 \
    #     --batch_size=16 \
    #     --full_data_training
