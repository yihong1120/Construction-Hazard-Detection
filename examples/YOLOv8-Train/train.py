import argparse
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import Any, Optional
from ultralytics import YOLO
from sklearn.model_selection import KFold
import os
import shutil


class YOLOModelHandler:
    """Handles loading, training, validating, and predicting with YOLO models.

    Attributes:
        model_name (str): The name of the model file to be loaded.
        model (YOLO, Optional): The loaded YOLO model object.
    """

    def __init__(self, model_name: str, batch_size: int = -1):
        """
        Initialises the YOLOModelHandler with a specified model.

        Args:
            model_name (str): The name of the model file (either .yaml or .pt).
            batch_size (int): The batch size for training and validation.

        Raises:
            ValueError: If the model format is not supported.
        """
        self.model_name: str = model_name
        self.model: Optional[YOLO] = None
        self.batch_size: int = batch_size
        self.load_model()

    def load_model(self) -> None:
        """Loads the YOLO model specified by the model name."""
        if self.model_name.endswith(".yaml"):
            # Build a new model from scratch
            self.model = YOLO(self.model_name)
        elif self.model_name.endswith(".pt"):
            # Load a pre-trained model (recommended for training)
            self.model = YOLO(self.model_name)

            # Check and set the device
            if torch.backends.mps.is_available():
                self.device = torch.device("mps")  # Use MPS if available
            elif torch.cuda.is_available():
                self.device = torch.device(
                    "cuda"
                )  # Use CUDA if MPS is unavailable but CUDA is
            else:
                self.device = torch.device(
                    "cpu"
                )  # Use CPU if neither MPS nor CUDA is available

        # Load the model onto the specified device
        if self.model:
            self.model.to(self.device)
        else:
            raise ValueError("Unsupported model format. Use '.yaml' or '.pt'")

    def train_model(self, data_config: str, epochs: int, optimizer: str) -> None:
        """
        Trains the YOLO model using the specified data configuration and for a number of epochs.

        Args:
            data_config (str): The path to the data configuration file.
            epochs (int): The number of training epochs.
            batch_size (int): The batch size for training and validation.
            optimizer (str): The type of optimizer to use.

        Raises:
            RuntimeError: If the model is not loaded properly before training.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Train the model
        self.model.train(data=data_config, epochs=epochs, batch=self.batch_size, optimizer=optimizer)

    def validate_model(self) -> Any:
        """
        Validates the YOLO model on the validation dataset.

        Args:
            batch_size (int): The batch size for training and validation.

        Returns:
            The validation results.

        Raises:
            RuntimeError: If the model is not loaded properly before validation.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
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
            RuntimeError: If the model is not loaded properly before prediction.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Predict on an image
        return self.model(image_path)

    @staticmethod
    def predict_image_sahi(yolov8_model_path: str, image_path: str) -> Any:
        """
        Makes a prediction using the YOLO model on the specified image with SAHI post-processing.

        Args:
            image_path (str): The path to the image file for prediction.

        Returns:
            The prediction results with SAHI post-processing.

        Raises:
            RuntimeError: If the model is not loaded properly before prediction.
        """
        if yolov8_model_path is None:
            raise RuntimeError("The model is not loaded properly.")

        # Convert YOLO model to SAHI model format, adjust according to your actual YOLO version
        sahi_model = AutoDetectionModel.from_pretrained(
            model_type='yolov8',
            model_path=yolov8_model_path,
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
            overlap_width_ratio=0.2
        )

        # Visualise the prediction results
        result.export_visuals(export_dir="./")

        # Access the object prediction list
        object_prediction_list = result.object_prediction_list

        # Return the SAHI formatted results
        return object_prediction_list

    def export_model(self, export_format: str = "onnx") -> str:
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
            raise RuntimeError("The model is not loaded properly.")
        # Export the model to the desired format
        return self.model.export(format=export_format)

    def save_model(self, save_path: str) -> None:
        """
        Saves the YOLO model to a .pt file.

        Args:
            save_path (str): The path to save the .pt model file.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Save the model to the specified path
        torch.save(self.model.state_dict(), save_path)

    def cross_validate_model(self, data_config: str, epochs: int, optimizer: str, n_splits: int = 5) -> None:
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
            raise RuntimeError("The model is not loaded properly.")

        # Load the data
        dataset_path = os.path.join(os.path.dirname(data_config))
        images_path = os.path.join(dataset_path, "images")
        labels_path = os.path.join(dataset_path, "labels")

        # List all image files
        image_files = [f for f in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, f))]
        kf = KFold(n_splits=n_splits)

        fold = 1
        for train_index, val_index in kf.split(image_files):
            train_images = [image_files[i] for i in train_index]
            val_images = [image_files[i] for i in val_index]

            # Create temporary directories for training and validation sets
            temp_train_dir = os.path.join(dataset_path, "train")
            temp_val_dir = os.path.join(dataset_path, "val")

            os.makedirs(temp_train_dir, exist_ok=True)
            os.makedirs(temp_val_dir, exist_ok=True)

            os.makedirs(os.path.join(temp_train_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(temp_train_dir, "labels"), exist_ok=True)
            os.makedirs(os.path.join(temp_val_dir, "images"), exist_ok=True)
            os.makedirs(os.path.join(temp_val_dir, "labels"), exist_ok=True)

            # Copy files to the temporary directories
            for image in train_images:
                shutil.copy(os.path.join(images_path, image), os.path.join(temp_train_dir, "images", image))
                shutil.copy(os.path.join(labels_path, image.replace(".jpg", ".txt").replace(".png", ".txt")), 
                            os.path.join(temp_train_dir, "labels", image.replace(".jpg", ".txt").replace(".png", ".txt")))

            for image in val_images:
                shutil.copy(os.path.join(images_path, image), os.path.join(temp_val_dir, "images", image))
                shutil.copy(os.path.join(labels_path, image.replace(".jpg", ".txt").replace(".png", ".txt")), 
                            os.path.join(temp_val_dir, "labels", image.replace(".jpg", ".txt").replace(".png", ".txt")))

            # Update data_config file for this fold
            with open(data_config, 'r') as file:
                data_yaml = file.read()

            data_yaml = data_yaml.replace('dataset/train/images', temp_train_dir + '/images')
            data_yaml = data_yaml.replace('dataset/valid/images', temp_val_dir + '/images')

            temp_data_config = os.path.join(dataset_path, f'data_fold{fold}.yaml')
            with open(temp_data_config, 'w') as file:
                file.write(data_yaml)

            print(f"Training fold {fold}/{n_splits}")
            self.train_model(data_config=temp_data_config, epochs=epochs, optimizer=optimizer)
            metrics = self.validate_model()
            print(f"Validation metrics for fold {fold}:", metrics)

            # Clean up temporary directories
            shutil.rmtree(temp_train_dir)
            shutil.rmtree(temp_val_dir)
            os.remove(temp_data_config)

            fold += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Handle YOLO model training, validation, prediction, and exporting."
    )

    parser.add_argument(
        "--data_config",
        type=str,
        default="dataset/data.yaml",
        help="Path to the data configuration file",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="./../../models/pt/best_yolov8x.pt",
        help="Name or path of the YOLO model file",
    )
    parser.add_argument(
        "--export_format",
        type=str,
        default="onnx",
        help="Format to export the model to",
    )
    parser.add_argument(
        "--onnx_path",
        type=str,
        default=None,
        help="Path to save the exported ONNX model",
    )
    parser.add_argument(
        "--pt_path",
        type=str,
        default="model.pt",
        help="Path to save the trained model in .pt format",
    )
    parser.add_argument(
        "--sahi_image_path",
        type=str,
        default="../../assets/IMG_1091.PNG",
        help="Path to the image file for SAHI prediction",
    )

    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=-1, 
        help="Batch size for training and validation"
    )

    parser.add_argument(
        "--optimizer", 
        type=str, 
        default="auto", 
        help="Type of optimizer to use"
    )

    parser.add_argument(
        "--cross_validate", 
        action='store_true', 
        help="Perform cross-validation"
    )

    parser.add_argument(
        "--n_splits", 
        type=int, 
        default=5, 
        help="Number of folds for cross-validation"
    )

    args = parser.parse_args()

    handler = YOLOModelHandler(args.model_name, args.batch_size)

    try:
        if args.cross_validate:
            handler.cross_validate_model(data_config=args.data_config, epochs=args.epochs, optimizer=args.optimizer, n_splits=args.n_splits)
        else:
            handler.train_model(data_config=args.data_config, epochs=args.epochs, optimizer=args.optimizer)
            metrics = handler.validate_model()
            print("Validation metrics:", metrics)
        
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

    # Predict on an image
    # results = handler.predict_image("https://ultralytics.com/images/bus.jpg")

    # SAHI Prediction
    # sahi_result = handler.predict_image_sahi(args.model_name, args.sahi_image_path)
    # print("SAHI Prediction Results:", sahi_result)

    # Example command to run the script
    '''
    python train.py --data_config=dataset/data.yaml --epochs=100 --model_name=../../models/pt/best_yolov8x.pt --batch_size=16 --optimizer=auto --cross_validate --n_splits=5
    '''
