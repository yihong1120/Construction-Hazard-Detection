import argparse
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from typing import Any, Optional
from ultralytics import YOLO


class YOLOModelHandler:
    """Handles loading, training, validating, and predicting with YOLO models.

    Attributes:
        model_name (str): The name of the model file to be loaded.
        model (YOLO, Optional): The loaded YOLO model object.
    """

    def __init__(self, model_name: str):
        """
        Initialises the YOLOModelHandler with a specified model.

        Args:
            model_name (str): The name of the model file (either .yaml or .pt).
        """
        self.model_name: str = model_name
        self.model: Optional[YOLO] = None
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

    def train_model(self, data_config: str, epochs: int) -> None:
        """
        Trains the YOLO model using the specified data configuration and for a number of epochs.

        Args:
            data_config (str): The path to the data configuration file.
            epochs (int): The number of training epochs.

        Raises:
            RuntimeError: If the model is not loaded properly before training.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Train the model
        self.model.train(data=data_config, epochs=epochs)

    def validate_model(self) -> Any:
        """
        Validates the YOLO model on the validation dataset.

        Returns:
            The validation results.

        Raises:
            RuntimeError: If the model is not loaded properly before validation.
        """
        if self.model is None:
            raise RuntimeError("The model is not loaded properly.")
        # Evaluate model performance on the validation set
        return self.model.val()

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
        default="../../models/best_yolov8n.pt",
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

    args = parser.parse_args()

    handler = YOLOModelHandler(args.model_name)

    try:
        handler.train_model(data_config=args.data_config, epochs=args.epochs)
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