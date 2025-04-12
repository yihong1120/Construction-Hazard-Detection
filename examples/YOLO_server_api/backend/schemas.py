from __future__ import annotations

from datetime import datetime

from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from pydantic import BaseModel


class DetectionRequest(BaseModel):
    model: str
    image: UploadFile

    @classmethod
    def as_form(
        cls,
        model: str = Form(...),
        image: UploadFile = File(...),
    ) -> DetectionRequest:
        return cls(model=model, image=image)


class ModelFileUpdate(BaseModel):
    """
    Represents the data required to update a model file.

    Attributes:
        model (str): The model identifier (e.g., 'yolo11n').
        file (UploadFile): The uploaded file (e.g., .pt file).
    """

    model: str
    file: UploadFile

    @classmethod
    def as_form(
        cls,
        model: str = Form(...),
        file: UploadFile = File(...),
    ) -> ModelFileUpdate:
        """
        Enables FastAPI to handle this model as FormData.

        Args:
            model (str): The model name.
            file (UploadFile): The uploaded model file.

        Returns:
            ModelFileUpdate:
                An instance of this class populated with the form data.
        """
        return cls(model=model, file=file)


class UpdateModelRequest(BaseModel):
    """
    Represents the data required to retrieve a new model file.

    Attributes:
        model (str):
            The model identifier to check.
        last_update_time (str):
            ISO 8601 string representing the last known update time.
    """
    model: str
    last_update_time: str

    def last_update_as_datetime(self) -> datetime | None:
        """
        Converts `last_update_time` to a datetime object.

        Returns:
            Optional[datetime]: A datetime object or None if parsing fails.
        """
        try:
            return datetime.fromisoformat(self.last_update_time)
        except ValueError:
            return None
