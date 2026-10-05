from __future__ import annotations

from datetime import datetime
from typing import Annotated
from typing import Literal

from fastapi import File
from fastapi import Form
from fastapi import UploadFile
from pydantic import BaseModel
from pydantic import model_validator

from examples.shared.overlay_schemas import OverlayPayload
from examples.YOLO_server_api.model_registry import ModelClass
from examples.YOLO_server_api.model_registry import Text


class DetectionModelInfo(BaseModel):
    """Public model identity, independent of its cache residency."""

    id: Text
    display_name: Text
    version: Text | None = None
    classes: list[ModelClass] | None = None

    @model_validator(mode='after')
    def validate_class_metadata(self):
        if self.classes is not None:
            if self.version is None:
                raise ValueError('Model classes require an immutable version')
            if len({c.id for c in self.classes}) != len(self.classes):
                raise ValueError('Duplicate class IDs')
            if len({c.code for c in self.classes}) != len(self.classes):
                raise ValueError('Duplicate class codes')
        return self


class DetectionModelList(BaseModel):
    """Selectable models and a default drawn from that same list."""

    default_model_id: str | None
    models: list[DetectionModelInfo]


class DetectionRequest(BaseModel):
    """Form payload for image detection."""

    model: str
    image: UploadFile
    model_version: str | None = None

    @classmethod
    def as_form(
        cls,
        model: str = Form(...),
        image: UploadFile = File(...),
        model_version: Annotated[str | None, Form()] = None,
    ) -> DetectionRequest:
        """Build a detection request from multipart form fields."""
        return cls(model=model, image=image, model_version=model_version)


class ModelFileUpdate(BaseModel):
    """Represents the data required to update a model file.

    Attributes:
        model (str): The model identifier (e.g., 'yolo26n').
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
        """Enables FastAPI to handle this model as FormData.

        Args:
            model (str): The model name.
            file (UploadFile): The uploaded model file.

        Returns:
            ModelFileUpdate:
                An instance of this class populated with the form data.
        """
        return cls(model=model, file=file)


class UpdateModelRequest(BaseModel):
    """Represents the data required to retrieve a new model file.

    Attributes:
        model (str):
            The model identifier to check.
        last_update_time (str):
            ISO 8601 string representing the last known update time.
    """

    model: str
    last_update_time: str

    def last_update_as_datetime(self) -> datetime | None:
        """Converts `last_update_time` to a datetime object.

        Returns:
            Optional[datetime]: A datetime object or None if parsing fails.
        """
        try:
            return datetime.fromisoformat(self.last_update_time)
        except ValueError:
            return None


class DetectionOverlayResponse(OverlayPayload):
    model_id: str
    model_version: str
    view: Literal['objects', 'regions']
