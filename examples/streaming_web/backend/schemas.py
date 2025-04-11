from __future__ import annotations

from pydantic import BaseModel


class LabelListResponse(BaseModel):
    """
    A response model for returning a list of labels.

    Args:
        labels (List[str]): A list of labels.
    """
    labels: list[str]


class FramePostResponse(BaseModel):
    """
    A response model for frame upload status.

    Args:
        status (str):
            The status of the upload operation.
        message (str):
            A message providing additional information about the operation.
    """
    status: str
    message: str
