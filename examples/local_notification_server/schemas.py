from __future__ import annotations

from pydantic import BaseModel


class TokenRequest(BaseModel):
    """
    Schema model for registering a device token for push notifications.

    Attributes:
        user_id (int): Unique identifier of the user.
        device_token (str): Device token used for sending notifications.
        device_lang (Optional[str]):
            Preferred device language, defaulting to British English ("en-GB").
    """

    user_id: int
    device_token: str
    device_lang: str | None = 'en-GB'


class SiteNotifyRequest(BaseModel):
    """
    Schema model for notifying a site-based safety violation or event.

    Attributes:
        site (str): Name or identifier of the site.
        stream_name (str): Name of the video stream associated with the event.
        body (dict[str, dict[str, int]]):
            Detection result data structured by category and item.
        image_path (Optional[str]):
            URL or path to the image related to the notification.
        violation_id (Optional[int]):
            Unique identifier for the violation, if applicable.
    """

    site: str
    stream_name: str
    body: dict[str, dict[str, int]]
    image_path: str | None = None
    violation_id: int | None = None
