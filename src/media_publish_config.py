from __future__ import annotations

import math
from typing import Any
from typing import Protocol
from typing import TypedDict

from src.media_encoding import EncoderOptions
from src.media_stream_publisher import MediaStreamPublisher
from src.settings.media import MediaPublishSettings


class PreviewPublisherOptions(TypedDict):
    fps: float
    width: int
    height: int
    bitrate: str
    maxrate: str
    bufsize: str


class SourceFrameRate(Protocol):
    @property
    def source_fps(self) -> float | None: ...


class MediaPublisherFactory:
    def __init__(
        self, settings: MediaPublishSettings,
        source: SourceFrameRate | None = None,
    ) -> None:
        self.settings = settings
        self.encoding = EncoderOptions(settings)
        self.source = source

    def fps(self, rendition: str) -> float:
        return self.settings.rendition_fps(
            rendition, self.source.source_fps if self.source else None,
        )

    async def ensure(
        self, publish_url: str, *, rendition: str,
        current: MediaStreamPublisher | None,
    ) -> MediaStreamPublisher:
        """Recreate an encoder when a reconnected camera reports a new rate."""
        if current is not None:
            if math.isclose(current.fps, self.fps(rendition), rel_tol=1e-6):
                return current
            await current.close()
        return self.create(publish_url, rendition=rendition)

    def preview_options(self) -> PreviewPublisherOptions:
        settings = self.settings
        return {
            'fps': self.fps('preview'),
            'width': settings.preview_width,
            'height': settings.preview_height,
            'bitrate': settings.preview_bitrate,
            'maxrate': settings.preview_maxrate,
            'bufsize': settings.preview_bufsize,
        }

    def create(
        self,
        publish_url: str,
        *,
        rendition: str,
        publisher_type: Any = MediaStreamPublisher,
    ) -> MediaStreamPublisher:
        if rendition == 'preview':
            return publisher_type(
                publish_url=publish_url, encoding=self.encoding,
                **self.preview_options(),
            )
        if rendition == 'detail':
            return publisher_type(
                publish_url=publish_url, fps=self.fps('detail'),
                encoding=self.encoding,
            )
        raise ValueError(f'unsupported media rendition: {rendition}')
