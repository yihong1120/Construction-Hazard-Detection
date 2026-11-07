from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from examples.YOLO_server_api.model_registry import ModelEntry
from examples.YOLO_server_api.overlay_response import build_overlay_response
from src.native_work import NativeWorkPool


@dataclass(frozen=True)
class OverlayWork:
    rows: list[list[float | int]]
    entry: ModelEntry
    image_size: tuple[int, int]
    compute_regions: bool
    view: Literal['objects', 'regions']

    def render(self):
        return build_overlay_response(
            self.rows,
            self.entry,
            self.image_size,
            self.compute_regions,
            self.view,
        )


class OverlayRuntime(NativeWorkPool):
    def __init__(self, concurrency: int = 2, capacity: int = 8):
        super().__init__(
            concurrency,
            capacity,
            code='OVERLAY_BUSY',
            name='overlay',
        )

    async def render(self, work: OverlayWork):
        return await self.run(work.render)
