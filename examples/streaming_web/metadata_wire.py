from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cached_property
from typing import Final

from examples.streaming_web.schemas import FrameOutData

_json_compact_separators: Final[tuple[str, str]] = (',', ':')


def encode_sse_event(
    payload: Mapping[str, object],
    event_type: str = 'metadata',
) -> bytes:
    """Encode one Server-Sent Events message with its optional event ID."""
    event_id = str(payload.get('id') or '')
    lines = []
    if event_id:
        lines.append(f'id: {event_id}')
    lines.append(f'event: {event_type}')
    lines.append(
        'data: ' + json.dumps(payload, separators=_json_compact_separators),
    )
    return ('\n'.join(lines) + '\n\n').encode('utf-8')


def build_metadata_payload(record: FrameOutData) -> dict[str, object]:
    """Project a Redis record onto the public frame metadata payload."""
    payload: dict[str, object] = {'has_warning': record['has_warning']}
    if 'overlay' in record:
        payload['overlay'] = record['overlay']
    return payload


@dataclass(frozen=True)
class MetadataFrame:
    """Share a frame's wire data among subscribers in the local event loop."""

    record: FrameOutData

    @cached_property
    def sse(self) -> bytes:
        payload = build_metadata_payload(self.record)
        payload['id'] = str(self.record['id'])
        return encode_sse_event(payload)

    @cached_property
    def websocket(self) -> str:
        # Match the compact UTF-8 text frame emitted by Starlette send_json.
        return json.dumps(
            build_metadata_payload(self.record),
            ensure_ascii=False,
            separators=_json_compact_separators,
        )
