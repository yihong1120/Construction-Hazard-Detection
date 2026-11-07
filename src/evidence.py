from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from datetime import datetime
from datetime import timezone

from pydantic import TypeAdapter


@dataclass(frozen=True, kw_only=True)
class EvidenceMetadata:
    site: str
    stream_name: str
    detection_time: datetime | None = None
    site_id: int | None = None
    event_id: str | None = None
    warnings_json: str | None = None
    detections_json: str | None = None
    cone_polygon_json: str | None = None
    pole_polygon_json: str | None = None
    model_id: str | None = None
    model_version: str | None = None

    def __post_init__(self):
        stamp = self.detection_time
        if stamp is not None:
            stamp = (
                stamp.replace(tzinfo=timezone.utc)
                if stamp.tzinfo is None
                else stamp
            )
            object.__setattr__(
                self,
                'detection_time',
                stamp.astimezone(timezone.utc),
            )

    def as_form(self) -> dict[str, str]:
        values = EVIDENCE_METADATA.dump_python(self, mode='json')
        result = {
            key: str(value)
            for key, value in values.items()
            if value is not None
        }
        if self.detection_time is not None:
            result['detection_time'] = self.detection_time.isoformat()
        return result

    @classmethod
    def from_legacy_payload(
        cls,
        payload: Mapping[str, object],
    ) -> EvidenceMetadata:
        values = {
            field.name: payload[field.name]
            for field in fields(cls)
            if field.name in payload
        }
        for old, new in (
            ('warnings', 'warnings_json'),
            ('detections', 'detections_json'),
            ('cone_polygon', 'cone_polygon_json'),
            ('pole_polygon', 'pole_polygon_json'),
        ):
            if old in payload and payload[old] is not None:
                values[new] = json.dumps(payload[old])
        return EVIDENCE_METADATA.validate_python(values)


EVIDENCE_METADATA = TypeAdapter(EvidenceMetadata)
