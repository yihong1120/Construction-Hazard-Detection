from __future__ import annotations

import hashlib
import json
from datetime import timezone

from examples.shared.evidence_metadata import validate_metadata
from examples.violation_records.evidence import ViolationEvidence


def normalized_evidence(evidence: ViolationEvidence) -> ViolationEvidence:
    return validate_metadata(evidence).source


def _numbers(value):
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, list):
        return [_numbers(item) for item in value]
    if isinstance(value, dict):
        return {key: _numbers(item) for key, item in value.items()}
    return value


def evidence_fingerprint(
    evidence: ViolationEvidence,
    image_sha256: str,
) -> str:
    payload: dict[str, object] = {
        'version': 1,
        'site_id': evidence.site_id,
        'camera': evidence.stream_name,
        'model_id': evidence.model_id,
        'model_version': evidence.model_version,
        'detection_time': (
            evidence.detection_time.astimezone(timezone.utc).isoformat()
            if evidence.detection_time is not None
            else None
        ),
        'image_sha256': image_sha256,
    }
    for name, empty in (
        ('warnings_json', {}),
        ('detections_json', []),
        ('cone_polygon_json', []),
        ('pole_polygon_json', []),
    ):
        value = getattr(evidence, name)
        payload[name] = (
            _numbers(json.loads(value)) if value is not None else empty
        )
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(',', ':'),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()
