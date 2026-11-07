from __future__ import annotations

import base64
import binascii
import json
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import ValidationError

from examples.mcp_server.settings import McpImageSettings
from examples.shared.evidence_metadata import validate_metadata
from examples.violation_records.schemas import ViolationDetectionRow
from src.detection_rows import TrackedDetectionRows
from src.evidence import EvidenceMetadata
from src.violation_sender import ViolationSender

_MAX_IMAGE_BYTES = McpImageSettings().max_remote_image_bytes


class ViolationUpload(BaseModel):
    """Required model snapshot and source identity for a real upload."""

    model_config = ConfigDict(extra='forbid')
    image_base64: str = Field(min_length=1)
    detections: list[ViolationDetectionRow]
    warnings: dict[str, dict[str, Any]]
    site: str = Field(min_length=1)
    stream_name: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    model_version: str = Field(min_length=1)
    timestamp: datetime | None = None
    cone_polygon: list | None = None
    pole_polygon: list | None = None

    @field_validator('detections')
    @classmethod
    def validate_detections(cls, value):
        return TrackedDetectionRows.model_validate(value).root

    def image_bytes(self) -> bytes:
        encoded = self.image_base64
        if encoded.startswith('data:'):
            encoded = encoded.split(',', 1)[1]
        if len(encoded) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4:
            raise ValueError('Image exceeds size limit')
        image = base64.b64decode(encoded, validate=True)
        if not image or len(image) > _MAX_IMAGE_BYTES:
            raise ValueError('Empty or oversized image')
        return image

    def evidence_metadata(self) -> EvidenceMetadata:
        return validate_metadata(
            EvidenceMetadata(
                site=self.site,
                stream_name=self.stream_name,
                detection_time=self.timestamp,
                model_id=self.model_id,
                model_version=self.model_version,
                warnings_json=json.dumps(self.warnings),
                detections_json=json.dumps(self.detections),
                cone_polygon_json=(
                    json.dumps(self.cone_polygon)
                    if self.cone_polygon is not None
                    else None
                ),
                pole_polygon_json=(
                    json.dumps(self.pole_polygon)
                    if self.pole_polygon is not None
                    else None
                ),
            ),
        ).source


class RecordTools:
    """Upload real records through the pooled, authenticated backend client."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._violation_sender: ViolationSender | None = None

    async def send_violation(self, upload: ViolationUpload) -> dict:
        """Decode validated evidence once before authentication or upload."""
        try:
            metadata = upload.evidence_metadata()
            image_bytes = upload.image_bytes()
        except (ValueError, binascii.Error, IndexError):
            return {
                'success': False,
                'record_id': None,
                'message': (
                    'Invalid upload: check image, source, model snapshot '
                    'and timestamp'
                ),
            }

        try:
            if self._violation_sender is None:
                self._violation_sender = ViolationSender()
            record_id = await self._violation_sender.send_violation(
                metadata,
                image_bytes,
            )
            return {'success': record_id is not None, 'record_id': record_id}
        except Exception as exc:
            self.logger.error(
                'Violation upload failed (%s)',
                type(exc).__name__,
            )
            return {
                'success': False,
                'record_id': None,
                'message': (
                    'Violation upload failed; check backend permissions '
                    'and configuration'
                ),
            }

    async def batch_send_violations(self, violations: list[dict]) -> dict:
        """Send up to 100 records, preserving individual failures and order."""
        if not 1 <= len(violations) <= 100:
            raise ValueError('A batch must contain 1 to 100 records')
        results = []
        for violation in violations:
            try:
                upload = ViolationUpload.model_validate(violation)
            except ValidationError:
                results.append(
                    {
                        'success': False,
                        'record_id': None,
                        'message': 'Invalid violation upload fields',
                    },
                )
                continue
            results.append(await self.send_violation(upload))
        successful = sum(bool(result['success']) for result in results)
        return {
            'success': successful == len(results),
            'total': len(results),
            'successful': successful,
            'failed': len(results) - successful,
            'results': results,
        }

    async def close(self) -> None:
        """Release the upload connection pool at MCP shutdown."""
        if self._violation_sender is not None:
            await self._violation_sender.close()
            self._violation_sender = None
