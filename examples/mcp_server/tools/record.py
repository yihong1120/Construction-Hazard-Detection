from __future__ import annotations

import base64
import binascii
import logging
from datetime import datetime
from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import ValidationError

from examples.mcp_server.config import get_env_int
from examples.violation_records.schemas import ViolationDetectionRow
from src.violation_sender import ViolationSender

_MAX_IMAGE_BYTES = max(
    1, get_env_int(
        'MCP_MAX_REMOTE_IMAGE_BYTES', 10 * 1024**2,
    ),
)


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


class RecordTools:
    """Upload real records through the pooled, authenticated backend client."""

    def __init__(self) -> None:
        self.logger = logging.getLogger(__name__)
        self._violation_sender: ViolationSender | None = None

    async def send_violation(
        self,
        image_base64: str,
        detections: list[ViolationDetectionRow],
        warnings: dict[str, dict[str, Any]],
        site: str,
        stream_name: str,
        model_id: str,
        model_version: str,
        timestamp: str | datetime | None = None,
        cone_polygon: list | None = None,
        pole_polygon: list | None = None,
    ) -> dict:
        """Validate all inputs before authentication or an upload begins."""
        try:
            upload = ViolationUpload(
                image_base64=image_base64, detections=detections,
                warnings=warnings, site=site, stream_name=stream_name,
                model_id=model_id, model_version=model_version,
                timestamp=timestamp, cone_polygon=cone_polygon,
                pole_polygon=pole_polygon,
            )
            encoded = upload.image_base64
            if encoded.startswith('data:'):
                encoded = encoded.split(',', 1)[1]
            if len(encoded) > ((_MAX_IMAGE_BYTES + 2) // 3) * 4:
                raise ValueError('Image exceeds size limit')
            image_bytes = base64.b64decode(encoded, validate=True)
            if not image_bytes or len(image_bytes) > _MAX_IMAGE_BYTES:
                raise ValueError('Empty or oversized image')
        except (ValidationError, ValueError, binascii.Error, IndexError):
            return {
                'success': False,
                'record_id': None,
                'message': (
                    'Invalid upload: check image, source, '
                    'model snapshot and timestamp'
                ),
            }

        try:
            if self._violation_sender is None:
                self._violation_sender = ViolationSender()
            record_id = await self._violation_sender.send_violation(
                site=upload.site,
                stream_name=upload.stream_name,
                image_bytes=image_bytes,
                detection_time=upload.timestamp,
                warnings=upload.warnings,
                detections=upload.detections,
                cone_polygon=upload.cone_polygon,
                pole_polygon=upload.pole_polygon,
                model_id=upload.model_id,
                model_version=upload.model_version,
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
                    'Violation upload failed; '
                    'check backend permissions and configuration'
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
                results.append({
                    'success': False, 'record_id': None,
                    'message': 'Invalid violation upload fields',
                })
                continue
            results.append(await self.send_violation(**upload.model_dump()))
        successful = sum(bool(result['success']) for result in results)
        return {
            'success': successful == len(results), 'total': len(results),
            'successful': successful, 'failed': len(results) - successful,
            'results': results,
        }

    async def close(self) -> None:
        """Release the upload connection pool at MCP shutdown."""
        if self._violation_sender is not None:
            await self._violation_sender.close()
            self._violation_sender = None
