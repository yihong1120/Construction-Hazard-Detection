from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path

from src.violation_sender import ViolationSender


class RecordTools:
    """Tools for managing violation records and data persistence."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._violation_sender = None

    async def send_violation(
        self,
        image_base64: str,
        detections: list[dict],
        warning_message: str,
        timestamp: str | None = None,
        site_id: str | None = None,
        metadata: dict | None = None,
    ) -> dict:
        """Send a violation record to the database/API.

        Args:
            image_base64: Base64-encoded violation image.
            detections: List of detection objects with bbox, class, confidence.
            warning_message: Violation warning message.
            timestamp: ISO timestamp (auto-generated if not provided).
            site_id: Construction site identifier.
            metadata: Additional metadata for the violation.

        Returns:
            dict[str, Any]: Upload status and record ID.
        """
        try:
            await self._ensure_violation_sender()

            # Convert base64 to bytes
            import base64
            if ',' in image_base64:
                image_base64 = image_base64.split(',', 1)[1]
            image_bytes = base64.b64decode(image_base64)

            # Demo mode: save locally and return mock success
            if str(os.getenv('MCP_DEMO_MODE', '')).lower() in {
                '1', 'true', 'yes',
            }:
                out_dir = Path('static') / datetime.now().strftime('%Y-%m-%d')
                out_dir.mkdir(parents=True, exist_ok=True)
                import uuid
                fname = f"{uuid.uuid4()}.jpg"
                fpath = out_dir / fname
                fpath.write_bytes(image_bytes)
                mock_id = fname
                return {
                    'success': True,
                    'record_id': mock_id,
                    'image_path': str(fpath.as_posix()),
                    'message': (
                        'Demo mode enabled: saved locally and returning '
                        'mock violation ID'
                    ),
                }

            # Map parameters to ViolationSender.send_violation signature
            site = (metadata or {}).get('site') or site_id or 'default-site'
            stream_name = (metadata or {}).get(
                'stream_name',
            ) or 'default-stream'

            det_json = json.dumps(
                detections,
            ) if detections is not None else None
            warn_json = json.dumps(
                {'message': warning_message},
            ) if warning_message else None

            dt: datetime | None = None
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp)
                except Exception:
                    dt = None

            # Call existing async API
            violation_id = await self._violation_sender.send_violation(
                site=site,
                stream_name=stream_name,
                image_bytes=image_bytes,
                detection_time=dt,
                warnings_json=warn_json,
                detections_json=det_json,
                cone_polygon_json=None,
                pole_polygon_json=None,
            )

            success = violation_id is not None
            return {
                'success': success,
                'record_id': violation_id,
                'message': (
                    f"Violation record saved with ID: {violation_id}"
                    if success else 'Failed to save violation record'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to send violation record: {e}")
            return {
                'success': False,
                'record_id': None,
                'message': str(e),
            }

    async def batch_send_violations(
        self,
        violations: list[dict],
    ) -> dict:
        """Send multiple violation records in a batch.

        Args:
            violations: List of violation objects each containing
                ``image_base64``, ``detections``, ``warning_message``, etc.

        Returns:
            dict[str, Any]: Batch upload status and record IDs.
        """
        try:
            results = []

            for violation in violations:
                result = await self.send_violation(**violation)
                results.append(result)

            success_count = sum(
                1 for r in results if bool(r.get('success', False))
            )

            return {
                'success': success_count == len(violations),
                'total': len(violations),
                'successful': success_count,
                'failed': len(violations) - success_count,
                'results': results,
                'message': (
                    f"Batch upload completed: {success_count}/"
                    f"{len(violations)} successful"
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to batch send violations: {e}")
            raise

    async def backup_local_records(
        self,
        backup_path: str | None = None,
    ) -> dict:
        """Back up violation records to local storage.

        Args:
            backup_path: Path for the backup file (uses a default when
                omitted).

        Returns:
            dict[str, Any]: Backup status and file path.
        """
        try:
            await self._ensure_violation_sender()

            # Use default backup path if not provided
            if backup_path is None:  # pragma: no cover
                backup_path = (
                    self._compute_default_backup_path()
                )  # pragma: no cover

            # Create backup
            success, backed_up_count = await (
                self._violation_sender.backup_to_local(
                    backup_path,
                )
            )

            return {
                'success': success,
                'backup_path': backup_path,
                'records_count': backed_up_count,
                'message': (
                    f"Backed up {backed_up_count} records to {backup_path}"
                    if success
                    else 'Backup failed'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to backup records: {e}")
            raise

    @staticmethod
    def _compute_default_backup_path() -> str:
        """Compute the default path used for backing up violations.

        Separated for testability and to avoid long multi-line coverage gaps.
        """
        import os
        return os.path.join(
            os.path.dirname(__file__),
            '../../../static',
            (
                f"violations_backup_"
                f"{int(asyncio.get_event_loop().time())}.json"
            ),
        )  # pragma: no cover

    async def sync_pending_records(self) -> dict:
        """Synchronise pending violation records from local cache to server.

        Returns:
            dict[str, Any]: Sync status and statistics.
        """
        try:
            await self._ensure_violation_sender()

            # Fallback: not implemented in current ViolationSender
            return {
                'success': False,
                'synced_count': 0,
                'failed_count': 0,
                'message': (
                    'Sync not supported by current ViolationSender '
                    'implementation'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to sync pending records: {e}")
            raise

    async def get_upload_statistics(self) -> dict:
        """Get upload statistics and queue status.

        Returns:
            dict[str, Any]: Upload statistics and queue details.
        """
        try:
            await self._ensure_violation_sender()

            # Fallback: not implemented in current ViolationSender
            return {
                'success': True,
                'statistics': {
                    'pending': 0,
                    'sent': 0,
                    'failed': 0,
                },
                'message': (
                    'Statistics not available; using default placeholders'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to get upload statistics: {e}")
            raise

    async def clear_local_cache(self) -> dict:
        """Clear the local violation record cache.

        Returns:
            dict[str, Any]: Clear operation status.
        """
        try:
            await self._ensure_violation_sender()

            # Fallback: not implemented in current ViolationSender
            return {
                'success': False,
                'cleared_count': 0,
                'message': (
                    'Cache clearing not supported by current '
                    'ViolationSender implementation'
                ),
            }

        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
            raise

    async def _ensure_violation_sender(self) -> None:
        """Ensure the violation sender is initialised."""
        if self._violation_sender is None:
            self._violation_sender = ViolationSender()
            self.logger.info('Initialised violation sender')
