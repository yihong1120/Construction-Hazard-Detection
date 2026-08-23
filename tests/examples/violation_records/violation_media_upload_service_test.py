from __future__ import annotations

import unittest
from datetime import datetime
from datetime import timezone
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock
from unittest.mock import patch

from fastapi import HTTPException
from fastapi import UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.db_management.schemas.auth import AccessTokenSubject
from examples.violation_records import violation_media_service
from examples.violation_records import violation_upload_service
from examples.violation_records.violation_manager import (
    EmptyViolationImageError,
)


def _credentials(username: str = 'reviewer') -> JwtAuthorizationCredentials:
    """Build credentials for media and upload service tests.

    Args:
        username: Username exposed by the JWT subject.

    Returns:
        Minimal authenticated access-token credentials.
    """
    return JwtAuthorizationCredentials(
        subject=cast(
            AccessTokenSubject,
            {
                'username': username,
                'user_id': 8,
                'role': 'admin',
                'jti': 'violation-media-test',
                'features': [],
            },
        ),
    )


class TestViolationMediaService(unittest.IsolatedAsyncioTestCase):
    """Verify authorised media and thumbnail response construction."""

    async def test_media_access_and_responses_keep_evidence_private(
        self,
    ) -> None:
        """Existing evidence is served only after record and site checks."""
        with TemporaryDirectory() as directory:
            static_dir = Path(directory)
            image_path = static_dir / 'evidence.jpg'
            image_path.write_bytes(b'image')
            thumbnail_path = static_dir / 'thumbnail.jpg'
            thumbnail_path.write_bytes(b'thumbnail')
            db = SimpleNamespace(scalar=AsyncMock(return_value=4))

            with (
                patch.object(
                    violation_media_service,
                    'STATIC_DIR',
                    static_dir,
                ),
                patch.object(
                    violation_media_service.user_service,
                    'get_cached_effective_site_names',
                    new=AsyncMock(return_value=['Site A']),
                ),
            ):
                full_path, media_type = (
                    await violation_media_service._authorise_media_access(
                        'evidence.jpg',
                        'reviewer',
                        cast(AsyncSession, db),
                    )
                )
                with (
                    patch.object(
                        violation_media_service,
                        '_authorise_media_access',
                        new=AsyncMock(return_value=(image_path, media_type)),
                    ),
                    patch.object(
                        violation_media_service,
                        'ensure_thumbnail',
                        new=AsyncMock(return_value=thumbnail_path),
                    ),
                ):
                    image_response = (
                        await violation_media_service.get_violation_image(
                            'evidence.jpg',
                            cast(AsyncSession, db),
                            _credentials(),
                        )
                    )
                    thumbnail_response = (
                        await violation_media_service.get_violation_thumbnail(
                            'evidence.jpg',
                            cast(AsyncSession, db),
                            _credentials(),
                        )
                    )

        self.assertEqual(full_path, image_path)
        self.assertEqual(media_type, 'image/jpeg')
        self.assertEqual(image_response.media_type, 'image/jpeg')
        self.assertEqual(thumbnail_response.media_type, 'image/jpeg')
        self.assertEqual(
            thumbnail_response.headers['cache-control'],
            'private, max-age=86400',
        )

    async def test_media_rejects_missing_token_and_unknown_files(self) -> None:
        """Media endpoints reject absent identities and unknown evidence."""
        db = SimpleNamespace(scalar=AsyncMock())
        anonymous = _credentials('')

        with self.assertRaises(HTTPException) as missing_identity:
            await violation_media_service.get_violation_image(
                'missing.jpg',
                cast(AsyncSession, db),
                anonymous,
            )
        self.assertEqual(missing_identity.exception.status_code, 401)

        with TemporaryDirectory() as directory:
            with patch.object(
                violation_media_service,
                'STATIC_DIR',
                Path(directory),
            ):
                with self.assertRaises(HTTPException) as missing_file:
                    await violation_media_service._authorise_media_access(
                        'missing.jpg',
                        'reviewer',
                        cast(AsyncSession, db),
                    )
        self.assertEqual(missing_file.exception.status_code, 404)


class TestViolationUploadService(unittest.IsolatedAsyncioTestCase):
    """Verify upload authorisation and manager error translation."""

    async def test_upload_persists_an_authorised_violation(self) -> None:
        """Authorised uploads return the manager's persisted record ID."""
        image = UploadFile(
            filename='evidence.jpg',
            file=BytesIO(b'image'),
        )
        db = SimpleNamespace()

        with (
            patch.object(
                violation_upload_service.user_service,
                'get_cached_effective_site_names',
                new=AsyncMock(return_value=['Site A']),
            ),
            patch.object(
                violation_upload_service.violation_manager,
                'save_violation',
                new=AsyncMock(return_value=37),
            ) as save_violation,
        ):
            response = await violation_upload_service.upload_violation(
                site='Site A',
                stream_name='Camera A',
                detection_time=datetime(2026, 8, 23, tzinfo=timezone.utc),
                warnings_json=None,
                detections_json=None,
                cone_polygon_json=None,
                pole_polygon_json=None,
                image=image,
                db=cast(AsyncSession, db),
                credentials=_credentials(),
            )

        self.assertEqual(response.violation_id, 37)
        save_violation.assert_awaited_once()

    async def test_upload_denies_bad_scope_and_invalid_image_content(
        self,
    ) -> None:
        """Unauthorised sites and unreadable images become client errors."""
        image = UploadFile(filename='evidence.jpg', file=BytesIO(b'image'))
        db = SimpleNamespace()

        with patch.object(
            violation_upload_service.user_service,
            'get_cached_effective_site_names',
            new=AsyncMock(return_value=[]),
        ):
            with self.assertRaises(HTTPException) as denied:
                await violation_upload_service.upload_violation(
                    site='Site A',
                    stream_name='Camera A',
                    detection_time=None,
                    warnings_json=None,
                    detections_json=None,
                    cone_polygon_json=None,
                    pole_polygon_json=None,
                    image=image,
                    db=cast(AsyncSession, db),
                    credentials=_credentials(),
                )
        self.assertEqual(denied.exception.status_code, 403)

        with (
            patch.object(
                violation_upload_service.user_service,
                'get_cached_effective_site_names',
                new=AsyncMock(return_value=['Site A']),
            ),
            patch.object(
                violation_upload_service.violation_manager,
                'save_violation',
                new=AsyncMock(side_effect=EmptyViolationImageError()),
            ),
        ):
            with self.assertRaises(HTTPException) as invalid_image:
                await violation_upload_service.upload_violation(
                    site='Site A',
                    stream_name='Camera A',
                    detection_time=None,
                    warnings_json=None,
                    detections_json=None,
                    cone_polygon_json=None,
                    pole_polygon_json=None,
                    image=image,
                    db=cast(AsyncSession, db),
                    credentials=_credentials(),
                )
        self.assertEqual(invalid_image.exception.status_code, 400)

    async def test_upload_rejects_a_token_without_a_username(self) -> None:
        """Evidence uploads require the authenticated username claim."""
        image = UploadFile(filename='evidence.jpg', file=BytesIO(b'image'))

        with self.assertRaises(HTTPException) as invalid_token:
            await violation_upload_service.upload_violation(
                site='Site A',
                stream_name='Camera A',
                detection_time=None,
                warnings_json=None,
                detections_json=None,
                cone_polygon_json=None,
                pole_polygon_json=None,
                image=image,
                db=cast(AsyncSession, SimpleNamespace()),
                credentials=_credentials(username=''),
            )

        self.assertEqual(invalid_token.exception.status_code, 401)


if __name__ == '__main__':
    unittest.main()
