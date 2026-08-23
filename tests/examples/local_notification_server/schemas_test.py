from __future__ import annotations

import unittest

from pydantic import ValidationError

from examples.local_notification_server.schemas import (
    DeviceRegistrationRequest,
)
from examples.local_notification_server.schemas import NotificationList
from examples.local_notification_server.schemas import NotificationOut
from examples.local_notification_server.schemas import (
    SiteNotificationPreferenceUpdateRequest,
)
from examples.local_notification_server.schemas import SiteNotifyRequest


class TestDeviceRegistrationRequest(unittest.TestCase):
    """Unit tests for the device-registration schema."""

    def test_valid_data(self) -> None:
        """Test the strict device-registration request schema."""
        data = {
            'device_token': 'abc123',
            'device_lang': 'en-GB',
            'platform': 'web',
        }
        token_request = DeviceRegistrationRequest(**data)
        self.assertEqual(token_request.device_token, 'abc123')

    def test_rejects_legacy_or_extra_fields(self) -> None:
        """Reject fields outside the explicit device-registration contract."""
        with self.assertRaises(ValidationError):
            DeviceRegistrationRequest(
                device_token='abc123',
                device_lang='en-GB',
                platform='web',
                user_id=123,
            )


class TestSiteNotifyRequest(unittest.TestCase):
    """Unit tests for the SiteNotifyRequest schema."""

    def test_valid_data_without_image(self) -> None:
        """Test valid input data without an image path."""
        data = {
            'site': 'MySite',
            'stream_name': 'Hello',
            'body': {
                'warning_no_safety_vest': {},
            },
            'type': 'site_alert',
            'title': 'Safety alert',
            'deep_link': '/sites/my-site',
            'metadata': {},
        }
        site_notify_request = SiteNotifyRequest(**data)
        self.assertEqual(site_notify_request.site, 'MySite')
        self.assertEqual(site_notify_request.stream_name, 'Hello')
        self.assertEqual(
            site_notify_request.body,
            {
                'warning_no_safety_vest': {},
            },
        )
        self.assertIsNone(site_notify_request.image_path)

    def test_valid_data_with_image(self) -> None:
        """Test valid input data including an image path and integer-based
        body."""
        data = {
            'site': 'AnotherSite',
            'stream_name': 'Title',
            'body': {
                'warning_no_hardhat': {'count': 123},
            },
            'image_path': 'https://example.com/image.png',
            'type': 'violation',
            'title': 'Violation alert',
            'deep_link': '/violations',
            'metadata': {},
        }
        site_notify_request = SiteNotifyRequest(**data)
        self.assertEqual(site_notify_request.site, 'AnotherSite')
        self.assertEqual(site_notify_request.stream_name, 'Title')
        self.assertEqual(
            site_notify_request.body,
            {'warning_no_hardhat': {'count': 123}},
        )
        self.assertEqual(
            site_notify_request.image_path,
            'https://example.com/image.png',
        )

    def test_valid_data_with_warning_metadata_lists(self) -> None:
        """Warning payloads may include bbox and track-id metadata."""
        data = {
            'site': 'AnotherSite',
            'stream_name': 'Title',
            'body': {
                'warning_close_to_machinery': {
                    'count': 1,
                    'person_bboxes': [[10, 20, 30, 40]],
                    'person_track_ids': ['42'],
                },
            },
            'type': 'violation',
            'title': 'Violation alert',
            'deep_link': '/violations',
            'metadata': {},
        }

        site_notify_request = SiteNotifyRequest(**data)

        self.assertEqual(
            site_notify_request.body['warning_close_to_machinery'][
                'person_bboxes'
            ],
            [[10, 20, 30, 40]],
        )

    def test_missing_site(self) -> None:
        """Test validation error when the 'site' field is missing."""
        data = {
            'stream_name': 'TestStream',
            'body': {
                'warning': {},
            },
        }
        with self.assertRaises(ValidationError) as context:
            SiteNotifyRequest(**data)
        self.assertIn('site', str(context.exception))

    def test_missing_stream_name(self) -> None:
        """Test validation error when the 'stream_name' field is missing."""
        data = {
            'site': 'MySite',
            'body': {
                'warning': {},
            },
        }
        with self.assertRaises(ValidationError) as context:
            SiteNotifyRequest(**data)
        self.assertIn('stream_name', str(context.exception))

    def test_missing_body(self) -> None:
        """Test validation error when the 'body' field is missing."""
        data = {
            'site': 'MySite',
            'stream_name': 'NoBody',
        }
        with self.assertRaises(ValidationError) as context:
            SiteNotifyRequest(**data)
        self.assertIn('body', str(context.exception))

    def test_extra_fields(self) -> None:
        """Test that extra fields in the input data are ignored."""
        data = {
            'site': 'ExtraSite',
            'stream_name': 'Test',
            'body': {
                'warning_no_hardhat': {'count': 123},
            },
            'type': 'violation',
            'title': 'Violation alert',
            'deep_link': '/violations',
            'metadata': {},
            'extra_field': 'should_not_fail',
        }
        site_notify_request = SiteNotifyRequest(**data)
        self.assertEqual(site_notify_request.site, 'ExtraSite')
        self.assertEqual(site_notify_request.stream_name, 'Test')
        self.assertEqual(
            site_notify_request.body,
            {
                'warning_no_hardhat': {'count': 123},
            },
        )
        # By default, Pydantic ignores unexpected fields
        self.assertFalse(hasattr(site_notify_request, 'extra_field'))

    def test_unknown_warning_key_is_rejected(self) -> None:
        """Notification payloads must contain translatable warning keys."""
        with self.assertRaises(ValidationError):
            SiteNotifyRequest(
                site='MySite',
                stream_name='Cam1',
                body={'unknown_warning': {}},
            )

    def test_notification_content_fields_are_required(self) -> None:
        """Notification producers must provide all display and route fields."""
        with self.assertRaises(ValidationError) as context:
            SiteNotifyRequest(
                site='MySite',
                stream_name='Cam1',
                body={'warning_no_hardhat': {'count': 1}},
            )
        self.assertIn('type', str(context.exception))
        self.assertIn('title', str(context.exception))
        self.assertIn('deep_link', str(context.exception))
        self.assertIn('metadata', str(context.exception))

    def test_site_preference_payload_rejects_an_empty_list(self) -> None:
        """A preference replacement must contain at least one site value."""
        with self.assertRaises(ValidationError):
            SiteNotificationPreferenceUpdateRequest(preferences=[])

    def test_notification_deep_link_fields(self) -> None:
        """Site notifications accept notification-center metadata."""
        data = {
            'site': 'MySite',
            'stream_name': 'Cam1',
            'body': {'warning_no_hardhat': {'count': 1}},
            'type': 'violation',
            'title': 'Violation alert',
            'deep_link': '/violations?violation_id=9',
            'metadata': {'violation_id': 9},
        }

        req = SiteNotifyRequest(**data)

        self.assertEqual(req.notification_type, 'violation')
        self.assertEqual(req.title, 'Violation alert')
        self.assertEqual(req.deep_link, '/violations?violation_id=9')
        self.assertEqual(req.metadata, {'violation_id': 9})


class TestNotificationSchemas(unittest.TestCase):
    """Unit tests for notification-center response schemas."""

    def test_notification_list_schema(self) -> None:
        """Notification list exposes an optional keyset cursor and items."""
        from datetime import datetime
        from datetime import timezone

        item = NotificationOut(
            id=1,
            type='violation',
            title='Alert',
            body='Body',
            deep_link='/violations?violation_id=1',
            is_read=False,
            created_at=datetime.now(timezone.utc),
            metadata={'violation_id': 1},
        )

        result = NotificationList(
            items=[item],
            next_cursor='eyJpZCI6MX0',
        )

        self.assertEqual(result.next_cursor, 'eyJpZCI6MX0')
        self.assertEqual(result.items[0].metadata['violation_id'], 1)


if __name__ == '__main__':
    unittest.main()
