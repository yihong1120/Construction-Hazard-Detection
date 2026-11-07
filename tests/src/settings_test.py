from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from examples.auth.config import Settings
from examples.deployment_registry.schemas import MAX_REGISTRY_TTL_SECONDS
from src.settings.inference import ApiInferenceSettings
from src.settings.inference import WorkerSettings
from src.settings.media import EncoderSettings
from src.settings.media import MediaPublishSettings
from src.settings.media import PlaybackSettings


class SettingsTest(unittest.TestCase):
    def test_database_url_is_read_for_each_settings_instance(self) -> None:
        with patch.dict(
            os.environ,
            {'DATABASE_URL': 'sqlite+aiosqlite:///new'},
        ):
            self.assertEqual(
                Settings().sqlalchemy_database_uri,
                'sqlite+aiosqlite:///new',
            )

    def test_provider_component_ids_and_explicit_override(self) -> None:
        with patch.dict(
            os.environ,
            {
                'GOOGLE_IOS_CLIENT_ID': 'ios',
                'GOOGLE_ANDROID_CLIENT_ID': 'android',
            },
            clear=True,
        ):
            self.assertEqual(Settings().google_client_ids, 'ios,android')
            os.environ['GOOGLE_CLIENT_IDS'] = 'explicit'
            self.assertEqual(Settings().google_client_ids, 'explicit')

    def test_url_normalisation_and_registry_default(self) -> None:
        with patch.dict(
            os.environ,
            {
                'OIDC_ISSUER_URL': ' https://id.test/realms/app/ ',
                'KEYCLOAK_ADMIN_BASE_URL': ' https://id.test/ ',
            },
            clear=True,
        ):
            settings = Settings()
            self.assertEqual(
                settings.oidc_issuer_url,
                'https://id.test/realms/app',
            )
            self.assertEqual(
                settings.resolved_keycloak_admin_base_url,
                'https://id.test',
            )
            self.assertEqual(
                settings.deployment_registry_ttl_seconds,
                MAX_REGISTRY_TTL_SECONDS,
            )

    def test_playback_clean_ttl_inherits_overlay_until_overridden(
        self,
    ) -> None:
        with patch.dict(
            os.environ,
            {'MEDIA_OVERLAY_DEMAND_TTL_SECONDS': '120'},
            clear=True,
        ):
            self.assertEqual(PlaybackSettings().clean_demand_ttl_seconds, 120)
            os.environ['MEDIA_CLEAN_DEMAND_TTL_SECONDS'] = '45'
            self.assertEqual(PlaybackSettings().clean_demand_ttl_seconds, 45)

    def test_inference_device_alias_priority(self) -> None:
        with patch.dict(os.environ, {'YOLO_DEVICE': ' CPU '}, clear=True):
            self.assertEqual(ApiInferenceSettings().device, 'cpu')
            os.environ['YOLO_INFERENCE_DEVICE'] = ' CUDA:1 '
            self.assertEqual(ApiInferenceSettings().device, 'cuda:1')

    @pytest.mark.ml
    def test_media_rates_have_one_default_and_are_captured_by_factory(self):
        from src.media_publish_config import MediaPublisherFactory
        from unittest.mock import MagicMock

        with patch.dict(os.environ, {}, clear=True):
            settings = MediaPublishSettings()
            self.assertEqual(settings.fps, EncoderSettings().fps)
            self.assertEqual(settings.rendition_fps('preview', 29.97), 29.97)
            factory = MediaPublisherFactory(settings)
            with patch.dict(os.environ, {'MEDIA_PUBLISH_FPS': '7'}):
                publisher = MagicMock()
                factory.create(
                    'rtsp://example/path',
                    rendition='detail',
                    publisher_type=publisher,
                )
                self.assertEqual(publisher.call_args.kwargs['fps'], 30)

    def test_invalid_limits_fail_before_runtime_work(self) -> None:
        for cls, key in [
            (PlaybackSettings, 'STREAM_PLAYBACK_SESSION_TTL_SECONDS'),
            (WorkerSettings, 'YOLO_WORKER_BATCH_SIZE'),
            (ApiInferenceSettings, 'MAX_LOADED_MODELS'),
            (Settings, 'OIDC_JWKS_TIMEOUT_SECONDS'),
        ]:
            with self.subTest(key=key), patch.dict(
                os.environ,
                {key: '0'},
                clear=True,
            ):
                with self.assertRaises(ValidationError):
                    cls()
