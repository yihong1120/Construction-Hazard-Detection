from __future__ import annotations

import math

from pydantic import AliasChoices
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

from src.settings.base import EnvironmentSettings
from src.settings.connections import http_endpoint


class PlaybackSettings(EnvironmentSettings):
    refresh_batch_size: int = Field(
        256,
        ge=1,
        le=2048,
        validation_alias='STREAM_PLAYBACK_REFRESH_BATCH_SIZE',
    )
    overlay_demand_ttl_seconds: int = Field(
        90,
        ge=1,
        validation_alias='MEDIA_OVERLAY_DEMAND_TTL_SECONDS',
    )
    clean_demand_ttl_seconds: int = Field(
        90,
        ge=1,
        validation_alias='MEDIA_CLEAN_DEMAND_TTL_SECONDS',
    )
    overlay_max_active_languages: int = Field(
        5,
        ge=1,
        validation_alias='MEDIA_OVERLAY_MAX_ACTIVE_LANGUAGES_PER_STREAM',
    )
    session_ttl_seconds: int = Field(
        4500,
        ge=1,
        validation_alias='STREAM_PLAYBACK_SESSION_TTL_SECONDS',
    )
    public_base_path: str = Field(
        '/hazard/api/stream-playback',
        validation_alias='STREAM_PLAYBACK_PUBLIC_BASE_PATH',
    )
    startup_wait_seconds: float = Field(
        3.0,
        ge=0,
        validation_alias='STREAM_PLAYBACK_STARTUP_WAIT_SECONDS',
    )
    refresh_interval_seconds: int = Field(
        20,
        ge=1,
        validation_alias='STREAM_PLAYBACK_SESSION_REFRESH_INTERVAL_SECONDS',
    )

    @model_validator(mode='before')
    @classmethod
    def clean_demand_default(cls, values: dict) -> dict:
        if (
            'MEDIA_CLEAN_DEMAND_TTL_SECONDS' not in values
            and 'clean_demand_ttl_seconds' not in values
        ):
            values['MEDIA_CLEAN_DEMAND_TTL_SECONDS'] = values.get(
                'MEDIA_OVERLAY_DEMAND_TTL_SECONDS',
                values.get(
                    'overlay_demand_ttl_seconds',
                    90,
                ),
            )
        return values


class EncoderSettings(EnvironmentSettings):
    ffmpeg_path: str = Field('', validation_alias='MEDIA_FFMPEG_PATH')
    ffmpeg_loglevel: str = Field(
        'error',
        validation_alias='MEDIA_FFMPEG_LOGLEVEL',
    )
    fps: float | None = Field(
        None, gt=0, allow_inf_nan=False, validation_alias='MEDIA_PUBLISH_FPS',
    )
    clean_fps: float | None = Field(
        None,
        gt=0,
        allow_inf_nan=False,
        validation_alias='MEDIA_PUBLISH_CLEAN_FPS',
    )
    keyframe_interval_seconds: float = Field(
        2.0,
        validation_alias='MEDIA_PUBLISH_KEYFRAME_INTERVAL_SECONDS',
    )
    rtp_packet_size: int = Field(
        1200,
        validation_alias='MEDIA_PUBLISH_RTP_PACKET_SIZE',
    )
    encoder: str = Field('libx264', validation_alias='MEDIA_PUBLISH_ENCODER')
    clean_encoder: str = Field(
        'h264_nvenc',
        validation_alias=AliasChoices(
            'MEDIA_PUBLISH_CLEAN_ENCODER',
            'MEDIA_RESTREAM_ENCODER',
        ),
    )
    preset: str = Field('veryfast', validation_alias='MEDIA_PUBLISH_PRESET')
    crf: int = Field(28, ge=0, le=51, validation_alias='MEDIA_PUBLISH_CRF')
    nvenc_preset: str = Field(
        'p1',
        validation_alias='MEDIA_PUBLISH_NVENC_PRESET',
    )
    nvenc_rc: str = Field('vbr', validation_alias='MEDIA_PUBLISH_NVENC_RC')
    nvenc_cq: int = Field(
        30,
        ge=0,
        le=51,
        validation_alias='MEDIA_PUBLISH_NVENC_CQ',
    )
    nvenc_bitrate: str = Field(
        '0',
        validation_alias='MEDIA_PUBLISH_NVENC_BITRATE',
    )
    nvenc_maxrate: str = Field(
        '8M',
        validation_alias='MEDIA_PUBLISH_NVENC_MAXRATE',
    )
    nvenc_bufsize: str = Field(
        '16M',
        validation_alias='MEDIA_PUBLISH_NVENC_BUFSIZE',
    )
    vaapi_bitrate: str = Field(
        '4M',
        validation_alias='MEDIA_PUBLISH_VAAPI_BITRATE',
    )
    vaapi_maxrate: str = Field(
        '8M',
        validation_alias='MEDIA_PUBLISH_VAAPI_MAXRATE',
    )
    vaapi_bufsize: str = Field(
        '16M',
        validation_alias='MEDIA_PUBLISH_VAAPI_BUFSIZE',
    )
    vaapi_rc_mode: str = Field(
        'CBR',
        validation_alias='MEDIA_PUBLISH_VAAPI_RC_MODE',
    )
    vaapi_async_depth: int = Field(
        1,
        ge=1,
        validation_alias='MEDIA_PUBLISH_VAAPI_ASYNC_DEPTH',
    )
    vaapi_device: str = Field(
        '',
        validation_alias='MEDIA_PUBLISH_VAAPI_DEVICE',
    )

    @field_validator('fps', 'clean_fps', mode='before')
    @classmethod
    def automatic_frame_rate(cls, value: object):
        if str(value).strip().lower() in {'', 'auto', 'none'}:
            return None
        return value

    @field_validator('keyframe_interval_seconds', mode='before')
    @classmethod
    def bounded_keyframe_interval(cls, value: object) -> float:
        try:
            return max(0.1, float(str(value)))
        except ValueError:
            return 2.0

    @field_validator('rtp_packet_size', mode='before')
    @classmethod
    def bounded_rtp_packet_size(cls, value: object) -> int:
        try:
            return min(1400, max(576, int(str(value))))
        except ValueError:
            return 1200


class OverlaySettings(EnvironmentSettings):
    model_config = {'env_prefix': 'STREAMING_OVERLAY_'}
    min_confidence: float = Field(0.4, ge=0, le=1)
    box_thickness: int = Field(2, ge=1)
    mode: str = 'backend'
    max_labels: int = Field(40, ge=0)
    draw_labels: bool = True
    label_warnings_only: bool = False
    draw_warning_summary: bool = True
    draw_warning_status: bool = False
    max_warning_summary_items: int = Field(5, ge=1)
    parse_cache_size: int = Field(256, ge=16)


class OverlayReadinessSettings(EnvironmentSettings):
    ready_ttl_seconds: int = Field(
        15,
        ge=5,
        validation_alias='MEDIA_OVERLAY_READY_TTL_SECONDS',
    )

    @field_validator('ready_ttl_seconds', mode='before')
    @classmethod
    def ready_ttl(cls, value):
        return max(5, int(float(str(value))))


class MediaPublishSettings(EncoderSettings, OverlayReadinessSettings):
    source_fps: float | None = Field(
        None, gt=0, allow_inf_nan=False,
        validation_alias='MEDIA_PUBLISH_SOURCE_FPS',
    )
    publish_base: str = Field(
        'rtsp://media-server:8554',
        validation_alias='MEDIA_PUBLISH_RTSP_BASE_URL',
    )
    clean: bool = Field(True, validation_alias='MEDIA_PUBLISH_CLEAN_STREAM')
    overlay: bool = Field(
        True, validation_alias='MEDIA_PUBLISH_ANNOTATED_STREAM',
    )
    restream_clean: bool = Field(
        True, validation_alias='MEDIA_PUBLISH_CLEAN_SOURCE_RESTREAM',
    )
    demand_cache_seconds: float = Field(
        0.5, ge=0, validation_alias='MEDIA_DEMAND_CACHE_SECONDS',
    )
    ready_grace_seconds: float = Field(
        2, ge=0, validation_alias='MEDIA_OVERLAY_READY_GRACE_SECONDS',
    )
    preview_fps: float | None = Field(
        None, gt=0, allow_inf_nan=False, validation_alias='MEDIA_PREVIEW_FPS',
    )
    preview_width: int = Field(
        640, ge=2, validation_alias='MEDIA_PREVIEW_WIDTH',
    )
    preview_height: int = Field(
        360, ge=2, validation_alias='MEDIA_PREVIEW_HEIGHT',
    )
    preview_bitrate: str = Field(
        '500k', validation_alias='MEDIA_PREVIEW_BITRATE',
    )
    preview_maxrate: str = Field(
        '700k', validation_alias='MEDIA_PREVIEW_MAXRATE',
    )
    preview_bufsize: str = Field(
        '1400k', validation_alias='MEDIA_PREVIEW_BUFSIZE',
    )
    overlay_min_confidence: float = Field(
        0.25,
        ge=0,
        le=1,
        validation_alias='MEDIA_PUBLISH_OVERLAY_MIN_CONFIDENCE',
    )
    box_thickness: int = Field(
        2, ge=1, validation_alias='MEDIA_PUBLISH_BOX_THICKNESS',
    )
    startup_width: int = Field(
        1280, ge=2, validation_alias='MEDIA_PUBLISH_STARTUP_WIDTH',
    )
    startup_height: int = Field(
        720, ge=2, validation_alias='MEDIA_PUBLISH_STARTUP_HEIGHT',
    )

    @field_validator('publish_base')
    @classmethod
    def normalise_publish_base(cls, value: str) -> str:
        return (value or 'rtsp://media-server:8554').rstrip('/')

    @field_validator(
        'demand_cache_seconds', 'ready_grace_seconds', mode='before',
    )
    @classmethod
    def bounded_optional_duration(cls, value: object, info) -> float:
        try:
            return max(0, float(str(value)))
        except ValueError:
            return 0.5 if info.field_name == 'demand_cache_seconds' else 2.0

    @field_validator('source_fps', 'preview_fps', mode='before')
    @classmethod
    def automatic_rendition_frame_rate(cls, value: object):
        return cls.automatic_frame_rate(value)

    @field_validator(
        'preview_width',
        'preview_height',
        'box_thickness',
        mode='before',
    )
    @classmethod
    def bounded_dimension(cls, value: object, info) -> int:
        lower = 1 if info.field_name == 'box_thickness' else 2
        return max(lower, int(float(str(value))))

    def rendition_fps(
        self, rendition: str, source_fps: float | None = None,
    ) -> float:
        override = self.preview_fps if rendition == 'preview' else None
        if override is not None:
            return override
        if self.fps is not None:
            return self.fps
        if (
            isinstance(source_fps, (int, float))
            and math.isfinite(source_fps) and source_fps > 0
        ):
            return source_fps
        # Keep startup slates usable until the camera reports its frame rate.
        return 30.0


class MediaGatewaySettings(OverlayReadinessSettings):
    model_config = {'env_prefix': 'MEDIA_'}
    publisher_idle_grace_seconds: int = Field(180, ge=30)
    internal_hls_base_url: str = 'http://127.0.0.1:8890'
    internal_hls_timeout_seconds: float = Field(10, gt=0, allow_inf_nan=False)
    public_hls_base_url: str = '/hazard/media'
    public_webrtc_base_url: str = '/hazard/media/webrtc'

    @field_validator('internal_hls_base_url')
    @classmethod
    def internal_endpoint(cls, value):
        return http_endpoint(value)

    @field_validator('public_hls_base_url', 'public_webrtc_base_url')
    @classmethod
    def public_base(cls, value, info):
        value = value or cls.model_fields[info.field_name].default
        if value.startswith('/') and not value.startswith('//'):
            return value.rstrip('/')
        return http_endpoint(value)


class OverlayTextSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'STREAMING_OVERLAY_'}
    text_bitmap_cache_size: int = Field(4096, ge=128)
    font_path: str = ''


class NvencSettings(EnvironmentSettings):
    max_sessions: int = Field(
        6,
        ge=1,
        validation_alias='MEDIA_NVENC_MAX_SESSIONS',
    )


class WarningEventSettings(EnvironmentSettings):
    throttle_seconds: int = Field(
        30,
        ge=1,
        validation_alias='WARNING_EVENT_THROTTLE_SECONDS',
    )
