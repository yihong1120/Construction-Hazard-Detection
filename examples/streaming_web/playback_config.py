from __future__ import annotations

from src.settings.media import PlaybackSettings

settings = PlaybackSettings()


OVERLAY_DEMAND_TTL_SECONDS = settings.overlay_demand_ttl_seconds


CLEAN_DEMAND_TTL_SECONDS = settings.clean_demand_ttl_seconds


OVERLAY_MAX_ACTIVE_LANGUAGES = settings.overlay_max_active_languages


STREAM_PLAYBACK_SESSION_TTL_SECONDS = settings.session_ttl_seconds


STREAM_PLAYBACK_SESSION_PREFIX = 'stream_playback_session'


STREAM_PLAYBACK_MEDIA_SESSION_PREFIX = 'stream_playback_media_session'


STREAM_PLAYBACK_DEMAND_SESSION_PREFIX = 'stream_playback_demand_session'


STREAM_PLAYBACK_SESSION_REFRESH_PREFIX = 'stream_playback_session_refresh'


STREAM_PLAYBACK_PUBLIC_BASE_PATH = settings.public_base_path.rstrip('/')


STREAM_PLAYBACK_STARTUP_WAIT_SECONDS = settings.startup_wait_seconds


STREAM_PLAYBACK_SESSION_REFRESH_INTERVAL_SECONDS = (
    settings.refresh_interval_seconds
)

STREAM_PLAYBACK_REFRESH_BATCH_SIZE = settings.refresh_batch_size
