from __future__ import annotations

import logging

from examples.bff.settings import ProxySettings


logger = logging.getLogger('uvicorn.error')


settings = ProxySettings()
BFF_UPSTREAMS = settings.upstreams
REFRESH_BEFORE_SECONDS = settings.refresh_before_seconds
UPSTREAM_TIMEOUT_SECONDS = settings.upstream_timeout_seconds


_DROP_REQUEST_HEADERS = {
    'x-company-ai-network-proxy',
    'x-company-ai-bridge',
    'x-company-ai-subject',
    'x-company-ai-admin',
    'x-company-ai-client-ip',
    'authorization',
    'connection',
    'content-length',
    'cookie',
    'host',
    'proxy-authorization',
    'te',
    'trailer',
    'transfer-encoding',
    'upgrade',
    'x-csrf-token',
    'x-forwarded-host',
    'x-forwarded-proto',
}


_PASS_RESPONSE_HEADERS = {
    'accept-ranges',
    'content-disposition',
    'content-language',
    'content-range',
    'content-type',
    'etag',
    'last-modified',
    'retry-after',
}


_VIOLATION_MEDIA_FIELDS = frozenset({'image_url', 'thumbnail_url'})


_VIOLATION_MEDIA_ENDPOINTS = frozenset(
    {'/get_violation_image', '/get_violation_thumbnail'},
)
