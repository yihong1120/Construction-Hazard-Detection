from __future__ import annotations

import json
import re
from urllib.parse import urlsplit
from urllib.parse import urlunsplit

from fastapi import Request

from examples.auth.deployment_context import DeploymentBinding
from examples.bff.proxy_config import _VIOLATION_MEDIA_ENDPOINTS
from examples.bff.proxy_config import _VIOLATION_MEDIA_FIELDS


def _bff_service_path(
    request: Request,
    service: str,
) -> str:
    """Return the public BFF prefix for one proxied service.

    The request path retains any deployment-specific API prefix, while the
    upstream service only knows its own root routes. Deriving the prefix from
    the original browser request therefore works for both ``/bff`` and a BFF
    mounted below a public API path.
    """
    marker = f'/{service}'
    request_path = request.url.path.rstrip('/')
    offset = request_path.find(marker)
    if offset >= 0:
        marker_end = offset + len(marker)
        if marker_end == len(request_path) or request_path[
            marker_end
        ] == '/':
            return request_path[:marker_end]
    return f'/bff/{service}'


def _bff_violation_media_url(
    value: object,
    request: Request,
    deployment: DeploymentBinding | None,
) -> str | None:
    """Map one internal violation-media URL to its BFF-protected URL.

    Violation services generate URLs for their own root endpoints, but browser
    sessions are authenticated only by the BFF cookie. Only recognised media
    paths are rewritten so an arbitrary URL in an upstream JSON response can
    never be redirected through the BFF.
    """
    if not isinstance(value, str):
        return None
    source = urlsplit(value)
    if source.path not in _VIOLATION_MEDIA_ENDPOINTS:
        return None

    path = f'{_bff_service_path(request, "violations")}{source.path}'
    if deployment is None:
        return urlunsplit(('', '', path, source.query, ''))
    public_url = urlsplit(deployment.api_base_url)
    return urlunsplit(
        (public_url.scheme, public_url.netloc, path, source.query, ''),
    )


def _rewrite_violation_media_urls(
    payload: object,
    request: Request,
    deployment: DeploymentBinding | None,
) -> bool:
    """Rewrite recognised evidence URLs in a violation JSON response.

    Returns:
        ``True`` when at least one URL was changed.
    """
    changed = False
    if isinstance(payload, list):
        for item in payload:
            changed |= _rewrite_violation_media_urls(
                item,
                request,
                deployment,
            )
        return changed
    if not isinstance(payload, dict):
        return False

    for key, value in payload.items():
        if key in _VIOLATION_MEDIA_FIELDS:
            rewritten = _bff_violation_media_url(
                value,
                request,
                deployment,
            )
            if rewritten is not None and rewritten != value:
                payload[key] = rewritten
                changed = True
            continue
        changed |= _rewrite_violation_media_urls(
            value,
            request,
            deployment,
        )
    return changed


def _rewrite_violation_response_content(
    content: bytes,
    content_type: str | None,
    request: Request,
    deployment: DeploymentBinding | None,
) -> tuple[bytes, bool]:
    """Return violation JSON content with browser-reachable media URLs.

    Non-JSON response bodies, including the protected JPEG endpoints
    themselves, must pass through without parsing or re-encoding.
    """
    if not content_type or content_type.split(';', 1)[0].lower() != (
        'application/json'
    ):
        return content, False
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return content, False
    if not _rewrite_violation_media_urls(payload, request, deployment):
        return content, False
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            separators=(',', ':'),
        ).encode('utf-8'),
        True,
    )


def _rewrite_file_download_response_content(
    content: bytes,
    content_type: str | None,
    request: Request,
    deployment: DeploymentBinding | None,
    service: str,
    suffix: str,
) -> tuple[bytes, bool]:
    """Keep browser download capabilities behind the cookie-authenticated
    BFF.
    """
    if not re.fullmatch(
        r'(?:versions|exports)/[0-9]+/generate_temp_url',
            suffix,
    ):
        return content, False
    if not content_type or content_type.split(
            ';', 1,
    )[0].lower() != 'application/json':
        return content, False
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return content, False
    if not isinstance(
            payload,
            dict,
    ) or not isinstance(
            payload.get('temp_url'),
            str,
    ):
        return content, False
    source = urlsplit(payload['temp_url'])
    if source.scheme or source.netloc or source.query or source.fragment:
        return content, False
    public_url = urlsplit(deployment.api_base_url) if deployment else None
    api_path = public_url.path.rstrip('/') if public_url else '/hazard/api'
    prefixes = ('/download/temp/', f'{api_path}/file_manage/download/temp/')
    token = next(
        (
            source.path[len(prefix):]
            for prefix in prefixes if source.path.startswith(prefix)
        ), '',
    )
    if not re.fullmatch(
        r'[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+',
            token,
    ):
        return content, False
    path = f'{_bff_service_path(request, service)}/download/temp/{token}'
    payload['temp_url'] = urlunsplit((
        public_url.scheme if public_url else '',
        public_url.netloc if public_url else '',
        path, '', '',
    ))
    return json.dumps(
        payload, ensure_ascii=False, separators=(
            ',', ':',
        ),
    ).encode('utf-8'), True
