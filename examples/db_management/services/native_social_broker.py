from __future__ import annotations

import base64
import hmac
import ipaddress
import json
import time

from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis

from examples.db_management.services.native_social_state import (
    _consume_transaction,
)
from examples.db_management.services.native_social_state import (
    _record_provider,
)
from examples.db_management.services.native_social_state import (
    _REDEEM_HMAC_MAX_AGE_SECONDS,
)
from examples.db_management.services.native_social_state import (
    _require_enabled,
)
from examples.db_management.services.native_social_state import (
    _required_string,
)
from examples.db_management.services.native_social_state import settings


def _is_loopback_request(request: Request) -> bool:
    """Return whether a Keycloak redemption request came from loopback."""
    if request.client is None:
        return False
    try:
        return ipaddress.ip_address(request.client.host).is_loopback
    except ValueError:
        return request.client.host.lower() == 'localhost'


def _hmac_signature(timestamp: str, body: bytes) -> str:
    """Build the canonical HMAC for a private Keycloak redemption request."""
    digest = hmac.digest(
        settings.native_social_exchange_shared_secret.encode('utf-8'),
        timestamp.encode('ascii') + b'.' + body,
        'sha256',
    )
    return base64.urlsafe_b64encode(digest).rstrip(b'=').decode('ascii')


def _parse_redeem_body(body: bytes) -> dict[str, str]:
    """Parse the custom Keycloak request as a strict, string-only object."""
    try:
        data = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        ) from exc
    if not isinstance(data, dict):
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        )
    fields = ('transaction_id', 'client_id', 'redirect_uri', 'code_challenge')
    if set(data) != set(fields) or any(
        not isinstance(data.get(field), str) or not data[field]
        for field in fields
    ):
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        )
    return {field: data[field] for field in fields}


async def redeem_keycloak_native_social_exchange(
    request: Request,
    redis: Redis,
) -> dict[str, str]:
    """Redeem one proof for the Keycloak authenticator over loopback HMAC."""
    _require_enabled()
    if not _is_loopback_request(request):
        raise HTTPException(
            status_code=403,
            detail='native_social_exchange_forbidden',
        )
    timestamp = request.headers.get('X-Visionnaire-Timestamp', '')
    signature = request.headers.get('X-Visionnaire-Signature', '')
    if not timestamp.isdecimal():
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        )
    if abs(time.time() - int(timestamp)) > _REDEEM_HMAC_MAX_AGE_SECONDS:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        )
    body = await request.body()
    if not hmac.compare_digest(_hmac_signature(timestamp, body), signature):
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        )
    payload = _parse_redeem_body(body)
    record = await _consume_transaction(redis, payload['transaction_id'])
    if record.get('kind') != 'login-redeem':
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        )
    for name in ('client_id', 'redirect_uri', 'code_challenge'):
        if not hmac.compare_digest(
            _required_string(record, name),
            payload[name],
        ):
            raise HTTPException(
                status_code=401,
                detail='native_social_exchange_invalid',
            )
    encoded_subject = _required_string(record, 'provider_subject_b64')
    try:
        padded = encoded_subject + '=' * (-len(encoded_subject) % 4)
        subject = base64.urlsafe_b64decode(
            padded.encode('ascii'),
        ).decode('utf-8')
    except (UnicodeDecodeError, ValueError) as exc:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        ) from exc
    if not subject or len(subject.encode('utf-8')) > 512:
        raise HTTPException(
            status_code=401,
            detail='native_social_exchange_invalid',
        )
    return {
        'provider': _record_provider(record),
        'provider_subject_b64': encoded_subject,
    }
