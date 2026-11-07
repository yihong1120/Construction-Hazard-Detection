from __future__ import annotations

import hashlib
from datetime import datetime

from cryptography.fernet import Fernet
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession

from examples.auth.models import FcmDeviceToken


def fcm_token_hash(device_token: str) -> str:
    """Hash an FCM token for metadata keys and API responses."""
    return hashlib.sha256(device_token.encode('utf-8')).hexdigest()


def encrypt_token(device_token: str, encryption_key: str) -> str:
    """Encrypt an FCM token using the configured Fernet key."""
    return (
        Fernet(encryption_key.encode('utf-8'))
        .encrypt(
            device_token.encode('utf-8'),
        )
        .decode('utf-8')
    )


def decrypt_token(encrypted_token: str, encryption_key: str) -> str:
    """Decrypt an FCM token using the configured Fernet key."""
    return (
        Fernet(encryption_key.encode('utf-8'))
        .decrypt(
            encrypted_token.encode('utf-8'),
        )
        .decode('utf-8')
    )


async def disable_undecryptable_token(
    db: AsyncSession,
    row: FcmDeviceToken,
    occurred_at: datetime,
) -> bool:
    """Disable only the exact registration whose ciphertext failed."""
    result = await db.execute(
        update(FcmDeviceToken).where(
            FcmDeviceToken.id == row.id,
            FcmDeviceToken.user_id == row.user_id,
            FcmDeviceToken.last_seen_at == row.last_seen_at,
            FcmDeviceToken.device_token_encrypted == (
                row.device_token_encrypted
            ),
            FcmDeviceToken.disabled_at.is_(None),
        ).values(
            disabled_at=occurred_at, last_failure_at=occurred_at,
            failure_reason='token_decryption_failed', updated_at=occurred_at,
        ).execution_options(synchronize_session=False).returning(
            FcmDeviceToken.id,
        ),
    )
    return result.scalar_one_or_none() is not None
