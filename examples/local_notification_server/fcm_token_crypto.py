from __future__ import annotations

import hashlib
from datetime import datetime

from cryptography.fernet import Fernet

from examples.auth.models import FcmDeviceToken


def fcm_token_hash(device_token: str) -> str:
    """Hash an FCM token for metadata keys and API responses."""
    return hashlib.sha256(device_token.encode('utf-8')).hexdigest()


def encrypt_token(device_token: str, encryption_key: str) -> str:
    """Encrypt an FCM token using the configured Fernet key."""
    return Fernet(encryption_key.encode('utf-8')).encrypt(
        device_token.encode('utf-8'),
    ).decode('utf-8')


def decrypt_token(encrypted_token: str, encryption_key: str) -> str:
    """Decrypt an FCM token using the configured Fernet key."""
    return Fernet(encryption_key.encode('utf-8')).decrypt(
        encrypted_token.encode('utf-8'),
    ).decode('utf-8')


def disable_undecryptable_token(
    row: FcmDeviceToken,
    occurred_at: datetime,
) -> None:
    """Mark a token row unusable when its ciphertext cannot be recovered."""
    row.disabled_at = occurred_at
    row.last_failure_at = occurred_at
    row.failure_reason = 'token_decryption_failed'
    row.updated_at = occurred_at
