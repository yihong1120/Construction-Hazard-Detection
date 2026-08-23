from __future__ import annotations

import base64
import json
import time
from typing import cast

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import load_pem_private_key

from examples.auth.deployment_context import canonical_api_base_url
from examples.auth.models import Deployment
from examples.deployment_registry.schemas import MAX_REGISTRY_TTL_SECONDS
from examples.deployment_registry.schemas import REGISTRY_SCHEMA_VERSION


def build_registry_document(
    deployment: Deployment,
    private_key_pem: str,
    key_id: str,
    ttl_seconds: int,
    issued_at: int | None = None,
) -> dict[str, object]:
    """Build the exact nine-field signed public Registry document.

    Args:
        deployment: Active deployment supplying the public configuration.
        private_key_pem: PEM-encoded Ed25519 private key.
        key_id: Identifier for the client-pinned public key.
        ttl_seconds: Lifetime of the signed document in seconds.
        issued_at: Optional Unix timestamp for deterministic callers.

    Returns:
        Public Registry document containing a Base64url signature.

    Raises:
        ValueError: If the configuration, deployment, or signing key is invalid.
    """
    # Restrict document lifetime to the public Registry contract.
    if not 0 < ttl_seconds <= MAX_REGISTRY_TTL_SECONDS:
        raise ValueError(
            'deployment registry signing configuration is invalid',
        )
    now: int = int(time.time()) if issued_at is None else issued_at
    if now < 0:
        raise ValueError(
            'deployment registry signing configuration is invalid',
        )
    # Sign only the canonical base URL that clients will subsequently use.
    api_base_url: str = canonical_api_base_url(deployment.api_base_url)
    if (
        api_base_url != deployment.api_base_url
        or deployment.config_revision < 1
    ):
        raise ValueError(
            'deployment registry configuration is invalid',
        )
    # Assemble all public fields before separating the signable payload.
    document: dict[str, object] = {
        'schema_version': REGISTRY_SCHEMA_VERSION,
        'deployment_id': str(deployment.id),
        'tenant_id': str(deployment.tenant_id),
        'api_base_url': api_base_url,
        'config_revision': deployment.config_revision,
        'issued_at': now,
        'expires_at': now + ttl_seconds,
        'key_id': key_id,
    }
    if not private_key_pem.strip():
        raise ValueError(
            'deployment registry signing key is not configured',
        )
    # Deployment secrets may preserve newlines as literal escape sequences.
    private_key = cast(
        Ed25519PrivateKey,
        load_pem_private_key(
            private_key_pem.strip().replace('\\n', '\n').encode('utf-8'),
            password=None,
        ),
    )
    # Key selection and the signature itself are deliberately not self-signed.
    signed_document: dict[str, object] = document.copy()
    del signed_document['key_id']
    # Canonical JSON ensures every client signs and verifies identical bytes.
    signature = private_key.sign(
        json.dumps(
            signed_document,
            ensure_ascii=False,
            separators=(',', ':'),
            sort_keys=True,
        ).encode('utf-8'),
    )
    if len(signature) != 64:
        raise ValueError(
            'deployment registry signing key must be Ed25519',
        )
    # Base64url carries binary signature bytes safely in the JSON document.
    document['signature'] = base64.urlsafe_b64encode(signature).decode(
        'ascii',
    ).rstrip('=')
    return document
