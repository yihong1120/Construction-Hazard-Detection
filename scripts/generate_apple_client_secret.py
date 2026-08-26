"""Generate the short-lived JWT Apple accepts as a Keycloak OIDC secret.

The generated value is a credential. It is written only to stdout so an
operator can place it directly in the deployment secret store or local .env;
the private Apple .p8 key is never copied into the Keycloak container.
"""
from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from pathlib import Path

import jwt

APPLE_ISSUER = 'https://appleid.apple.com'
APPLE_MAX_CLIENT_SECRET_DAYS = 180


def _arguments() -> argparse.Namespace:
    """Parse the Apple developer identifiers and private-key location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--team-id', required=True)
    parser.add_argument('--key-id', required=True)
    parser.add_argument('--client-id', required=True)
    parser.add_argument('--private-key-file', type=Path, required=True)
    parser.add_argument(
        '--expires-in-days',
        type=int,
        default=150,
        help=(
            'Lifetime from 1 to 180 days. 150 leaves time for a safe '
            'rotation before Apple\'s maximum expiry.'
        ),
    )
    return parser.parse_args()


def _required(value: str, option: str) -> str:
    """Return a trimmed nonempty configuration value."""
    value = value.strip()
    if not value:
        raise ValueError(f'{option} must not be empty')
    return value


def _apple_client_secret(args: argparse.Namespace) -> str:
    """Build one ES256 Apple OAuth client-secret JWT without persisting it."""
    expires_in_days = args.expires_in_days
    if not 1 <= expires_in_days <= APPLE_MAX_CLIENT_SECRET_DAYS:
        raise ValueError(
            '--expires-in-days must be between 1 and '
            f'{APPLE_MAX_CLIENT_SECRET_DAYS}',
        )
    key_path = args.private_key_file.expanduser()
    try:
        private_key = key_path.read_text(encoding='utf-8')
    except OSError as exc:
        raise ValueError(f'cannot read --private-key-file: {exc}') from exc
    if not private_key.strip():
        raise ValueError('--private-key-file is empty')

    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            'iss': _required(args.team_id, '--team-id'),
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(days=expires_in_days)).timestamp()),
            'aud': APPLE_ISSUER,
            'sub': _required(args.client_id, '--client-id'),
        },
        private_key,
        algorithm='ES256',
        headers={'kid': _required(args.key_id, '--key-id')},
    )


def main() -> int:
    """Print the generated secret without accidentally logging it elsewhere."""
    try:
        print(_apple_client_secret(_arguments()))
    except ValueError as exc:
        raise SystemExit(
            f'Apple client-secret generation failed: {exc}',
        ) from exc
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
