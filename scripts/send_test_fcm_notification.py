#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import httpx

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - keeps the script usable in lean envs
    load_dotenv = None


DEFAULT_WARNING_KEY = 'warning_no_hardhat'
_SENSITIVE_KEYS = {
    'access_token',
    'device_token',
    'fcm_token',
    'refresh_token',
    'token',
}
_SENSITIVE_KEY_ALIASES = {
    key.replace('_', '').replace('-', '').lower() for key in _SENSITIVE_KEYS
}


def _is_sensitive_key(key: object) -> bool:
    """Return whether a JSON key may contain token material."""
    if not isinstance(key, str):
        return False
    normalised = key.replace('_', '').replace('-', '').lower()
    return (
        key.lower() in _SENSITIVE_KEYS or normalised in _SENSITIVE_KEY_ALIASES
    )


def _redact_sensitive(value: object) -> object:
    """Recursively redact token-like values before printing."""
    if isinstance(value, dict):
        return {
            key: (
                '<redacted>'
                if _is_sensitive_key(key)
                else _redact_sensitive(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _print_json(label: str, value: object) -> None:
    """Print JSON with token-like values redacted."""
    print(label)
    print(json.dumps(_redact_sensitive(value), ensure_ascii=False, indent=2))


def _load_env() -> None:
    """Load `.env` from the repository root when python-dotenv is available."""
    if load_dotenv is None:
        return
    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / '.env')


def _clean_base_url(value: str) -> str:
    """Normalise a base URL without removing path prefixes."""
    return value.rstrip('/') + '/'


def _url(base_url: str, path: str) -> str:
    """Join a base URL and endpoint path."""
    return urljoin(_clean_base_url(base_url), path.lstrip('/'))


def _load_json_arg(value: str | None) -> dict[str, Any] | None:
    """Load JSON from an inline string or from `@path`."""
    if value is None:
        return None

    raw = value
    if value.startswith('@'):
        raw = Path(value[1:]).read_text(encoding='utf-8')

    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError('JSON value must be an object.')
    return parsed


def _build_body(args: argparse.Namespace) -> dict[str, dict[str, object]]:
    """Build the warning body accepted by `/send_fcm_notification`."""
    body_json = _load_json_arg(args.body_json)
    if body_json is not None:
        return body_json

    return {
        args.warning_key: {
            'count': args.count,
        },
    }


def _login(client: httpx.Client, args: argparse.Namespace) -> str:
    """Authenticate against db-management and return an access token."""
    username = args.username or os.getenv('API_USERNAME', '')
    password = args.password or os.getenv('API_PASSWORD', '')
    if not username or not password:
        raise RuntimeError(
            'Missing login credentials. Provide --access-token, or set '
            'API_USERNAME/API_PASSWORD in .env, or pass '
            '--username/--password.',
        )

    headers: dict[str, str] = {}
    bypass_key = args.hcaptcha_bypass_key or os.getenv(
        'HCAPTCHA_BYPASS_KEY', '',
    )
    if bypass_key:
        headers['X-HCaptcha-Bypass-Key'] = bypass_key

    response = client.post(
        _url(args.db_api_url, '/login'),
        json={'identifier': username, 'password': password},
        headers=headers,
    )
    if response.status_code != 200:
        raise RuntimeError(
            'Login failed: ' f"HTTP {response.status_code} {response.text}",
        )

    data = response.json()
    token = data.get('access_token')
    if not isinstance(token, str) or not token:
        raise RuntimeError('Login response did not contain access_token.')
    return token


def _store_device_token(
    client: httpx.Client,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    """Optionally register one frontend FCM token before sending."""
    if args.device_token is None and args.user_id is None:
        return None
    if args.device_token is None or args.user_id is None:
        raise RuntimeError(
            'Use --user-id and --device-token together when registering a '
            'test token.',
        )

    payload = {
        'user_id': args.user_id,
        'device_token': args.device_token,
        'device_lang': args.device_lang,
    }
    response = client.post(_url(args.fcm_url, '/store_token'), json=payload)
    if response.status_code >= 400:
        raise RuntimeError(
            'Store token failed: '
            f"HTTP {response.status_code} {response.text}",
        )
    return response.json()


def _send_notification(
    client: httpx.Client,
    args: argparse.Namespace,
    access_token: str,
) -> dict[str, Any]:
    """Send one site notification through the backend FCM API."""
    payload: dict[str, object] = {
        'site': args.site,
        'stream_name': args.stream_name,
        'body': _build_body(args),
        'image_path': args.image_path,
        'violation_id': args.violation_id,
        'deep_link': args.deep_link,
        'type': args.notification_type,
        'title': args.title,
        'metadata': _load_json_arg(args.metadata_json),
    }
    payload = {
        key: value for key, value in payload.items() if value is not None
    }

    if args.print_payload or args.dry_run:
        _print_json('Payload:', payload)
    if args.dry_run:
        return {'success': True, 'message': 'Dry run only.'}

    response = client.post(
        _url(args.fcm_url, '/send_fcm_notification'),
        json=payload,
        headers={'Authorization': f"Bearer {access_token}"},
    )
    if response.status_code >= 400:
        raise RuntimeError(
            'Send notification failed: '
            f"HTTP {response.status_code} {response.text}",
        )
    return response.json()


def _parser() -> argparse.ArgumentParser:
    """Create the command-line parser."""
    parser = argparse.ArgumentParser(
        description='Send a test FCM notification through the backend API.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--fcm-url',
        default=os.getenv('FCM_API_URL', 'http://127.0.0.1:8003'),
        help='Base URL of examples.local_notification_server.',
    )
    parser.add_argument(
        '--db-api-url',
        default=os.getenv('DB_MANAGEMENT_API_URL', 'http://127.0.0.1:8005'),
        help='Base URL of examples.db_management for login.',
    )
    parser.add_argument(
        '--access-token',
        default=os.getenv('FCM_TEST_ACCESS_TOKEN', ''),
        help='JWT access token. If omitted, the script logs in first.',
    )
    parser.add_argument('--username', default='', help='Login username/email.')
    parser.add_argument('--password', default='', help='Login password.')
    parser.add_argument(
        '--hcaptcha-bypass-key',
        default='',
        help='Optional local hCaptcha bypass key for login.',
    )

    parser.add_argument(
        '--site',
        default=os.getenv('FCM_TEST_SITE', ''),
        required=not bool(os.getenv('FCM_TEST_SITE')),
        help='Site name. Must match subscribed users in backend data.',
    )
    parser.add_argument(
        '--stream-name',
        default=os.getenv('FCM_TEST_STREAM', 'TestStream'),
        help='Camera stream name shown in the notification.',
    )
    parser.add_argument(
        '--warning-key',
        default=DEFAULT_WARNING_KEY,
        help='Warning key from local_notification_server/lang_config.py.',
    )
    parser.add_argument(
        '--count',
        type=int,
        default=1,
        help='Count placeholder for the default warning body.',
    )
    parser.add_argument(
        '--body-json',
        default=None,
        help='Inline JSON body, or @path/to/body.json.',
    )
    parser.add_argument('--image-path', default=None)
    parser.add_argument('--violation-id', type=int, default=None)
    parser.add_argument('--deep-link', default=None)
    parser.add_argument(
        '--notification-type',
        choices=['signature', 'violation', 'document', 'site_alert', 'system'],
        default='violation',
    )
    parser.add_argument('--title', default=None)
    parser.add_argument(
        '--metadata-json',
        default=None,
        help='Inline JSON metadata, or @path/to/metadata.json.',
    )

    parser.add_argument(
        '--user-id',
        type=int,
        default=None,
        help='Optional user id for pre-registering one FCM token.',
    )
    parser.add_argument(
        '--device-token',
        default=None,
        help='Optional frontend FCM token to register before sending.',
    )
    parser.add_argument(
        '--device-lang',
        default='zh-TW',
        help='Language stored with --device-token.',
    )

    parser.add_argument('--timeout', type=float, default=30.0)
    parser.add_argument(
        '--print-payload',
        action='store_true',
        help='Print request payload before sending.',
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Build payload and login, but do not call send endpoint.',
    )
    parser.add_argument(
        '--allow-unsuccessful',
        action='store_true',
        help='Exit 0 even when backend returns success=false.',
    )
    return parser


def main() -> int:
    """Run the test notification sender."""
    _load_env()
    args = _parser().parse_args()

    try:
        with httpx.Client(timeout=args.timeout) as client:
            store_result = _store_device_token(client, args)
            if store_result is not None:
                _print_json('Store token response:', store_result)

            access_token = args.access_token or _login(client, args)
            result = _send_notification(client, args, access_token)

        _print_json('Send response:', result)

        success = bool(result.get('success'))
        if success or args.allow_unsuccessful:
            return 0
        return 2
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
