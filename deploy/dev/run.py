from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
STATE = Path.home() / '.local/share/construction-dev'
CONFIG = Path.home() / '.config/construction-dev'
PORTS = {
    'management': 19005,
    'violations': 19002,
    'streaming': 19800,
    'notifications': 19003,
    'files': 19004,
    'detection': 19000,
}


def validate(values):
    parsed = urlsplit(values.get('DATABASE_URL', ''))
    if (
        parsed.scheme != 'postgresql+asyncpg'
        or parsed.hostname != '127.0.0.1'
        or parsed.port != 15433
        or parsed.username != 'construction_dev'
        or parsed.path != '/construction_dev'
    ):
        raise RuntimeError('Refusing a non-dev database.')
    if (
        values.get('REDIS_HOST') != '127.0.0.1'
        or values.get('REDIS_PORT') != '16380'
    ):
        raise RuntimeError('Refusing shared Redis.')
    if values.get('STORAGE_ROOT') != str(STATE / 'documents'):
        raise RuntimeError('Refusing shared document storage.')
    if values.get('SIGNATURE_ROOT') != str(
        STATE / 'documents/signatures',
    ) or values.get('WIN_UPLOAD_ROOT') != str(STATE / 'documents'):
        raise RuntimeError('Refusing shared signatures or worker storage.')
    if values.get('BFF_SESSION_COOKIE_NAME') != '__Host-vn_dev_session':
        raise RuntimeError('Refusing a production cookie name.')
    if values.get('OIDC_AUDIENCE') != 'visionnaire-dev-api':
        raise RuntimeError('Refusing a production token audience.')
    if values.get('KEYCLOAK_USER_LINKER_CLIENT_SECRET'):
        raise RuntimeError(
            'Shared identity administration is disabled in dev.',
        )
    if values.get('FIREBASE_CRED_PATH') or values.get('FIREBASE_PROJECT_ID'):
        raise RuntimeError('Production push credentials are disabled in dev.')
    expected = {
        'BFF_MANAGEMENT_API_URL': 19005,
        'BFF_FILES_API_URL': 19004,
        'BFF_FCM_API_URL': 19003,
        'BFF_VIOLATIONS_API_URL': 19002,
        'BFF_STREAMING_API_URL': 19800,
        'STREAMING_PLAYBACK_API_URL': 19800,
        'BFF_CHAT_API_URL': 19009,
        'BFF_DETECTION_API_URL': 19000,
        'BFF_AI_USAGE_API_URL': 19009,
        'BFF_AI_ACCESS_API_URL': 18083,
    }
    for key, port in expected.items():
        if values.get(key) != f'http://127.0.0.1:{port}':
            raise RuntimeError('Refusing a shared upstream: ' + key)


def load_environment():
    values = json.loads((CONFIG / 'environment.json').read_text())
    validate(values)
    # Remove inherited configuration, which may contain production secrets.
    keep = {
        k: v
        for k, v in os.environ.items()
        if k
        in {
            'PATH',
            'HOME',
            'USER',
            'LANG',
            'LC_ALL',
            'SYSTEMD_EXEC_PID',
            'NOTIFY_SOCKET',
        }
    }
    os.environ.clear()
    os.environ.update(keep)
    os.environ.update(values)
    os.environ['PYTHON_DOTENV_DISABLED'] = '1'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    os.chdir(STATE / 'work')
    sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('service', choices=[*PORTS, 'unavailable'])
    service = parser.parse_args().service
    load_environment()
    if service != 'unavailable':
        import asyncio
        from deploy.dev.schema import require_current_schema

        asyncio.run(require_current_schema())
    import uvicorn

    if service == 'files':
        sys.path.insert(0, str(Path.home() / 'Documents/DocFormify'))
        target = 'app:app'
    else:
        target = 'deploy.dev.apps:' + service
    transport = {}
    if service == 'detection':
        from src.settings.inference import ApiInferenceSettings

        transport['ws_max_size'] = ApiInferenceSettings().max_upload_bytes
    uvicorn.run(
        target,
        factory=service != 'files',
        host='127.0.0.1',
        port=PORTS.get(service, 19009),
        workers=1,
        proxy_headers=True,
        forwarded_allow_ips='127.0.0.1',
        access_log=False,
        **transport,
    )


if __name__ == '__main__':
    main()
