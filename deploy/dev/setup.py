from __future__ import annotations

import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

# noreorder
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from deploy.dev.identity import (  # noqa: E402
    provision,
    BASE,
    CLIENT,
    AUDIENCE,
    ORIGIN,
)  # noqa: E402
from deploy.dev.run import CONFIG, STATE, PORTS, validate  # noqa: E402
from src.settings.sessions import SESSION_POLICY  # noqa: E402

PYTHON = Path.home() / '.conda/envs/construction_env/bin/python'
DOC_PYTHON = Path.home() / '.conda/envs/docx2forms_env/bin/python'
FRONTEND = ROOT / 'frontend/web'


def private(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if path.is_symlink():
        raise RuntimeError('Refusing a symlink for private configuration.')
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, 'w') as output:
        output.write(value)
    path.chmod(0o600)


def nginx_configuration():
    config = f'''# Managed by construction-dev-v1
worker_processes 1;
pid {STATE}/gateway/nginx.pid;
error_log {STATE}/gateway/error.log warn;
events {{ worker_connections 256; }}
http {{
    include /etc/nginx/mime.types;
    default_type application/octet-stream;
    access_log off;
    client_body_temp_path {STATE}/gateway/client_temp;
    proxy_temp_path {STATE}/gateway/proxy_temp;
    server {{
        listen 127.0.0.1:18090;
        listen 100.96.17.117:18090;
        server_name dev.changdar-server.mooo.com;
        allow 127.0.0.1;
        allow 100.93.178.78;
        deny all;
        if ($host != dev.changdar-server.mooo.com) {{ return 444; }}
        root {FRONTEND};
        index index.html;
        client_max_body_size 100M;
        add_header X-Construction-Environment dev always;
        add_header Cache-Control "no-store" always;
        proxy_http_version 1.1;
        proxy_set_header Host dev.changdar-server.mooo.com;
        proxy_set_header X-Forwarded-Proto https;
        proxy_set_header X-Forwarded-For $http_x_forwarded_for;
        proxy_set_header X-Real-IP $http_x_real_ip;
        proxy_buffering off;
        proxy_read_timeout 3600s;
        location = /health {{ proxy_pass http://127.0.0.1:19005/health; }}
        location = /dev-build.json {{
            alias {STATE}/gateway/frontend-manifest.json;
            default_type application/json;
        }}
        location = /ai {{
            return 302 https://ai-dev.changdar-server.mooo.com/;
        }}
        location ^~ /ai/ {{
            return 302 https://ai-dev.changdar-server.mooo.com/;
        }}
        location ^~ /bff/ {{ proxy_pass http://127.0.0.1:19005; }}
'''
    for path, port in [
        ('db_management', 19005),
        ('file_manage', 19004),
        ('violations', 19002),
        ('fcm', 19003),
    ]:
        config += (
            f'        location ^~ /hazard/api/{path}/ {{ '
            f'proxy_pass http://127.0.0.1:{port}/; }}\n'
        )
    config += '''        location ^~ /hazard/api/ {
            proxy_pass http://127.0.0.1:19800/;
        }
        location ^~ /hazard/api/detect/ { proxy_pass http://127.0.0.1:19000/; }
        location ^~ /hazard/api/chat/ { proxy_pass http://127.0.0.1:19009/; }
        location ^~ /hazard/media/ {
            default_type application/json;
            return 503 '{"detail":"dev_media_not_connected"}';
        }
        location ^~ /keycloak/ { return 404; }
        location ~* ^/(?:index\\.html|.*\\.js|(?:version|manifest)\\.json)$ {
            try_files $uri =404;
        }
        location / { try_files $uri $uri/ /index.html; }
    }
}
'''
    return config


def unit(name, body):
    units = Path.home() / '.config/systemd/user'
    units.mkdir(parents=True, exist_ok=True)
    text = (
        '[Unit]\nDescription=Visionnaire isolated DEV '
        + name
        + '\nAfter=network-online.target\n'
    )
    if name not in {'db'}:
        text += (
            'After=construction-dev-db.service\nRequires=const'
            'ruction-dev-db.service\n'
        )
    text += '\n[Service]\n' + body + '\n\n[Install]\nWantedBy=default.target\n'
    private(units / f'construction-dev-{name}.service', text)


def main():
    if os.geteuid() == 0 or Path.home() != Path('/home/changdar'):
        raise RuntimeError('Run as changdar on A without sudo.')
    for path in (
        CONFIG,
        STATE,
        STATE / 'work',
        STATE / 'work/static',
        STATE / 'documents',
        STATE / 'documents/signatures',
        STATE / 'gateway',
        STATE / 'gateway/client_temp',
        STATE / 'gateway/proxy_temp',
    ):
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
    CONFIG.chmod(0o700)
    if (CONFIG / 'credentials.json').exists():
        credentials = json.loads((CONFIG / 'credentials.json').read_text())
    else:
        from cryptography.fernet import Fernet

        credentials = {
            k: secrets.token_urlsafe(32)
            for k in (
                'database',
                'redis',
                'jwt',
                'pepper',
            )
        }
        credentials.update(
            {
                'session': Fernet.generate_key().decode(),
                'fcm': Fernet.generate_key().decode(),
            },
        )
        private(CONFIG / 'credentials.json', json.dumps(credentials))
    client_secret, operator = provision()
    private(CONFIG / 'operator.json', json.dumps(operator))
    issuer = BASE + '/realms/visionnaire'
    protocol = issuer + '/protocol/openid-connect'
    values = {
        'DATABASE_URL': 'postgresql+asyncpg://construction_dev:'
        + credentials['database']
        + '@127.0.0.1:15433/construction_dev',
        'REDIS_HOST': '127.0.0.1',
        'REDIS_PORT': '16380',
        'REDIS_PASSWORD': credentials['redis'],
        'REDIS_DB': '0',
        'JWT_SECRET_KEY': credentials['jwt'],
        'BFF_TOKEN_ENCRYPTION_KEY': credentials['session'],
        'BFF_SESSION_COOKIE_NAME': '__Host-vn_dev_session',
        'BFF_SESSION_COOKIE_SECURE': 'true',
        'BFF_SESSION_TTL_SECONDS': str(SESSION_POLICY.idle_seconds),
        'FCM_TOKEN_ENCRYPTION_KEY': credentials['fcm'],
        'OIDC_ENABLED': 'true',
        'OIDC_ISSUER_URL': issuer,
        'OIDC_JWKS_URL': protocol + '/certs',
        'OIDC_AUDIENCE': AUDIENCE,
        'OIDC_IDENTITY_PROVIDER': 'keycloak',
        'OIDC_WEB_CLIENT_ID': CLIENT,
        'OIDC_WEB_CLIENT_SECRET': client_secret,
        'OIDC_WEB_AUTHORIZATION_ENDPOINT': protocol + '/auth',
        'OIDC_WEB_TOKEN_ENDPOINT': protocol + '/token',
        'OIDC_WEB_REDIRECT_URI': ORIGIN + '/bff/auth/oidc/callback',
        'OIDC_WEB_END_SESSION_ENDPOINT': protocol + '/logout',
        'OIDC_WEB_POST_LOGOUT_REDIRECT_URI': ORIGIN + '/',
        'OIDC_ACCOUNT_URL': issuer + '/account',
        'CORS_ALLOWED_ORIGINS': ORIGIN,
        'KEYCLOAK_USER_LINKER_CLIENT_SECRET': '',
        'NATIVE_SOCIAL_EXCHANGE_ENABLED': 'false',
        'LOCAL_DEVELOPMENT_AUTH_ENABLED': 'false',
        'AUTO_CREATE_SCHEMA': 'false',
        'SITE_MEDIA_CLEANUP_WORKER_ENABLED': 'false',
        'DEPLOYMENT_API_BASE_PATH': '/hazard/api',
        'DEPLOYMENT_ENROLLMENT_CODE_PEPPER': credentials['pepper'],
        'BFF_MANAGEMENT_API_URL': 'http://127.0.0.1:19005',
        'BFF_FILES_API_URL': 'http://127.0.0.1:19004',
        'BFF_FCM_API_URL': 'http://127.0.0.1:19003',
        'BFF_VIOLATIONS_API_URL': 'http://127.0.0.1:19002',
        'BFF_STREAMING_API_URL': 'http://127.0.0.1:19800',
        'STREAMING_PLAYBACK_API_URL': 'http://127.0.0.1:19800',
        'BFF_CHAT_API_URL': 'http://127.0.0.1:19009',
        'BFF_DETECTION_API_URL': 'http://127.0.0.1:19000',
        'BFF_AI_USAGE_API_URL': 'http://127.0.0.1:19009',
        'BFF_AI_ACCESS_API_URL': 'http://127.0.0.1:18083',
        'MEDIA_INTERNAL_HLS_BASE_URL': 'http://127.0.0.1:19009',
        'STORAGE_ROOT': str(STATE / 'documents'),
        'SIGNATURE_ROOT': str(STATE / 'documents/signatures'),
        'WIN_UPLOAD_ROOT': str(STATE / 'documents'),
        'APP_PUBLIC_URL': ORIGIN,
        'PDF_CONVERSION_BACKEND': 'libre',
        'FIREBASE_CRED_PATH': '',
        'FIREBASE_PROJECT_ID': '',
        'HCAPTCHA_ENABLED': 'false',
        'DB_POOL_SIZE': '2',
        'DB_MAX_OVERFLOW': '1',
        'YOLO_INFERENCE_DEVICE': 'cuda:0',
        'USE_TENSORRT': 'false',
        'USE_SAHI': 'false',
        'LAZY_LOAD_MODELS': 'true',
        'PRELOAD_SMALLEST': 'false',
        'MAX_LOADED_MODELS': '1',
        'DETECT_INGRESS_CONCURRENCY': '1',
        'OMP_NUM_THREADS': '2',
        'MKL_NUM_THREADS': '2',
    }
    validate(values)
    private(CONFIG / 'environment.json', json.dumps(values, indent=2))
    private(
        CONFIG / 'postgres.env',
        (
            'POSTGRES_USER=construction_dev\nPOSTGRES_DB=const'
            'ruction_dev\nPOSTGRES_PASSWORD='
        )
        + credentials['database']
        + '\n',
    )
    private(
        CONFIG / 'redis.env',
        'REDIS_PASSWORD=' + credentials['redis'] + '\n',
    )
    compose = [
        'docker',
        'compose',
        '-f',
        str(ROOT / 'deploy/dev/compose.yaml'),
    ]
    subprocess.run(
        [
            'docker',
            'volume',
            'create',
            'construction-dev_postgres-parity-data',
        ],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    subprocess.run(
        [
            *compose,
            'up',
            '-d',
            '--wait',
            '--wait-timeout',
            '150',
        ],
        check=True,
    )
    # Apply DDL and seed before allowing a service to import the latest ORM.
    active = [
        f'construction-dev-{name}.service'
        for name in PORTS
        if subprocess.run(
            [
                'systemctl',
                '--user',
                'is-active',
                '--quiet',
                f'construction-dev-{name}.service',
            ],
        ).returncode
        == 0
    ]
    if active:
        subprocess.run(['systemctl', '--user', 'stop', *active], check=True)
    result = subprocess.run(
        [str(PYTHON), '-m', 'deploy.dev.seed'],
        cwd=ROOT,
    )
    if result.returncode:
        raise RuntimeError(
            'Dev schema preparation failed; APIs remain stopped. Inspect '
            'the migration backup/log and rerun setup after repair.',
        )
    unit(
        'db',
        'Type=oneshot\nRemainAfterExit=yes\nExecStart='
        + ' '.join(
            [*compose, 'up', '-d', '--wait', '--wait-timeout', '150'],
        )
        + '\nTimeoutStartSec=180\nRestart=on-failure\nRestartSec=5',
    )
    for name in [*PORTS, 'unavailable']:
        runtime = DOC_PYTHON if name == 'files' else PYTHON
        limits = (
            'MemoryMax=4G\nCPUQuota=200%'
            if name == 'detection'
            else 'MemoryMax=768M\nCPUQuota=100%'
        )
        unit(
            name,
            f'Type=simple\nExecStart={runtime} {ROOT}/deploy/dev/run.py '
            f'{name}\n'
            f'Restart=on-failure\nRestartSec=5\nUMask=0077\n{limits}',
        )
    (STATE / 'gateway/nginx.conf').write_text(nginx_configuration())
    subprocess.run(
        [
            '/usr/sbin/nginx',
            '-t',
            '-p',
            str(STATE / 'gateway'),
            '-c',
            str(STATE / 'gateway/nginx.conf'),
        ],
        check=True,
    )
    unit(
        'gateway',
        f'Type=simple\nExecStart=/usr/sbin/nginx -p {STATE}/gateway '
        f'-c {STATE}/gateway/nginx.conf -g "daemon off;"\n'
        'Restart=on-failure\nRestartSec=5\nUMask=0077\nMemoryMax=192M',
    )
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(
        [
            'systemctl',
            '--user',
            'enable',
            '--now',
            *[
                f'construction-dev-{n}.service'
                for n in [
                    'db',
                    *PORTS,
                    'unavailable',
                    'gateway',
                ]
            ],
        ],
        check=True,
    )
    print(
        (
            'Dev services prepared on A. Register frontend/web with '
            'deploy/dev/build_frontend.py before publishing.'
        ),
    )


if __name__ == '__main__':
    main()
