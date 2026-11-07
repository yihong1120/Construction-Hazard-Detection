from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import time
from pathlib import Path

SITE = Path('/etc/nginx/sites-enabled/phase6-main.conf')
MAP = Path('/etc/nginx/conf.d/ai-usage-mobile-routing.conf')
BACKUPS = Path('/home/changdar02/ai-usage-mobile-routing')
HOST = 'changdar-server.mooo.com'
ROUTE = '/hazard/api/ai_usage/v1/me/usage/daily?record_limit=1'
BLOCK = re.compile(
    r'(?m)^    location \^~ /hazard/api/ai_usage/ \{\n.*?^    \}\n',
    re.DOTALL,
)
OLD_MARKER = 'proxy_set_header Authorization "";'
NEW_MARKER = 'proxy_pass $vn_ai_usage_upstream;'

MAP_TEXT = (
    '# A native client supplies a Keycloak Bearer token.  Browser clients\n# '
    'supply only the BFF session cookie.  Both backends verify their own '
    'credential.\nmap $http_authorization $vn_ai_usage_native {\n    default '
    '0;\n    ~*^Bearer\\s+\\S+$ 1;\n}\nmap $vn_ai_usage_native '
    '$vn_ai_usage_upstream {\n    0 http://127.0.0.1:8006;\n    1 '
    'http://127.0.0.1:8012;\n}\nmap $vn_ai_usage_native $vn_ai_usage_prefix '
    '{\n    0 /bff/ai_usage/;\n    1 '
    '/;\n}\n'
)

NEW_BLOCK = (
    '    location ^~ /hazard/api/ai_usage/ {\n        rewrite '
    '^/hazard/api/ai_usage/(?<ai_usage_suffix>.*)$ '
    '$vn_ai_usage_prefix$ai_usage_suffix break;\n        proxy_pass '
    '$vn_ai_usage_upstream;\n        proxy_http_version 1.1;\n\n        '
    'proxy_set_header Host $host;\n        proxy_set_header X-Real-IP '
    '$remote_addr;\n        proxy_set_header X-Forwarded-For '
    '$proxy_add_x_forwarded_for;\n        proxy_set_header X-Forwarded-Proto '
    'https;\n        proxy_set_header Cookie $http_cookie;\n        '
    'proxy_set_header Authorization $http_authorization;\n\n        '
    'proxy_hide_header X-Powered-By;\n        proxy_hide_header '
    'Cache-Control;\n        proxy_buffering off;\n        proxy_cache off;\n '
    '       add_header Cache-Control "no-store" always;\n    '
    '}\n'
)


def run(*args: str) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(args, capture_output=True, text=True, timeout=30)
    if result.returncode:
        raise RuntimeError(f"{args[0]} failed (exit {result.returncode})")
    return result


def probe(with_bearer: bool) -> str:
    command = [
        'curl',
        '--noproxy',
        '*',
        '--silent',
        '--show-error',
        '--max-time',
        '10',
        '--resolve',
        f"{HOST}:443:127.0.0.1",
    ]
    if with_bearer:
        command += ['--header', 'Authorization: Bearer deliberately-invalid']
    command += ['--write-out', '\n%{http_code}', f"https://{HOST}{ROUTE}"]
    body, _, status = run(*command).stdout.rpartition('\n')
    if status != '401':
        raise RuntimeError(
            f"Unexpected AI usage authentication status {status}",
        )
    try:
        detail = json.loads(body)['detail']
        return detail['code'] if isinstance(detail, dict) else detail
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            'AI usage authentication response was not JSON',
        ) from exc


def wait_for_new_routes() -> None:
    """Allow the reloaded Nginx workers to replace the old configuration."""
    for _ in range(20):
        try:
            if (
                probe(False) == 'app_session_expired'
                and probe(True) == 'invalid_access_token'
            ):
                return
        except RuntimeError:
            pass
        time.sleep(0.25)
    raise RuntimeError(
        'Web or native AI usage route did not pass smoke checks',
    )


def check_sites() -> None:
    """Accept Wiki's intended OIDC login redirect as a healthy response."""
    for site in (
        HOST,
        'ai.changdar-server.mooo.com',
        'wiki.changdar-server.mooo.com',
    ):
        result = run(
            'curl',
            '--noproxy',
            '*',
            '--silent',
            '--show-error',
            '--max-time',
            '10',
            '--output',
            '/dev/null',
            '--dump-header',
            '-',
            '--write-out',
            '\n%{http_code}',
            '--resolve',
            f"{site}:443:127.0.0.1",
            f"https://{site}/",
        )
        headers, _, status = result.stdout.rpartition('\n')
        location = next(
            (
                line.split(':', 1)[1].strip()
                for line in headers.splitlines()
                if line.lower().startswith('location:')
            ),
            None,
        )
        if site.startswith('wiki.'):
            healthy = status == '302' and location == '/login'
        else:
            healthy = status == '200'
        if not healthy:
            raise RuntimeError(f"Website health failed: {site}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    source = SITE.read_text()
    matches = BLOCK.findall(source)
    if len(matches) != 1:
        raise RuntimeError('Expected exactly one AI usage gateway location')
    old_block = matches[0]
    if OLD_MARKER not in old_block or NEW_MARKER in old_block:
        raise RuntimeError(
            'AI usage gateway has changed; review before applying',
        )
    if MAP.exists():
        raise RuntimeError('AI usage routing map already exists; review first')
    if probe(False) != 'app_session_expired':
        raise RuntimeError(
            'Current Web session route did not return expected 401',
        )
    if probe(True) != 'app_session_expired':
        raise RuntimeError(
            'Current native route already differs from baseline',
        )
    check_sites()
    if args.check:
        print(
            (
                'Ready: Web and native requests currently share the '
                'Cookie-only BFF '
                'route.'
            ),
        )
        return

    backup = BACKUPS / ('backup-' + str(time.time_ns()))
    backup.mkdir(parents=True, mode=0o700)
    shutil.copy2(SITE, backup / 'phase6-main.conf')
    try:
        MAP.write_text(MAP_TEXT)
        SITE.write_text(source.replace(old_block, NEW_BLOCK, 1))
        run('nginx', '-t')
        run('systemctl', 'reload', 'nginx')
        wait_for_new_routes()
        check_sites()
    except Exception:
        shutil.copy2(backup / 'phase6-main.conf', SITE)
        MAP.unlink(missing_ok=True)
        run('nginx', '-t')
        run('systemctl', 'reload', 'nginx')
        raise
    print(
        (
            'Native AI usage Bearer routing verified; Web BFF and three '
            'public websites '
            'passed.'
        ),
    )
    print(f"Backup: {backup}")


if __name__ == '__main__':
    main()
