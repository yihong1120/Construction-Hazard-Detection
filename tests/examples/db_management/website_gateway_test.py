from __future__ import annotations

import shutil
import socket
import subprocess
import time
import unittest
from functools import partial
from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread

from deploy.website_access.agent import render


class _NativeGateRejectsForgeryAndAdminBackend(BaseHTTPRequestHandler):
    def __init__(self, *args, _test_admin_requests, **kwargs):
        self._test_admin_requests = _test_admin_requests
        super().__init__(*args, **kwargs)

    def do_GET(self):
        cookie = self.headers.get('Cookie')
        self._test_admin_requests.append(cookie)
        authenticated = cookie == 'session=test-global-admin'
        self.send_response(200 if authenticated else 401)
        self.end_headers()
        self.wfile.write(
            b'global-admin' if authenticated else b'unauthenticated',
        )

    def log_message(self, *args):
        pass


@unittest.skipUnless(shutil.which('nginx'), 'Nginx is not installed')
class WebsiteGatewayIntegrationTest(unittest.TestCase):
    def test_native_gate_rejects_forgery_and_protects_all_paths(self):
        """Only allowlisted source IPs pass, including API/WS/media paths."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with socket.socket() as listener:
                listener.bind(('127.0.0.1', 0))
                port = listener.getsockname()[1]
            data = {
                'policies': [
                    {
                        'site_key': site,
                        'version': 1,
                        'restricted': site == 'main',
                        'rules': [
                            {'network': '127.0.0.2/32', 'enabled': True},
                        ],
                    }
                    for site in ('main', 'ai', 'wiki')
                ],
            }
            maps, _ = render(data)
            admin_requests = []

            AdminBackend = partial(
                _NativeGateRejectsForgeryAndAdminBackend,
                _test_admin_requests=admin_requests,
            )

            backend = HTTPServer(('127.0.0.1', 0), AdminBackend)
            thread = Thread(target=backend.serve_forever, daemon=True)
            thread.start()
            config = root / 'nginx.conf'
            config.write_text(f'''
pid {root}/nginx.pid;
error_log {root}/error.log;
events {{}}
http {{
    access_log off;
    client_body_temp_path {root}/body;
    proxy_temp_path {root}/proxy;
    fastcgi_temp_path {root}/fastcgi;
    uwsgi_temp_path {root}/uwsgi;
    scgi_temp_path {root}/scgi;
    {maps}
    server {{
        listen 127.0.0.1:{port};
        if ($vn_website_main_allowed = 0) {{ return 403; }}
        location /admin-test {{
            proxy_pass http://127.0.0.1:{backend.server_port};
        }}
        location / {{ return 200 "ok"; }}
        location /media/ {{ auth_request /media-auth; return 200 "ok"; }}
        location = /media-auth {{ internal; return 204; }}
    }}
}}
''')
            process = subprocess.Popen(
                [
                    shutil.which('nginx'),
                    '-p',
                    str(root),
                    '-c',
                    str(config),
                    '-g',
                    'daemon off;',
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            try:
                for _ in range(50):
                    if process.poll() is not None:
                        self.fail('Isolated Nginx failed to start')
                    try:
                        with socket.create_connection(
                            ('127.0.0.1', port),
                            timeout=0.1,
                        ):
                            break
                    except OSError:
                        time.sleep(0.02)
                for path in (
                    '/',
                    '/static/app.js',
                    '/api/users',
                    '/media/camera/index.m3u8',
                    '/ws/stream',
                ):
                    for source, expected in [
                        ('127.0.0.1', b'403'),
                        ('127.0.0.2', b'200'),
                    ]:
                        with self.subTest(path=path, source=source):
                            with socket.create_connection(
                                ('127.0.0.1', port),
                                timeout=2,
                                source_address=(source, 0),
                            ) as client:
                                client.sendall(
                                    (
                                        f'GET {path} HTTP/1.1\r\n'
                                        'Host: test\r\n'
                                        'X-Forwarded-For: 127.0.0.2\r\n'
                                        'X-Real-IP: 127.0.0.2\r\n'
                                        'Connection: close\r\n\r\n'
                                    ).encode(),
                                )
                                status = client.recv(4096).split(b'\r\n')[0]
                                self.assertIn(expected, status)
                # The same authenticated administrator moves from an allowed
                # network to a denied network. The denied request must never
                # reach the backend, even with forged forwarding headers.
                for source, expected in [
                    ('127.0.0.2', b'200'),
                    ('127.0.0.1', b'403'),
                ]:
                    with self.subTest(admin=True, source=source):
                        with socket.create_connection(
                            ('127.0.0.1', port),
                            timeout=2,
                            source_address=(source, 0),
                        ) as client:
                            client.sendall(
                                (
                                    b'GET /admin-test HTTP/1.1\r\n'
                                    b'Host: test\r\n'
                                    b'Cookie: session=test-global-admin\r\n'
                                    b'X-Forwarded-For: 127.0.0.2\r\n'
                                    b'X-Real-IP: 127.0.0.2\r\n'
                                    b'Connection: close\r\n\r\n'
                                ),
                            )
                            with client.makefile('rb') as response:
                                status = response.readline()
                                body = response.read()
                            self.assertIn(expected, status)
                            if expected == b'200':
                                self.assertIn(b'global-admin', body)
                            self.assertEqual(
                                admin_requests,
                                ['session=test-global-admin'],
                            )
            finally:
                process.terminate()
                process.communicate(timeout=5)
                backend.shutdown()
                backend.server_close()
                thread.join(timeout=5)
