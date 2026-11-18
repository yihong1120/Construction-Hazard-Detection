"""Loopback evidence/OAuth/notification sinks with no user messages."""
from __future__ import annotations

import argparse
import json
import threading
import time
from email import policy
from email.parser import BytesParser
from http.server import BaseHTTPRequestHandler
from http.server import ThreadingHTTPServer
from pathlib import Path

import jwt


class Handler(BaseHTTPRequestHandler):
    events: dict[str, int] = {}
    lock = threading.Lock()
    journal = None

    def log_message(self, *_args):
        return

    def do_POST(self):
        size = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(size)
        if self.path == '/token':
            response = {
                'access_token': jwt.encode(
                    {'exp': int(time.time()) + 3600, 'sub': 'benchmark'},
                    'isolated-benchmark-secret-32bytes', algorithm='HS256',
                ), 'expires_in': 3600, 'token_type': 'Bearer',
            }
        else:
            key = self.headers.get('Idempotency-Key', '')
            if self.path == '/upload' and not key:
                message = BytesParser(policy=policy.default).parsebytes(
                    (
                        'Content-Type: ' +
                        self.headers.get('Content-Type', '') +
                        '\r\nMIME-Version: 1.0\r\n\r\n'
                    ).encode() + body,
                )
                for part in message.iter_parts():
                    if part.get_param(
                            'name', header='content-disposition',
                    ) == 'event_id':
                        key = part.get_payload(decode=True).decode()
            with self.lock:
                violation = self.events.setdefault(key, len(self.events) + 1)
                self.journal.write(
                    json.dumps({
                        'time': time.time(), 'path': self.path,
                        'bytes': size, 'event_id': key,
                        'violation_id': violation,
                    }) + '\n',
                )
                self.journal.flush()
            response = {
                'violation_id': violation, 'success': True,
                'outcome': 'completed', 'image_path': 'isolated-fixture.jpg',
            }
        payload = json.dumps(response).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=28003)
    parser.add_argument('--journal', type=Path, required=True)
    args = parser.parse_args()
    with args.journal.open('a') as stream:
        Handler.journal = stream
        ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()


if __name__ == '__main__':
    main()
