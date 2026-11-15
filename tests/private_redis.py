"""Private Unix-socket Redis fixture, isolated from developer services."""
from __future__ import annotations

import asyncio
import shutil
import subprocess
import tempfile
from pathlib import Path

from redis.asyncio import Redis


def stop_redis_process(process):
    process.terminate()
    process.wait(timeout=5)


class RedisFixture:
    async def start_redis(self):
        binary = shutil.which('redis-server')
        if not binary:
            self.skipTest('redis-server is needed for atomic session tests')
        temporary = tempfile.TemporaryDirectory(
            prefix='construction-counter-redis-',
        )
        self.addCleanup(temporary.cleanup)
        socket = str(Path(temporary.name) / 'redis.sock')
        process = subprocess.Popen(
            [
                binary,
                '--port',
                '0',
                '--unixsocket',
                socket,
                '--unixsocketperm',
                '700',
                '--save',
                '',
                '--appendonly',
                'no',
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        self.addCleanup(stop_redis_process, process)
        redis = Redis(unix_socket_path=socket)
        self.addAsyncCleanup(redis.aclose)
        for _ in range(200):
            try:
                await redis.ping()
                return redis
            except Exception:
                await asyncio.sleep(0.01)
        self.fail('Private Redis did not start')
