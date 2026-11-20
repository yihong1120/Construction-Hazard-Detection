"""Measure metadata delivery over loopback HTTP and Redis."""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import platform
import shutil
import socket
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from dataclasses import field
from importlib.metadata import version
from pathlib import Path

import aiohttp
import redis.asyncio as redis

from examples.streaming_web.metadata_keys import build_metadata_key

ROOT = Path(__file__).resolve().parents[2]
VARIANTS = ('python', 'python-shared', 'rust')
TICKS_PER_SECOND = os.sysconf('SC_CLK_TCK')


def available_port() -> int:
    with socket.socket() as listener:
        listener.bind(('127.0.0.1', 0))
        return listener.getsockname()[1]


@asynccontextmanager
async def run_process(
    command: list[str],
    log: Path,
) -> AsyncIterator[asyncio.subprocess.Process]:
    with log.open('wb') as output:
        child = await asyncio.create_subprocess_exec(
            *command, cwd=ROOT, stdout=output,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            yield child
        except Exception as error:
            error.add_note(
                f'{log.name} tail:\n{log.read_text(errors="replace")[-4000:]}',
            )
            raise
        finally:
            if child.returncode is None:
                child.terminate()
                try:
                    await asyncio.wait_for(child.wait(), timeout=5)
                except TimeoutError:
                    child.kill()
                    await child.wait()


def check_alive(child: asyncio.subprocess.Process) -> None:
    if child.returncode is not None:
        raise RuntimeError(
            f'Process {child.pid} exited with code {child.returncode}',
        )


async def wait_for_redis(
    connection: redis.Redis,
    child: asyncio.subprocess.Process,
) -> None:
    async with asyncio.timeout(15):
        while True:
            check_alive(child)
            try:
                if await connection.ping():
                    return
            except redis.ConnectionError:
                pass
            await asyncio.sleep(0.05)


async def wait_for_subscribers(
    client: aiohttp.ClientSession,
    base_url: str,
    count: int,
    child: asyncio.subprocess.Process,
    viewer_tasks: list[asyncio.Task[None]] | None = None,
) -> None:
    async with asyncio.timeout(15):
        while True:
            check_alive(child)
            if viewer_tasks:
                validate_viewer_tasks(viewer_tasks)
            try:
                async with client.get(f'{base_url}/health') as response:
                    state = await response.json()
                    if (
                        response.status == 200 and state['status'] == 'ok'
                        and state['subscribers'] == count
                    ):
                        return
            except aiohttp.ClientConnectionError:
                pass
            await asyncio.sleep(0.05)


def process_cpu_seconds(pid: int) -> float:
    # /proc stat comm is parenthesised and can include spaces or parentheses.
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return (int(fields[11]) + int(fields[12])) / TICKS_PER_SECOND


def process_memory_kib(pid: int) -> tuple[int, int]:
    values = {}
    for line in Path(f'/proc/{pid}/smaps_rollup').read_text().splitlines():
        key, _, value = line.partition(':')
        if key in ('Rss', 'Pss'):
            values[key] = int(value.split()[0])
    return values['Rss'], values['Pss']


def percentiles(samples: list[float]) -> dict[str, float | None]:
    ordered = sorted(samples)
    return {
        f'p{rank}': (
            round(ordered[math.ceil(rank / 100 * len(ordered)) - 1], 3)
            if ordered else None
        )
        for rank in (50, 95, 99)
    }


def camera_overlay(camera: int, detections: int) -> dict[str, object]:
    return {
        'camera': camera,
        'owner': 'benchmark',
        'generation': 1,
        'frame_width': 1920,
        'frame_height': 1080,
        'labels': {
            '0': {'zh-TW': '人員', 'en': 'Person'},
            '2': {'zh-TW': '未戴安全帽', 'en': 'No Helmet'},
        },
        'track_data': [
            [
                40 + index * 25, 70 + camera * 3, 100 + index * 25,
                250 + camera * 3, 0.93, 2 if index % 3 == 0 else 0, index,
            ]
            for index in range(detections)
        ],
        'cone_polys': [[[17, 22], [98, 22], [107, 96], [15, 67]]],
        'pole_polys': [],
        'warnings': {'no_safety_helmet': {'count': 1}},
    }


@dataclass
class Viewer:
    camera: int
    overlay_template: dict[str, object]
    first_sequence: int
    end_sequence: int
    slow_seconds: float = 0.0
    last_sequence: int = -1
    bytes_received: int = 0
    first_event: asyncio.Event = field(default_factory=asyncio.Event)
    latencies_ms: list[float] = field(default_factory=list)

    async def receive(
        self,
        data: bytes | str,
        message_id: str | None = None,
    ) -> None:
        receive_ns = time.monotonic_ns()
        payload = json.loads(data)
        if message_id is not None and payload.pop('id', None) != message_id:
            raise RuntimeError('SSE envelope and payload IDs differ')
        if set(payload) != {'has_warning', 'overlay'}:
            raise RuntimeError(
                f'Unexpected metadata payload keys {set(payload)}',
            )
        overlay = payload['overlay']
        sequence = overlay['sequence']
        published_ns = overlay['published_ns']
        if (
            type(sequence) is not int or sequence <= self.last_sequence
            or type(published_ns) is not int or published_ns > receive_ns
            or overlay['camera'] != self.camera
            or payload['has_warning'] is not (sequence % 3 == 0)
        ):
            raise RuntimeError(
                f'Invalid metadata event for camera {self.camera}',
            )
        if self.last_sequence < 0:
            expected_overlay = {
                **self.overlay_template, 'sequence': sequence,
                'published_ns': published_ns,
            }
            if overlay != expected_overlay:
                raise RuntimeError(
                    f'Overlay parity failed on camera {self.camera}',
                )
            self.first_event.set()
        self.last_sequence = sequence
        if self.first_sequence <= sequence < self.end_sequence:
            self.latencies_ms.append((receive_ns - published_ns) / 1_000_000)
            self.bytes_received += (
                len(data.encode('utf-8'))
                if isinstance(data, str) else len(data)
            )
        if self.slow_seconds:
            await asyncio.sleep(self.slow_seconds)


async def consume(
    client: aiohttp.ClientSession,
    protocol: str,
    base_url: str,
    viewer: Viewer,
) -> None:
    url = f'{base_url}/{protocol}/{viewer.camera}'
    if protocol == 'ws':
        async with client.ws_connect(url, compress=0) as response:
            async for message in response:
                if message.type != aiohttp.WSMsgType.TEXT:
                    raise RuntimeError(
                        f'Unexpected WebSocket message {message.type}',
                    )
                await viewer.receive(message.data)
        raise RuntimeError('WebSocket stream closed during measurement')

    async with client.get(url) as response:
        response.raise_for_status()
        if response.content_type != 'text/event-stream':
            raise RuntimeError(
                f'Unexpected SSE content type {response.content_type}',
            )
        event_id = None
        data = None
        event_type = None
        async for line in response.content:
            if line.startswith(b'id: '):
                event_id = line[4:].rstrip(b'\r\n').decode('ascii')
            elif line.startswith(b'event: '):
                event_type = line[7:].rstrip(b'\r\n')
            elif line.startswith(b'data: '):
                data = line[6:].rstrip(b'\r\n')
            elif line in (b'\n', b'\r\n'):
                if data is not None:
                    if event_type != b'metadata' or not event_id:
                        raise RuntimeError(
                            'SSE metadata event type or ID missing',
                        )
                    await viewer.receive(data, event_id)
                data = event_id = event_type = None
    raise RuntimeError('SSE stream closed during measurement')


async def publish(
    connection: redis.Redis,
    keys: list[str],
    overlays: list[dict[str, object]],
    first_sequence: int,
    count: int,
    rate: float,
) -> list[float]:
    clock = asyncio.get_running_loop()
    start = clock.time()
    lateness_ms: list[float] = []
    for offset in range(count):
        deadline = start + offset / rate
        await asyncio.sleep(max(0, deadline - clock.time()))
        lateness_ms.append(max(0, clock.time() - deadline) * 1000)
        sequence = first_sequence + offset
        timestamp = time.monotonic_ns()
        async with connection.pipeline(transaction=False) as batch:
            for key, template in zip(keys, overlays, strict=True):
                overlay = {
                    **template,
                    'sequence': sequence,
                    'published_ns': timestamp,
                }
                fields = {
                    'has_warning': '1' if sequence % 3 == 0 else '0',
                    'overlay': json.dumps(
                        overlay, ensure_ascii=False, separators=(',', ':'),
                    ),
                }
                batch.xadd(key, fields, maxlen=256)
            await batch.execute()
    await asyncio.sleep(max(0, start + count / rate - clock.time()))
    return lateness_ms


def start_service(
    args: argparse.Namespace,
    variant: str,
    port: int,
    redis_url: str,
) -> list[str]:
    options = [
        '--port', str(port), '--redis-url', redis_url,
        '--streams', str(args.cameras),
    ]
    if variant == 'rust':
        return [str(args.rust_binary), *options, '--workers', '1']
    command = [
        sys.executable, '-m',
        'experiments.metadata_api_benchmark.python_service',
        *options,
    ]
    return command + (['--shared'] if variant == 'python-shared' else [])


async def sample_memory(pid: int, samples: list[tuple[int, int]]) -> None:
    while True:
        samples.append(process_memory_kib(pid))
        await asyncio.sleep(0.25)


def validate_viewer_tasks(tasks: list[asyncio.Task[None]]) -> None:
    for task in tasks:
        if task.done():
            task.result()
            raise RuntimeError('Viewer returned before the benchmark ended')


def delivery_stats(
    viewers: list[Viewer],
    ticks: int,
) -> dict[str, object]:
    latencies = [
        sample for viewer in viewers for sample in viewer.latencies_ms
    ]
    expected = ticks * len(viewers)
    payload_bytes = sum(viewer.bytes_received for viewer in viewers)
    return {
        'delivery_expected': expected,
        'delivery_observed': len(latencies),
        'delivery_percent': (
            round(100 * len(latencies) / expected, 3) if expected else None
        ),
        'payload_bytes_observed': payload_bytes,
        'latency_ms': percentiles(latencies),
    }


async def measure(
    args: argparse.Namespace,
    client: aiohttp.ClientSession,
    connection: redis.Redis,
    child: asyncio.subprocess.Process,
    base_url: str,
    protocol: str,
    viewers_per_camera: int,
) -> dict[str, object]:
    overlays = [
        camera_overlay(camera, args.detections)
        for camera in range(args.cameras)
    ]
    keys = [
        build_metadata_key('benchmark', f'camera-{camera:03d}')
        for camera in range(args.cameras)
    ]
    first = max(1, round(args.warmup * args.events_per_second))
    ticks = max(1, round(args.duration * args.events_per_second))
    delay = args.slow_client_delay_ms / 1000
    viewers = [
        Viewer(
            camera, overlay, first, first + ticks,
            delay if index == viewers_per_camera - 1 else 0.0,
        )
        for camera, overlay in enumerate(overlays)
        for index in range(viewers_per_camera)
    ]
    tasks = [
        asyncio.create_task(consume(client, protocol, base_url, viewer))
        for viewer in viewers
    ]
    try:
        await wait_for_subscribers(
            client, base_url, len(viewers), child, tasks,
        )
        # Let the first blocking XREAD become established before publishing.
        await asyncio.sleep(0.1)
        await publish(
            connection, keys, overlays, 0, first, args.events_per_second,
        )
        validate_viewer_tasks(tasks)
        if not all(viewer.first_event.is_set() for viewer in viewers):
            raise RuntimeError('Some subscribers received no warm-up payload')

        samples = [process_memory_kib(child.pid)]
        sampler = asyncio.create_task(sample_memory(child.pid, samples))
        cpu_start = process_cpu_seconds(child.pid)
        start = time.monotonic()
        try:
            lateness = await publish(
                connection, keys, overlays, first, ticks,
                args.events_per_second,
            )
            wall_seconds = time.monotonic() - start
            cpu_seconds = process_cpu_seconds(child.pid) - cpu_start
            samples.append(process_memory_kib(child.pid))
        finally:
            sampler.cancel()
            await asyncio.gather(sampler, return_exceptions=True)
        await asyncio.sleep(args.drain)
        validate_viewer_tasks(tasks)
    finally:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
    await wait_for_subscribers(client, base_url, 0, child)
    peak_rss = max(rss for rss, _ in samples) / 1024
    peak_pss = max(pss for _, pss in samples) / 1024
    result = {
        'protocol': protocol,
        'cameras': args.cameras,
        'viewers_per_camera': viewers_per_camera,
        'connections': len(viewers),
        'wall_seconds': round(wall_seconds, 4),
        'published': ticks * args.cameras,
        **delivery_stats(viewers, ticks),
        'publish_lateness_ms': percentiles(lateness),
        'api_cpu_seconds': round(cpu_seconds, 4),
        'api_cpu_percent_one_core': round(100 * cpu_seconds / wall_seconds, 3),
        'api_peak_sampled_rss_mib': round(peak_rss, 3),
        'api_peak_sampled_pss_mib': round(peak_pss, 3),
    }
    if delay:
        result['regular_viewers'] = delivery_stats(
            [viewer for viewer in viewers if not viewer.slow_seconds], ticks,
        )
        result['paced_viewers'] = delivery_stats(
            [viewer for viewer in viewers if viewer.slow_seconds], ticks,
        )
    return result


async def run_case(
    args: argparse.Namespace,
    variant: str,
    protocol: str,
    viewers_per_camera: int,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix='chd-metadata-') as temp:
        directory = Path(temp)
        redis_port = available_port()
        redis_url = f'redis://127.0.0.1:{redis_port}/0'
        redis_command = [
            args.redis_server, '--bind', '127.0.0.1', '--port', str(
                redis_port,
            ),
            '--protected-mode', 'yes', '--save', '', '--appendonly', 'no',
            '--dir', str(directory), '--loglevel', 'warning',
        ]
        async with (
            run_process(redis_command, directory / 'redis.log') as redis_child,
            redis.Redis.from_url(redis_url) as connection,
            aiohttp.ClientSession(
                connector=aiohttp.TCPConnector(limit=0, force_close=True),
                timeout=aiohttp.ClientTimeout(total=None, sock_connect=5),
                read_bufsize=4096,
            ) as client,
        ):
            await wait_for_redis(connection, redis_child)
            port = available_port()
            command = start_service(args, variant, port, redis_url)
            async with run_process(command, directory / 'api.log') as child:
                base_url = f'http://127.0.0.1:{port}'
                await wait_for_subscribers(client, base_url, 0, child)
                return await measure(
                    args, client, connection, child,
                    base_url, protocol, viewers_per_camera,
                )


async def benchmark(args: argparse.Namespace) -> dict[str, object]:
    results = []
    for protocol in args.protocol:
        for viewers in args.viewers_per_camera:
            for repeat in range(args.repeats):
                order = repeat % len(args.variants)
                for variant in args.variants[order:] + args.variants[:order]:
                    result = await run_case(args, variant, protocol, viewers)
                    result.update(variant=variant, repeat=repeat + 1)
                    results.append(result)
                    print(json.dumps(result), flush=True)
    return {
        'host': platform.platform(),
        'python': sys.version.split()[0],
        'python_dependencies': {
            name: version(name)
            for name in ('aiohttp', 'fastapi', 'redis', 'starlette', 'uvicorn')
        },
        'events_per_second_per_camera': args.events_per_second,
        'detections_per_frame': args.detections,
        'requested_duration_seconds': args.duration,
        'warmup_seconds': args.warmup,
        'drain_seconds': args.drain,
        'slow_client_delay_ms': args.slow_client_delay_ms,
        'results': results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cameras', type=int, default=32)
    parser.add_argument(
        '--viewers-per-camera', type=int,
        nargs='+', default=[1, 4],
    )
    parser.add_argument(
        '--protocol', choices=('sse', 'ws'),
        nargs='+', default=['sse'],
    )
    parser.add_argument(
        '--variants', choices=VARIANTS,
        nargs='+', default=list(VARIANTS),
    )
    parser.add_argument('--rust-binary', type=Path)
    parser.add_argument('--redis-server', default='redis-server')
    parser.add_argument('--events-per-second', type=float, default=15)
    parser.add_argument('--detections', type=int, default=12)
    parser.add_argument('--duration', type=float, default=8)
    parser.add_argument('--warmup', type=float, default=2)
    parser.add_argument('--drain', type=float, default=0.25)
    parser.add_argument('--slow-client-delay-ms', type=float, default=0)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if not shutil.which(args.redis_server):
        parser.error('redis-server is required')
    if 'rust' in args.variants:
        if not args.rust_binary or not os.access(args.rust_binary, os.X_OK):
            parser.error('select the Rust executable with --rust-binary')
        args.rust_binary = args.rust_binary.resolve()
    positive = [
        args.cameras, args.events_per_second, args.duration, args.warmup,
        args.repeats, *args.viewers_per_camera,
    ]
    nonnegative = [args.detections, args.drain, args.slow_client_delay_ms]
    if not all(math.isfinite(value) and value > 0 for value in positive):
        parser.error(
            'cameras, viewers, rate, duration and repeats must be positive',
        )
    if not all(math.isfinite(value) and value >= 0 for value in nonnegative):
        parser.error(
            'detections, drain and slow client delay must be non-negative',
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(benchmark(args))
    args.output.write_text(
        json.dumps(
            result, indent=2,
        ) + '\n', encoding='utf-8',
    )


if __name__ == '__main__':
    main()
