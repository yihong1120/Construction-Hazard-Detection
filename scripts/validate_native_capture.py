"""Validate native capture against an isolated local RTSP server and worker."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
from functools import partial
from pathlib import Path
from uuid import uuid4

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def fixture_registry(value):
    return value


def inference_worker(requests, replies, engine, digest, status):
    from examples.YOLO_server_api.model_registry import ModelEntry
    from examples.YOLO_server_api.model_registry import ModelRegistry
    from src import yolo_worker
    entry = ModelEntry(
        id='native-capture-test', display_name='Isolated capture test',
        version='fixture', artifact=engine, sha256=digest,
        capabilities=['stream'], classes=[],
    )
    yolo_worker.load_registry = partial(
        fixture_registry, ModelRegistry(revision='fixture', models=[entry]),
    )
    worker = yolo_worker.YoloWorker(requests, result_queues=replies)
    worker.run()
    status.put({
        'borrowed_frames': worker.native_borrowed_frames,
        'snapshot_bytes': worker.native_snapshot_bytes,
        'torch_loaded': 'torch' in sys.modules,
    })


async def send_frames(captures, requests, replies):
    from src.yolo_worker import YoloWorkerClient
    clients = [
        YoloWorkerClient(requests, reply, name, timeout_seconds=60)
        for name, reply in replies.items()
    ]
    held = []
    try:
        for _ in range(3):
            frames = []
            for capture in captures:
                ok, frame = capture.read()
                assert ok and not frame.flags.writeable
                frames.append(frame)
            held.append((frames[0], frames[0].copy()))
            if requests is not None:
                result = await asyncio.gather(
                    *[
                        client.detect(frame, 'native-capture-test')
                        for client, frame in zip(clients, frames, strict=True)
                    ],
                )
                assert len(result) == 32
            await asyncio.sleep(.25)
        return held
    finally:
        await asyncio.gather(*[client.close() for client in clients])


def main():
    from src.native_capture import RustCapture
    from src.settings.capture import CaptureSettings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', type=Path)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--image', default='bluenviron/mediamtx:latest')
    logs = ROOT / 'outputs/native_capture_validation'
    parser.add_argument(
        '--output',
        type=Path,
        default=logs / 'capture_validation.json',
    )
    args = parser.parse_args()
    name = 'chd-native-capture-' + uuid4().hex[:8]
    captures, publisher, process, queues = [], None, None, []
    requests, replies, status = None, {}, None
    logs.mkdir(exist_ok=True, parents=True)
    log = (logs / 'publisher.log').open('w')
    started = time.monotonic()
    try:
        subprocess.run(
            [
                'docker', 'run', '--rm', '-d', '--name', name,
                '-p', '127.0.0.1::8554', '-e', 'MTX_RTSPTRANSPORTS=tcp',
                '-e', 'MTX_HLS=no', '-e', 'MTX_WEBRTC=no',
                '-e', 'MTX_RTMP=no', '-e', 'MTX_SRT=no', args.image,
            ], check=True, stdout=subprocess.DEVNULL,
        )
        port = subprocess.check_output(
            ['docker', 'port', name, '8554/tcp'], text=True,
        ).strip().rsplit(':', 1)[1]
        url = f'rtsp://127.0.0.1:{port}/native-test'
        publisher = subprocess.Popen(
            [
                'ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error',
                '-re', '-stream_loop', '-1', '-i', str(
                    ROOT / 'tests/videos/test.mp4',
                ),
                '-an', '-c:v', 'libx264', '-preset', 'ultrafast',
                '-tune', 'zerolatency', '-pix_fmt', 'yuv420p', '-r', '15',
                '-g', '15', '-f', 'rtsp', '-rtsp_transport', 'tcp', url,
            ], stdout=subprocess.DEVNULL, stderr=log,
        )
        settings = CaptureSettings(backend='rust_ffmpeg')
        for attempt in range(10):
            try:
                captures.append(RustCapture(url, settings))
                assert captures[0].read()[0]
                break
            except RuntimeError:
                if attempt == 9:
                    raise
                time.sleep(.2)
        if args.engine:
            context = multiprocessing.get_context('spawn')
            requests = context.Queue(maxsize=64)
            replies = {
                f'camera-{index}': context.Queue(maxsize=4)
                for index in range(32)
            }
            status = context.Queue(maxsize=1)
            queues = [requests, status, *replies.values()]
            os.environ['YOLO_WORKER_BACKEND'] = 'rust'
            os.environ['YOLO_WORKER_PRECISION'] = 'int8'
            os.environ['YOLO_WORKER_BATCH_SIZE'] = '8'
            os.environ['YOLO_WORKER_ARTIFACT_MANIFEST'] = ''
            engine = args.engine.resolve()
            digest = hashlib.sha256(engine.read_bytes()).hexdigest()
            process = context.Process(
                target=inference_worker, args=(
                    requests, replies, str(engine), digest, status,
                ),
            )
            process.start()
        for _ in range(31):
            captures.append(RustCapture(url, settings))
        held = asyncio.run(send_frames(captures, requests, replies))
        counters = [capture.capture.counters for capture in captures]
        assert all(received > 0 for received, _ in counters)
        assert sum(dropped for _, dropped in counters) > 0
        cuda_validated = False
        if args.cuda:
            cuda = RustCapture(
                url, CaptureSettings(
                    backend='rust_ffmpeg', decoder='cuda',
                ),
            )
            try:
                ok, frame = cuda.read()
                assert ok and frame.shape[2] == 3
                cuda_validated = True
            finally:
                cuda.release()
        worker = None
        if process is not None:
            from src.yolo_worker_protocol import YOLO_WORKER_STOP_MESSAGE
            requests.put(YOLO_WORKER_STOP_MESSAGE)
            worker = status.get(timeout=30)
            process.join(timeout=30)
            assert process.exitcode == 0
            assert worker == {
                'borrowed_frames': 96,
                'snapshot_bytes': 0,
                'torch_loaded': False,
            }
        for capture in captures:
            capture.release()
        for frame, expected in held:
            np.testing.assert_array_equal(frame, expected)
        result = {
            'scope': (
                'One local H264 RTSP publisher, 32 independent Rust/FFmpeg '
                'subscribers, 3 rounds; functional test, '
                'not a capacity benchmark'
            ),
            'cuda_decode_host_download_validated': cuda_validated,
            'end_to_end_shared_worker': worker,
            'captures': 32, 'mailbox_capacity_per_capture': 1,
            'counters_received_dropped': counters,
            'exported_frames_survive_close': True,
            'elapsed_seconds': time.monotonic() - started,
            'server_image_id': subprocess.check_output(
                ['docker', 'inspect', name, '--format', '{{.Image}}'],
                text=True,
            ).strip(),
        }
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result, indent=2))
    finally:
        for capture in captures:
            capture.release()
        if process is not None and process.is_alive():
            process.terminate()
            process.join(timeout=10)
        for queue in queues:
            queue.close()
            queue.join_thread()
        if publisher is not None:
            publisher.terminate()
            try:
                publisher.wait(timeout=5)
            except subprocess.TimeoutExpired:
                publisher.kill()
                publisher.wait()
        log.close()
        (logs / 'server.log').write_bytes(
            subprocess.run(
                ['docker', 'logs', name], capture_output=True,
            ).stderr,
        )
        subprocess.run(
            ['docker', 'rm', '-f', name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


if __name__ == '__main__':
    main()
