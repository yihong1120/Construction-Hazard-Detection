"""Compare capture admission on an isolated 30 FPS RTSP feed."""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

import cv2

from src.stream_capture import StreamCapture


def load(path):
    spec = importlib.util.spec_from_file_location('legacy_capture_probe', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.StreamCapture


async def trial(factory, args):
    capture = factory(args.source, capture_interval=1 / 30)
    count = 0
    original = capture._run_capture_operation

    async def operation(call, *values):
        nonlocal count
        result = await original(call, *values)
        if getattr(call, '__name__', '') == 'read' and result[0]:
            count += 1
        return result

    capture._run_capture_operation = operation
    capture.freeze_reconnect_seconds = 0
    capture.timestamp_reconnect_seconds = 0
    generator = capture.execute_capture()
    started = None
    measured = []
    before = 0
    async with asyncio.timeout(30):
        try:
            async for _, _ in generator:
                now = time.monotonic()
                if started is None:
                    started = now
                if now - started < args.warmup:
                    before = count
                    continue
                measured.append(now)
                if measured[-1] - measured[0] >= args.seconds:
                    break
        finally:
            await generator.aclose()
    duration = measured[-1] - measured[0]
    return {
        'admitted_fps': (len(measured) - 1) / duration,
        'admitted_frames': len(measured),
        'successful_reads_including_warmup': count,
        'read_since_last_warmup_frame': count - before,
        'window_seconds': duration,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--legacy-source', type=Path, required=True)
    parser.add_argument('--source', default='rtsp://127.0.0.1:28554/cadence')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=float, default=2)
    parser.add_argument('--seconds', type=float, default=8)
    args = parser.parse_args()
    assert urlsplit(args.source).hostname == '127.0.0.1'
    cv2.setNumThreads(1)
    original = cv2.VideoCapture
    cv2.VideoCapture = lambda src, api, params: original(
        src, api, [*params, cv2.CAP_PROP_N_THREADS, 1],
    )
    factories = {'before': load(args.legacy_source), 'after': StreamCapture}
    rows = []
    for index in range(args.rounds):
        order = ['before', 'after'] if index % 2 == 0 else ['after', 'before']
        for name in order:
            row = {
                'round': index, 'version': name,
                **asyncio.run(trial(factories[name], args)),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    result = {
        'scope': 'Loopback 30 FPS RTSP, capture admission only; '
        'same OpenCV decode, no inference or browser. Source has '
        '24.7 FPS footage resampled to 30 FPS.', 'trials': rows,
        'median_admitted_fps': {
            name: statistics.median(
                r['admitted_fps'] for r in rows
                if r['version'] == name
            )
            for name in factories
        },
    }
    args.output.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
