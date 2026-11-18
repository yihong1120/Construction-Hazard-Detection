"""Instrument main.py without replacing its camera processing stages."""
from __future__ import annotations

import asyncio
import json
import logging
import multiprocessing
import os
import signal
import time
from pathlib import Path

import cv2

CAPTURE = cv2.VideoCapture
if os.environ.get('CHD_BENCH_LIMIT_THREADS') == '1':
    cv2.setNumThreads(1)

ORIGINAL_RECORD = None
TRACE = None


def limited_capture(source, api, params):
    return CAPTURE(source, api, [*params, cv2.CAP_PROP_N_THREADS, 1])


if os.environ.get('CHD_BENCH_LIMIT_THREADS') == '1':
    cv2.VideoCapture = limited_capture  # type: ignore[misc,assignment]


class ProbeDanger:
    def __init__(self, inner):
        self.inner = inner
        self.result = None
        self.flushed = time.monotonic()

    def detect_danger(self, tracks):
        warnings, cones, poles = self.inner.detect_danger(tracks)
        result = self.result
        completed = time.time()
        TRACE.write(
            json.dumps({
                'sequence': result.sequence,
                'timestamp_ms': int(result.timestamp * 1000),
                'completed_timestamp_ms': int(completed * 1000),
                'frame_age_ms': (completed - result.timestamp) * 1000,
                'tracks': tracks, 'warnings': warnings,
                'cone_polys': cones, 'pole_polys': poles,
            }) + '\n',
        )
        if time.monotonic() - self.flushed >= 1:
            TRACE.flush()
            self.flushed = time.monotonic()
        return warnings, cones, poles


async def record(runtime, result):
    if not isinstance(runtime.danger, ProbeDanger):
        runtime.danger = ProbeDanger(runtime.danger)
    runtime.danger.result = result
    return await ORIGINAL_RECORD(runtime, result)


def camera(cfg, requests, results):
    global ORIGINAL_RECORD, TRACE
    from src import stream_detection
    from src.stream_processor import process_single_stream

    logging.disable(logging.INFO)
    path = Path(os.environ['CHD_BENCH_TRACE_DIR'])
    path.mkdir(parents=True, exist_ok=True)
    TRACE = (
        path / f'detections-camera-{cfg["stream_id"]}.jsonl'
    ).open('w', buffering=8192)
    ORIGINAL_RECORD = stream_detection._record_detection_result
    stream_detection._record_detection_result = record
    try:
        process_single_stream(cfg, requests, results)
    finally:
        TRACE.close()


def main():
    import main as application
    from src import worker_supervisor

    logging.disable(logging.INFO)
    multiprocessing.set_start_method('spawn', force=True)
    worker_supervisor.process_single_stream = camera
    signal.signal(signal.SIGTERM, application._handle_sigterm)
    asyncio.run(application.main())


if __name__ == '__main__':
    main()
