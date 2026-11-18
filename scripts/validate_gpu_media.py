"""Validate experimental device media using a local video."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import av
import numpy as np

from src.experimental_gpu_media import ExperimentalGpuMedia

ROOT = Path(__file__).resolve().parents[1]


def preflight(pipeline, decoded):
    frame = pipeline.import_frame(decoded)
    try:
        shape = (frame.height, frame.width, 3)
        actual = np.frombuffer(frame.download_bgr(), dtype=np.uint8)
        actual = actual.reshape(shape)
        legacy = decoded.reformat(format='bgr24').to_ndarray()
        point = decoded.reformat(
            format='bgr24', interpolation='POINT',
        ).to_ndarray()
        reference = pipeline.model.predict_detections([actual])
        device = pipeline.predict([frame])
        legacy_rows = pipeline.model.predict_detections([legacy])
        return {
            'identical_bgr_detection_parity': reference == device,
            'legacy_detection_exact_parity': legacy_rows == device,
            'legacy_bgr_max_absolute_error': int(
                np.max(
                    np.abs(
                        actual.astype(np.int16) - legacy.astype(np.int16),
                    ),
                ),
            ),
            'point_bgr_max_absolute_error': int(
                np.max(
                    np.abs(
                        actual.astype(np.int16) - point.astype(np.int16),
                    ),
                ),
            ),
            'legacy_color_gate_passed': bool(np.array_equal(actual, legacy)),
            'diagnostic_downloads_excluded_from_hot_path': True,
            'width': frame.width, 'height': frame.height,
            'source_colorspace': decoded.colorspace,
            'source_color_range': decoded.color_range,
        }
    finally:
        frame.close()


def validate(args):
    args.output.parent.mkdir(parents=True, exist_ok=True)
    timings = []
    with ExperimentalGpuMedia(args.engine) as pipeline:
        with pipeline.open_source(str(args.video)) as source:
            iterator = source.decode(video=0)
            parity = preflight(pipeline, next(iterator))
            assert parity['identical_bgr_detection_parity']
            factory = next(iter(pipeline.factories.values()))
            before_factory = list(factory.stats)
            before_engine = list(pipeline.model.engine.async_stats)
            started_cpu = time.process_time_ns()
            started = time.perf_counter_ns()
            with args.output.open('wb') as output:
                for index, decoded in enumerate(iterator):
                    if index >= args.frames:
                        break
                    frame_started = time.perf_counter_ns()
                    frame = pipeline.import_frame(decoded)
                    try:
                        rows = pipeline.predict([frame])[0]
                        rendered = pipeline.annotate(frame, rows)
                        try:
                            output.write(pipeline.encode(rendered))
                        finally:
                            rendered.close()
                    finally:
                        frame.close()
                    timings.append(
                        (time.perf_counter_ns() - frame_started) / 1e6,
                    )
                output.write(pipeline.flush())
            elapsed = (time.perf_counter_ns() - started) / 1e9
            cpu_seconds = (time.process_time_ns() - started_cpu) / 1e9
            downloads = factory.stats[1] - before_factory[1]
            uploads = pipeline.model.engine.async_stats[0] - before_engine[0]
            counters = {
                'raw_frame_download_bytes': downloads,
                'inference_host_upload_bytes': uploads,
                'glyph_upload_bytes': pipeline.glyph_upload_bytes,
                'compressed_output_bytes': pipeline.encoded_bytes,
                'frames': len(timings), 'elapsed_seconds': elapsed,
                'cpu_seconds': cpu_seconds,
                'single_stream_fps': len(timings) / elapsed,
                'frame_latency_ms': timings,
            }
            assert downloads == uploads == 0
    with av.open(str(args.output), format='h264') as encoded:
        shapes = [(frame.width, frame.height) for frame in encoded.decode(0)]
    assert len(shapes) == len(timings)
    assert set(shapes) == {(parity['width'], parity['height'])}
    return {
        'status': 'experimental; legacy color gate required before rollout',
        'scope': (
            'Local video NVDEC, D2D-owned NV12/BGR, integer downscale, '
            'TensorRT, GPU glyph blending, direct GPU NVENC. Encoded bytes '
            'reach host; diagnostic color preflight downloads excluded. '
            'Device-wide barrier required by PyAV 17 DLPack export. '
            'Single stream smoke; not a 32-camera capacity measurement.'
        ), 'parity': parity, 'hot_path': counters,
        'decoded_output_frames': len(shapes),
        'video_sha256': hashlib.sha256(args.video.read_bytes()).hexdigest(),
        'engine_sha256': hashlib.sha256(args.engine.read_bytes()).hexdigest(),
        'source_sha256': {
            str(path.relative_to(ROOT)): hashlib.sha256(
                path.read_bytes(),
            ).hexdigest() for path in [
                Path(__file__).resolve(),
                ROOT / 'src/experimental_gpu_media.py',
                ROOT / 'src/gpu_color_lookup.py',
                ROOT / 'native/inference/gpu_frames.inc',
                ROOT / 'native/inference/gpu_api.inc',
                ROOT / 'native/inference/gpu_encoder.inc',
                ROOT / 'native/inference/preprocess.cu',
                ROOT / 'native/inference/src/gpu.rs',
            ]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frames', type=int, default=32)
    parser.add_argument(
        '--video', type=Path, default=ROOT / 'tests/videos/test.mp4',
    )
    parser.add_argument(
        '--engine', type=Path,
        default=ROOT / 'models/int8_engine/best_yolo26n.engine',
    )
    parser.add_argument(
        '--output', type=Path, default=ROOT / 'outputs/native_gpu_test.h264',
    )
    parser.add_argument(
        '--report', type=Path,
        default=ROOT / 'native/inference/gpu_media_results.json',
    )
    args = parser.parse_args()
    if args.frames < 1:
        parser.error('--frames must be positive')
    result = validate(args)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
