"""Matched live RTSP comparison of main.py and the standalone Rust binary."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen

import asyncpg
import numpy as np
import psutil
import pynvml
import redis

ROOT = Path(__file__).resolve().parents[2]
DB = 'postgresql://fixture:fixture@127.0.0.1:25433/chd_e2e_benchmark'
RTSP = 'rtsp://127.0.0.1:28554'
API = 'http://127.0.0.1:29997/v3/paths/list?itemsPerPage=1000'
MOCK = 'http://127.0.0.1:28003'
RULES = dict.fromkeys(
    (
        'detect_no_safety_vest_or_helmet', 'detect_near_machinery_or_vehicle',
        'detect_in_restricted_area', 'detect_in_utility_pole_restricted_area',
        'detect_machinery_close_to_pole',
    ), True,
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def paths():
    with urlopen(API, timeout=5) as response:
        return {row['name']: row for row in json.load(response)['items']}


async def catalog(engine):
    from examples.YOLO_server_api.model_registry import ModelEntry

    checkpoint = ROOT / 'models/pt/best_yolo26n.pt'
    codes = [
        'helmet', 'mask', 'no_helmet', 'no_mask', 'no_safety_vest',
        'person', 'safety_cone', 'safety_vest', 'machinery',
        'utility_pole', 'vehicle',
    ]
    definition = ModelEntry(
        id='yolo26n', display_name='Isolated benchmark', version='fixture-v1',
        artifact=str(checkpoint), sha256=digest(checkpoint),
        capabilities=['stream'], tenant_ids=['fixture'], site_ids=[1],
        classes=[
            {'id': i, 'code': code, 'display_name': code.replace('_', ' ')}
            for i, code in enumerate(codes)
        ],
    ).model_dump(exclude={'id'})
    db = await asyncpg.connect(DB)
    try:
        await db.execute('''CREATE TABLE IF NOT EXISTS detection_model_catalog(
            model_key text PRIMARY KEY, definition jsonb NOT NULL,
            is_default bool NOT NULL DEFAULT false,
            display_order int NOT NULL DEFAULT 0)''')
        await db.execute(
            '''INSERT INTO detection_model_catalog
            (model_key,definition) VALUES('yolo26n',$1)
            ON CONFLICT(model_key) DO UPDATE
            SET definition=EXCLUDED.definition''',
            json.dumps(definition),
        )
    finally:
        await db.close()
    return definition, {
        'format': 1, 'models': {
            'yolo26n': {
                'artifact': str(engine), 'sha256': digest(engine),
                'source_sha256': digest(checkpoint), 'version': 'fixture-v1',
            },
        },
    }


def configurations(args, root, cameras, definition, manifest, backend):
    from examples.YOLO_server_api.model_registry import revision_for
    from examples.streaming_web.media_paths import build_annotated_media_path
    from examples.streaming_web.media_paths import build_clean_demand_key
    from examples.streaming_web.media_paths import build_media_path
    from examples.streaming_web.media_paths import build_overlay_demand_key

    client = redis.Redis(host='127.0.0.1', port=26380)
    legacy, streams, expected = [], [], []
    site = 'bench-' + args.output.name + '-' + root.name
    modern = backend in (
        'rust', 'rust-sync', 'rust-four',
        'rust-async', 'rust-yuv',
    )
    host_only = backend == 'rust-yuv'
    previous = backend == 'rust-before' and args.before_yuv_pipeline
    modern = modern or previous
    client_overlay = modern and not host_only and args.rust_client_overlay
    on_demand = modern and not host_only and args.rust_on_demand
    passthrough = modern and not host_only and args.rust_clean_passthrough
    for i in range(1, cameras + 1):
        name = f'camera-{i}'
        base = build_media_path(site, name)
        overlay = build_annotated_media_path(base, args.language)
        for key in ([
            build_clean_demand_key(base),
            build_overlay_demand_key(base, args.language),
        ] if i <= args.viewers else []):
            client.set(key, '1', ex=600)
        if i <= args.viewers:
            expected.extend([base] if client_overlay else [base, overlay])
        source = (
            RTSP + '/source' if i <= cameras - args.unavailable_cameras
            else 'rtsp://127.0.0.1:28555/unavailable'
        )
        mixed = args.mixed_source_url and i % 2 == 0
        if mixed:
            source = args.mixed_source_url
        legacy.append({
            'stream_id': i, 'site_id': 1, 'site': site, 'stream_name': name,
            'video_url': source, 'model_key': 'yolo26n',
            'model_revision': revision_for(definition),
            'recognition_enabled': True, 'updated_at': 'fixture',
            'work_start_hour': 0, 'work_end_hour': 24,
            'detection_items': RULES,
        })
        streams.append({
            'id': name, 'site_id': 1, 'site': site, 'name': name,
            'source': source, 'model': 'yolo26n',
            'width': args.mixed_width if mixed else args.width,
            'height': args.mixed_height if mixed else args.height,
            'fps': args.fps,
            'languages': [args.language], 'rules': RULES,
            'work_start_hour': 0, 'work_end_hour': 24,
            'output': {
                'clean': RTSP + '/' + base,
                'annotated': RTSP + '/' + overlay,
            } if i <= args.viewers or on_demand else {},
        })
        if modern and streams[-1]['output']:
            streams[-1]['output'].update({
                'clean_passthrough': passthrough,
                'client_overlay': client_overlay,
                'on_demand': on_demand,
            })
    client.close()
    engine = manifest['models']['yolo26n']
    standalone = {
        'state_dir': str(root / 'state'), 'lossless': False,
        'record_detections': True, 'batch_size': 8,
        'models': [{
            'key': 'yolo26n', 'artifact': engine['artifact'],
            'sha256': engine['sha256'], 'version': engine['version'],
            'source_artifact': definition['artifact'],
            'source_sha256': definition['sha256'],
            'catalog_revision': revision_for(definition),
            'class_metadata': definition['classes'],
        }], 'streams': streams,
        'redis': {'url_env': 'CHD_BENCH_REDIS_URL'},
        'delivery': {
            'enabled': True, 'upload_url': MOCK + '/upload',
            'notification_url': MOCK + '/send_fcm_notification',
            'token_url': MOCK + '/token',
            'client_id_env': 'WORKER_OIDC_CLIENT_ID',
            'client_secret_env': 'WORKER_OIDC_CLIENT_SECRET',
        },
    }
    if backend == 'rust-before' and args.before_preprocess_workers is not None:
        standalone['preprocess_workers'] = args.before_preprocess_workers
    if backend == 'rust-before' and args.before_asynchronous:
        standalone['gpu'] = {'asynchronous': True, 'pipeline_depth': 1}
    if backend == 'rust-before' and args.before_capture_fps:
        standalone['capture'] = {'max_convert_fps': args.before_capture_fps}
    if modern:
        capture = {}
        if args.rust_capture_fps:
            capture['max_convert_fps'] = args.rust_capture_fps
        if args.rust_nvdec:
            capture['nvdec_device'] = 0
        if args.rust_rtsp:
            capture['rust_rtsp'] = True
        if args.rust_host_yuv or host_only:
            capture['host_yuv'] = True
        if capture:
            standalone['capture'] = capture
        standalone['exact_cpu_resize'] = (
            args.rust_cpu_exact and backend != 'rust-async'
        )
        standalone['preprocess_workers'] = (
            4 if backend == 'rust-four' else args.rust_preprocess_workers
        )
        standalone['gpu'] = {
            'asynchronous': (
                args.rust_asynchronous if backend in (
                    'rust', 'rust-async', 'rust-yuv', 'rust-before',
                ) else False
            ),
            'pipeline_depth': (
                args.rust_pipeline_depth if backend in (
                    'rust', 'rust-async', 'rust-yuv', 'rust-before',
                ) else 1
            ),
        }
    if backend == 'rust':
        standalone['metadata_async'] = args.rust_metadata_async
        standalone['gpu']['direct_pinned'] = args.rust_direct_pinned
        standalone['gpu']['exact_resize'] = args.rust_exact_resize
    write(root / 'legacy.json', legacy)
    write(root / 'rust.json', standalone)
    return expected


def environment(args, root, backend):
    env = dict(os.environ)
    env.update({
        'PYTHON_DOTENV_DISABLED': '1', 'RUST_LOG': 'warn',
        'CHD_BENCH_LIMIT_THREADS': '1',
        'DATABASE_URL': DB.replace('postgresql://', 'postgresql+asyncpg://'),
        'REDIS_HOST': '127.0.0.1', 'REDIS_PORT': '26380', 'REDIS_PASSWORD': '',
        'CHD_BENCH_REDIS_URL': 'redis://127.0.0.1:26380/0',
        'WORKER_OIDC_TOKEN_ENDPOINT': MOCK + '/token',
        'WORKER_OIDC_CLIENT_ID': 'benchmark',
        'WORKER_OIDC_CLIENT_SECRET': 'isolated-benchmark-only',
        'VIOLATION_RECORD_API_URL': MOCK, 'FCM_API_URL': MOCK,
        'VIOLATION_OUTBOX_DIRECTORY': str(root / 'outbox'),
        'VIOLATION_OUTBOX_DISCOVERY_SECONDS': '1',
        'VIOLATION_OUTBOX_POLL_SECONDS': '0.1',
        'CHD_BENCH_TRACE_DIR': str(root / 'state'),
        'YOLO_WORKER_BACKEND': 'python' if backend == 'python' else 'rust',
        'POSTPROCESS_BACKEND': 'python' if backend == 'python' else 'rust',
        'YOLO_WORKER_CAMERAS_PER_ENGINE': '32',
        'YOLO_WORKER_BATCH_SIZE': '8', 'YOLO_WORKER_BATCH_WAIT_MS': '2',
        'YOLO_WORKER_NATIVE_ASYNC': 'false',
        'YOLO_WORKER_PRECISION': 'int8',
        'YOLO_WORKER_ARTIFACT_MANIFEST': str(args.output / 'manifest.json'),
        'YOLO_WORKER_METRICS_INTERVAL_SECONDS': '5',
        'DETECT_SERVER_MODEL_KEYS': 'yolo26n',
        'STREAM_CAPTURE_BACKEND': 'opencv',
        'OPENCV_FFMPEG_CAPTURE_OPTIONS': 'rtsp_transport;tcp|threads;1',
        'MEDIA_PUBLISH_RTSP_BASE_URL': RTSP,
        'MEDIA_PUBLISH_FPS': str(args.fps),
        'MEDIA_PUBLISH_CLEAN_FPS': str(args.fps),
        'MEDIA_PUBLISH_CLEAN_SOURCE_RESTREAM': 'false',
        'MEDIA_PUBLISH_ENCODER': 'libx264',
        'MEDIA_PUBLISH_CLEAN_ENCODER': 'libx264',
        'MEDIA_PUBLISH_PRESET': 'ultrafast', 'MEDIA_PUBLISH_CRF': '23',
        'MEDIA_PUBLISH_STARTUP_WIDTH': str(args.width),
        'MEDIA_PUBLISH_STARTUP_HEIGHT': str(args.height),
        'MEDIA_OVERLAY_ALLOWED_LANGUAGES': args.language,
        'MEDIA_DEFAULT_OVERLAY_LANGUAGE': args.language,
        'STREAMING_OVERLAY_MIN_CONFIDENCE': '0.25',
        'MEDIA_PUBLISH_OVERLAY_MIN_CONFIDENCE': '0.25',
        'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1',
        'GPU_RESOURCE_DIRECTORY': str(root / 'leases'),
        'GPU_RESOURCE_INFERENCE_SLOTS': '0',
    })
    env['PATH'] = str(args.output / 'tools') + ':' + env['PATH']
    return env


def trace_rows(root):
    result = {}
    for path in (root / 'state').glob('detections-*.jsonl*'):
        rows = []
        for line in path.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        result.setdefault(path.name.split('.jsonl')[0], []).extend(rows)
    return result


COMPANIONS: dict[str, int] = {}


def companion_cpu():
    result = {}
    for name, pid in COMPANIONS.items():
        try:
            parent = psutil.Process(pid)
            result[name] = sum(
                p.cpu_times().user + p.cpu_times().system
                for p in [parent, *parent.children(recursive=True)]
            )
        except psutil.Error:
            pass
    return result


def sample(pid, gpu):
    parent = psutil.Process(pid)
    processes = [parent, *parent.children(recursive=True)]
    cpu, pss, uss, threads, swapped = {}, 0, 0, 0, 0
    for process in processes:
        try:
            times = process.cpu_times()
            cpu[process.pid] = times.user + times.system
            memory = process.memory_full_info()
            pss += memory.pss
            uss += memory.uss
            swapped += memory.swap
            threads += process.num_threads()
        except psutil.Error:
            continue
    gpu_processes = {}
    for process in [
        *pynvml.nvmlDeviceGetComputeRunningProcesses(gpu),
        *pynvml.nvmlDeviceGetGraphicsRunningProcesses(gpu),
    ]:
        if process.pid in cpu:
            gpu_processes[process.pid] = process.usedGpuMemory
    utilisation = pynvml.nvmlDeviceGetUtilizationRates(gpu).gpu
    return {
        'time': time.time(), 'cpu': cpu, 'pss': pss, 'uss': uss,
        'companion_cpu': companion_cpu(),
        'processes': len(cpu), 'threads': threads,
        'gpu_memory': sum(gpu_processes.values()),
        'gpu_utilisation_global': utilisation, 'swap': swapped,
    }


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass


def trial(args, backend, cameras, index, definition, manifest, gpu):
    for name in ('chd-e2e-redis', 'chd-e2e-pg', 'chd-e2e-rtsp'):
        inspected = subprocess.run(
            ['docker', 'inspect', '-f', '{{.State.Pid}}', name],
            capture_output=True, text=True, check=True,
        )
        COMPANIONS[name] = int(inspected.stdout.strip())

    root = args.output / f'r{index}-{cameras}-{backend}'
    root.mkdir(parents=True, exist_ok=False)
    expected = configurations(
        args, root, cameras, definition, manifest, backend,
    )
    env = environment(args, root, backend)
    command = ([
        str(args.before_binary if backend == 'rust-before' else args.binary),
        'run',
        '--config',
        str(root / 'rust.json'),
    ] if backend in (
        'rust', 'rust-before', 'rust-sync', 'rust-four',
        'rust-async', 'rust-yuv',
    ) else [
        sys.executable,
        '-m',
        'experiments.full_pipeline_benchmark.legacy',
        '--config',
        str(root / 'legacy.json'),
    ])
    with (root / 'stderr.log').open('w') as stderr, (
            root / 'stdout.log'
    ).open('w') as stdout:
        started = time.monotonic()
        process = subprocess.Popen(
            command, cwd=ROOT, env=env, stdout=stdout,
            stderr=stderr, start_new_session=True,
        )
        samples = []
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f'{root.name} exited; see stderr.log')
                current = paths()
                ready = all(
                    current.get(name, {}).get('ready', False)
                    for name in expected
                )
                traces = trace_rows(root)
                active = sum(bool(rows) for rows in traces.values())
                if ready and active == cameras - args.unavailable_cameras:
                    for name in expected:
                        properties = current[name]['tracks2'][0]['codecProps']
                        specs = json.loads((root / 'rust.json').read_text())
                        dimensions = {
                            urlsplit(target).path.lstrip('/'): (
                                spec['width'], spec['height'],
                            )
                            for spec in specs['streams']
                            for target in spec['output'].values()
                            if isinstance(target, str)
                        }
                        assert (properties['width'], properties['height']) == (
                            dimensions[name]
                        ), name
                    break
                if time.monotonic() - started > 150:
                    raise TimeoutError(f'{root.name}: outputs not ready')
                time.sleep(.5)
            startup = time.monotonic() - started
            time.sleep(args.warmup)
            beginning_paths = paths()
            first = sample(process.pid, gpu)
            samples.append(first)
            until = time.monotonic() + args.seconds
            while time.monotonic() < until:
                time.sleep(.5)
                samples.append(sample(process.pid, gpu))
                if process.poll() is not None:
                    raise RuntimeError(f'{root.name} died during measurement')
            last = samples[-1]
            ending_paths = paths()
            if not all(ending_paths.get(n, {}).get('ready') for n in expected):
                raise RuntimeError(
                    'A publisher disappeared during measurement',
                )
            time.sleep(2)
        finally:
            stop(process)
    write(root / 'samples.json', samples)
    rows = trace_rows(root)
    duration = last['time'] - first['time']
    selected = {
        camera: [
            row for row in values if (
                first['time'] *
                1000 <= row['completed_timestamp_ms'] < last['time'] * 1000
            )
        ] for camera, values in rows.items()
    }
    ages = [
        row['frame_age_ms']
        for values in selected.values() for row in values
    ]
    count = len(ages)
    if not count:
        raise RuntimeError(
            f'No processed frames in {
                first["time"]
            }..{
                last["time"]
            }; inspect trace and samples', )
    cpu_seconds = sum(
        max(0, value - first['cpu'].get(pid, value))
        for pid, value in last['cpu'].items()
    )
    result = {
        'backend': backend, 'cameras': cameras, 'round': index,
        'unavailable_cameras': args.unavailable_cameras,
        'startup_all_outputs_seconds': startup, 'window_seconds': duration,
        'processed_frames': count, 'processed_fps': count / duration,
        'per_camera_fps': {
            key: len(value) / duration
            for key, value in selected.items()
        },
        'frame_age_p50_ms': float(np.percentile(ages, 50)),
        'frame_age_p95_ms': float(np.percentile(ages, 95)),
        'frame_age_p99_ms': float(np.percentile(ages, 99)),
        'cpu_seconds': cpu_seconds,
        'companion_cpu_cores': {
            name: max(
                0, value - first['companion_cpu'].get(name, value),
            ) / duration
            for name, value in last['companion_cpu'].items()
        },
        'cpu_percent_one_core': cpu_seconds / duration * 100,
        'cpu_ms_per_processed_frame': cpu_seconds / count * 1000,
        'pss_mean_bytes': statistics.mean(s['pss'] for s in samples),
        'pss_peak_bytes': max(s['pss'] for s in samples),
        'uss_mean_bytes': statistics.mean(s['uss'] for s in samples),
        'gpu_memory_mean_bytes': statistics.mean(
            s['gpu_memory'] for s in samples
        ),
        'process_count': statistics.median(s['processes'] for s in samples),
        'thread_count': statistics.median(s['threads'] for s in samples),
        'gpu_utilisation_global_mean': statistics.mean(
            s['gpu_utilisation_global'] for s in samples
        ),
        'pid_set_stable': set(first['cpu']) == set(last['cpu']),
        'outputs_ready': len(expected),
        'viewed_cameras': min(cameras, args.viewers),
        'swap_peak_bytes': max(s['swap'] for s in samples),
        'output_received_bytes': {
            name: ending_paths[name]['bytesReceived']
            - beginning_paths[name]['bytesReceived'] for name in expected
        },
        'source_lag_p95_ms': float(
            np.percentile(
                [
                    row['source_lag_ms'] for values in selected.values()
                    for row in values if 'source_lag_ms' in row
                ], 95,
            ),
        ) if any(
            'source_lag_ms' in row for values in selected.values()
            for row in values
        ) else None,
        'trace_warning_frames': sum(
            bool(row['warnings']) for values in selected.values()
            for row in values
        ),
    }
    write(root / 'samples.json', samples)
    write(root / 'result.json', result)
    print(json.dumps(result), flush=True)
    return result


def setup_tools(args):
    tools = args.output / 'tools'
    tools.mkdir(exist_ok=True)
    wrapper = tools / 'ffmpeg'
    wrapper.write_text(
        '''#!/bin/bash
set -euo pipefail
args=("$@")
if [[ ${#args[@]} -lt 5 ]]; then exec /usr/bin/ffmpeg "$@"; fi
last=$((${#args[@]}-1))
exec /usr/bin/ffmpeg -filter_threads 1 "${args[@]:0:last}" \\
  -threads 1 -g __GOP__ -keyint_min __GOP__ -sc_threshold 0 \\
  -r __FPS__ -fps_mode cfr -pkt_size 1200 "${args[last]}"
'''.replace('__GOP__', str(args.fps * 2)).replace('__FPS__', str(args.fps)),
    )
    wrapper.chmod(0o755)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--binary', type=Path, default=ROOT /
        'native/pipeline/target/release/construction-pipeline',
    )
    parser.add_argument(
        '--before-binary', type=Path,
        help='Preserved Rust binary for alternating rearchitecture trials',
    )
    parser.add_argument(
        '--engine', type=Path, default=ROOT /
        'models/int8_engine/best_yolo26n.engine',
    )
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cameras', type=int, nargs='+', default=[1, 32])
    parser.add_argument(
        '--backends', nargs='+',
        default=['python', 'hybrid', 'rust'],
    )
    parser.add_argument('--rust-preprocess-workers', type=int, default=1)
    parser.add_argument('--rust-capture-fps', type=int, default=0)
    parser.add_argument('--before-preprocess-workers', type=int)
    parser.add_argument('--before-asynchronous', action='store_true')
    parser.add_argument('--before-capture-fps', type=int, default=0)
    parser.add_argument('--rust-nvdec', action='store_true')
    parser.add_argument('--rust-rtsp', action='store_true')
    parser.add_argument('--before-yuv-pipeline', action='store_true')
    parser.add_argument('--rust-direct-pinned', action='store_true')
    parser.add_argument('--rust-metadata-async', action='store_true')
    parser.add_argument('--rust-host-yuv', action='store_true')
    parser.add_argument('--rust-clean-passthrough', action='store_true')
    parser.add_argument('--rust-client-overlay', action='store_true')
    parser.add_argument('--rust-on-demand', action='store_true')
    parser.add_argument('--rust-cpu-exact', action='store_true')
    parser.add_argument('--rust-exact-resize', action='store_true')
    parser.add_argument('--rust-asynchronous', action='store_true')
    parser.add_argument(
        '--rust-pipeline-depth', type=int,
        choices=[1, 2], default=1,
    )
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--warmup', type=float, default=10)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--mixed-source-url')
    parser.add_argument('--mixed-width', type=int, default=1280)
    parser.add_argument('--mixed-height', type=int, default=720)
    parser.add_argument('--width', type=int, default=1920)
    parser.add_argument('--height', type=int, default=1080)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--language', default='en')
    parser.add_argument('--viewers', type=int, default=32)
    parser.add_argument('--unavailable-cameras', type=int, default=0)
    args = parser.parse_args()
    if not 0 <= args.unavailable_cameras < min(args.cameras):
        parser.error('Unavailable cameras must leave at least one source')
    healthy = min(args.cameras) - args.unavailable_cameras
    if args.unavailable_cameras and args.viewers > healthy:
        parser.error('Viewed cameras must use the healthy test source')
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    args.binary = args.binary.resolve()
    if args.before_binary is not None:
        args.before_binary = args.before_binary.resolve()
    if 'rust-before' in args.backends and args.before_binary is None:
        parser.error('rust-before requires --before-binary')
    args.engine = args.engine.resolve()
    setup_tools(args)
    definition, manifest = asyncio.run(catalog(args.engine))
    write(args.output / 'manifest.json', manifest)
    write(
        args.output / 'settings.json',
        {
            **{
                key: str(value) if isinstance(
                    value,
                    Path,
                ) else value for key,
                value in vars(args).items()
            },
            'binary_sha256': digest(args.binary),
            'driver_sha256': digest(Path(__file__)),
            'before_binary_sha256': (
                digest(args.before_binary) if args.before_binary else None
            ),
            'engine_sha256': digest(args.engine),
            'shared_contexts': 1,
            'encoder': 'libx264 ultrafast crf23 threads1',
            'clean_source_restream': False,
            'notes': (
                'Actual main.py camera processes and shared worker vs Rust; '
                '30s periodic motion source; source and services excluded '
                'from both process trees. Global GPU utilisation confounded.'
            ),
        },
    )
    pynvml.nvmlInit()
    gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
    results = []
    try:
        for cameras in args.cameras:
            for index in range(args.rounds):
                backends = args.backends[
                    index % len(
                        args.backends,
                    ):
                ] + args.backends[:index % len(args.backends)]
                for backend in backends:
                    results.append(
                        trial(
                            args, backend, cameras, index,
                            definition, manifest, gpu,
                        ),
                    )
                    write(args.output / 'results.json', results)
    finally:
        pynvml.nvmlShutdown()


if __name__ == '__main__':
    main()
