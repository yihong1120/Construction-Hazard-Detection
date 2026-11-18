"""Alternate implementations on one P-core and retain every measured sample."""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / 'data'
BINARY = HERE / 'target/release/chd-rust-benchmark'
CPU_ORT_VAR = 'CHD_BENCH_CPU_ORT'
GPU_ORT_VAR = 'CHD_BENCH_GPU_ORT'


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def gpu_state():
    result = subprocess.run(
        [
            'nvidia-smi',
            '--query-gpu=timestamp,utilization.gpu,memory.used,'
            'power.draw,clocks.sm',
            '--format=csv,noheader',
        ],
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def environment(backend):
    env = dict(os.environ)
    env.update(
        {
            'PYTHON_DOTENV_DISABLED': '1',
            'OPENBLAS_NUM_THREADS': '1',
            'OMP_NUM_THREADS': '1',
            'MKL_NUM_THREADS': '1',
            'ORT_DYLIB_PATH': os.environ[GPU_ORT_VAR if backend == 'cuda' else CPU_ORT_VAR],
        },
    )
    if backend == 'cuda' and (gpu_site := env.get('CHD_BENCH_GPU_SITE')):
        env['PYTHONPATH'] = os.pathsep.join(
            path for path in (gpu_site, env.get('PYTHONPATH')) if path
        )
    return env


def invoke(command, backend, label, core):
    env = environment(backend)
    run = subprocess.run(
        ['taskset', '-c', str(core), *map(str, command)],
        env=env,
        capture_output=True,
        text=True,
    )
    (DATA / f'{label}.stderr.txt').write_text(run.stderr)
    if run.returncode:
        raise RuntimeError(f'{label}: {run.stderr[-4000:]}')
    result = json.loads(run.stdout)
    dump(DATA / f'{label}.json', result)
    return result


def replay_command(language, name, repetitions, trace=None):
    if language == 'python':
        command = [
            sys.executable,
            HERE / 'benchmark.py',
            'replay',
            '--case',
            name,
            '--repetitions',
            str(repetitions),
        ]
        if trace:
            command += ['--trace', trace]
    else:
        command = [
            BINARY,
            'replay',
            DATA / 'corpus.json',
            name,
            str(repetitions),
        ]
        if trace:
            command.append(trace)
    return command


def infer_command(language, backend, post, runs, trace=None):
    count = json.loads((DATA / 'manifest.json').read_text())['input_count']
    if language == 'python':
        python = sys.executable
        if backend == 'cuda':
            python = os.getenv('CHD_BENCH_GPU_PYTHON', python)
        command = [
            python,
            HERE / 'benchmark.py',
            'infer',
            '--backend',
            backend,
            '--runs',
            str(runs),
            '--warmup',
            '20',
        ]
        if post:
            command.append('--post')
        if trace:
            command += ['--trace', trace]
    else:
        command = [
            BINARY,
            'infer',
            ROOT / 'models/pt/best_yolo26n.onnx',
            DATA / 'inputs.f32',
            str(count),
            backend,
            str(runs),
            '20',
            'post' if post else 'raw',
        ]
        if trace:
            command.append(trace)
    return command


def summarize(name, measurements):
    result = {'name': name, 'rounds': len(measurements['python'])}
    for language, rows in measurements.items():
        per_round = [
            row['elapsed_seconds'] / row['frames'] * 1000 for row in rows
        ]
        result[language] = {
            'median_round_ms_per_frame': statistics.median(per_round),
            'round_ms_per_frame': per_round,
            'median_round_p95_ms': statistics.median(
                [
                    float(np.percentile(row['latencies_us'], 95)) / 1000
                    for row in rows
                ],
            ),
            'peak_rss_kib': statistics.median(
                [row['peak_rss_kib'] for row in rows],
            ),
        }
    ratios = [
        a / b
        for a, b in zip(
            result['python']['round_ms_per_frame'],
            result['rust']['round_ms_per_frame'],
        )
    ]
    result['median_paired_speedup'] = statistics.median(ratios)
    result['paired_speedup_range'] = [min(ratios), max(ratios)]
    return result


def verify(core):
    checks = {}
    corpus = json.loads((DATA / 'corpus.json').read_text())
    for case in corpus['cases']:
        name = case['name']
        traces = {}
        for language in ('python', 'rust'):
            trace = DATA / f'{name}.{language}.trace.json'
            invoke(
                replay_command(language, name, 1, trace),
                'cpu',
                f'validate.{name}.{language}',
                core,
            )
            traces[language] = json.loads(trace.read_text())
        if traces['python'] != traces['rust']:
            for index, (a, b) in enumerate(
                zip(traces['python'], traces['rust']),
            ):
                if a != b:
                    dump(
                        DATA / 'parity_failure.json',
                        {'case': name, 'frame': index, 'python': a, 'rust': b},
                    )
                    raise AssertionError(
                        f'{name}: frame {index} differs; '
                        'see parity_failure.json',
                    )
        checks[name] = {
            'frames': len(case['frames']),
            'exact_tracks_and_warnings': True,
        }
        print(f'parity {name}: {len(case["frames"])} frames exact', flush=True)
    for backend in ('cpu', 'cuda'):
        for language in ('python', 'rust'):
            trace = DATA / f'{backend}.{language}.output.f32'
            invoke(
                infer_command(language, backend, False, 2, trace),
                backend,
                f'validate.infer.{backend}.{language}',
                core,
            )
        python = np.fromfile(
            DATA / f'{backend}.python.output.f32', dtype='<f4',
        )
        rust = np.fromfile(DATA / f'{backend}.rust.output.f32', dtype='<f4')
        np.testing.assert_allclose(python, rust, rtol=1e-6, atol=1e-6)
        checks[f'inference_{backend}'] = {
            'values': len(python),
            'exact': bool(np.array_equal(python, rust)),
            'max_abs_difference': float(np.max(np.abs(python - rust))),
        }
        print(
            f'inference {backend}: {checks[f"inference_{backend}"]}',
            flush=True,
        )
    dump(DATA / 'correctness.json', checks)
    return checks


def measure(core, rounds):
    summary = {
        'manifest': json.loads((DATA / 'manifest.json').read_text()),
        'correctness': json.loads((DATA / 'correctness.json').read_text()),
        'cpu_core': core,
        'rounds': rounds,
        'results': [],
        'gpu_observations': [],
    }
    names = [
        case['name']
        for case in json.loads((DATA / 'corpus.json').read_text())['cases']
    ]
    for name in names:
        calibration = json.loads(
            (DATA / f'validate.{name}.python.json').read_text(),
        )
        reps = min(
            200, max(1, math.ceil(0.25 / calibration['elapsed_seconds'])),
        )
        measurements = {'python': [], 'rust': []}
        for round_number in range(rounds):
            order = (
                ('python', 'rust')
                if round_number % 2 == 0
                else ('rust', 'python')
            )
            for language in order:
                result = invoke(
                    replay_command(language, name, reps),
                    'cpu',
                    f'replay.{name}.{language}.{round_number}',
                    core,
                )
                measurements[language].append(result)
        row = summarize(f'replay_{name}', measurements)
        row['repetitions_per_round'] = reps
        summary['results'].append(row)
        dump(HERE / 'results.json', summary)
        print(
            f'{row["name"]}: {row["median_paired_speedup"]:.3f}x; '
            f'Python {row["python"]["median_round_ms_per_frame"]:.4f}ms '
            f'Rust {row["rust"]["median_round_ms_per_frame"]:.4f}ms',
            flush=True,
        )
    for backend in ('cpu', 'cuda'):
        for post in (False, True):
            measurements = {'python': [], 'rust': []}
            name = f'infer_{backend}_{"post" if post else "raw"}'
            for round_number in range(rounds):
                order = (
                    ('python', 'rust')
                    if round_number % 2 == 0
                    else ('rust', 'python')
                )
                for language in order:
                    before = gpu_state() if backend == 'cuda' else ''
                    result = invoke(
                        infer_command(
                            language,
                            backend,
                            post,
                            160 if backend == 'cuda' else 64,
                        ),
                        backend,
                        f'{name}.{language}.{round_number}',
                        core,
                    )
                    measurements[language].append(result)
                    if backend == 'cuda':
                        summary['gpu_observations'].append(
                            {
                                'test': name,
                                'language': language,
                                'round': round_number,
                                'before': before,
                                'after': gpu_state(),
                            },
                        )
            row = summarize(name, measurements)
            summary['results'].append(row)
            dump(HERE / 'results.json', summary)
            print(
                f'{name}: {row["median_paired_speedup"]:.3f}x; '
                f'Python {row["python"]["median_round_ms_per_frame"]:.4f}ms '
                f'Rust {row["rust"]["median_round_ms_per_frame"]:.4f}ms',
                flush=True,
            )
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--core', type=int, default=4)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--skip-verify', action='store_true')
    args = parser.parse_args()
    for name in (CPU_ORT_VAR, GPU_ORT_VAR):
        if not (location := os.getenv(name)) or not Path(location).is_file():
            parser.error(
                f'Set {name} to the corresponding libonnxruntime shared library',
            )
    if not args.skip_verify:
        verify(args.core)
    if not args.verify_only:
        measure(args.core, args.rounds)


if __name__ == '__main__':
    main()
