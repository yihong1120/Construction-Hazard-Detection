"""Summarise matched end-to-end measurements; keep adverse results visible."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

matplotlib.use('Agg')

BACKENDS = ['python', 'hybrid', 'rust']
METRICS = [
    'processed_fps', 'cpu_percent_one_core', 'cpu_ms_per_processed_frame',
    'pss_mean_bytes', 'gpu_memory_mean_bytes', 'frame_age_p50_ms',
    'frame_age_p95_ms', 'startup_all_outputs_seconds', 'process_count',
    'thread_count',
]


def summary(rows):
    result = {}
    for backend in BACKENDS:
        selected = [row for row in rows if row['backend'] == backend]
        result[backend] = {
            'rounds': len(selected),
            'metrics': {
                key: {
                    'median': statistics.median(row[key] for row in selected),
                    'min': min(row[key] for row in selected),
                    'max': max(row[key] for row in selected),
                } for key in METRICS
            },
            'all_pid_sets_stable': all(
                row['pid_set_stable'] for row in selected
            ),
            'max_swap_bytes': max(
                row.get('swap_peak_bytes', 0)
                for row in selected
            ),
        }
    for baseline in ['python', 'hybrid']:
        left = result[baseline]['metrics']
        right = result['rust']['metrics']
        throughput = left['processed_fps']['median']
        comparison = {
            'throughput_ratio': (
                right['processed_fps']['median'] / throughput
                if throughput else None
            ),
        }
        for name, metric in (
            ('cpu_total', 'cpu_percent_one_core'),
            ('cpu_per_decision', 'cpu_ms_per_processed_frame'),
            ('pss', 'pss_mean_bytes'),
            ('vram', 'gpu_memory_mean_bytes'),
            ('p95', 'frame_age_p95_ms'),
        ):
            original = left[metric]['median']
            comparison[name + '_change_percent'] = (
                (right[metric]['median'] / original - 1) * 100
                if original else None
            )
        result['rust_vs_' + baseline] = comparison
    return result


def capture_rates(directory):
    result = []
    for path in sorted(directory.glob('*/samples.json')):
        samples = json.loads(path.read_text())
        if len(samples) < 2:
            continue
        low, high = samples[0]['time'] * 1000, samples[-1]['time'] * 1000
        rates = []
        for trace in (path.parent / 'state').glob('detections-*.jsonl'):
            rows = [
                json.loads(line)
                for line in trace.read_text().splitlines()
            ]
            selected = [
                row for row in rows
                if low <= row['completed_timestamp_ms'] < high
            ]
            if len(selected) < 2:
                continue
            first, last = selected[0], selected[-1]
            span_ms = last['timestamp_ms'] - first['timestamp_ms']
            frames = last['sequence'] - first['sequence']
            if span_ms > 0 and frames > 0:
                rates.append(frames * 1000 / span_ms)
        if not rates:
            continue
        result.append({
            'trial': path.parent.name,
            'decoded_fps_per_camera_median': statistics.median(rates),
            'decoded_fps_per_camera_min': min(rates),
            'decoded_fps_per_camera_max': max(rates),
        })
    return result


def figure(summary_data, settings, cameras, output):
    panels = [
        ('processed_fps', 'Completed detection decisions / second', 1),
        ('cpu_percent_one_core', 'CPU core-equivalents', 100),
        ('pss_mean_bytes', 'System RAM PSS (GiB)', 2**30),
        ('gpu_memory_mean_bytes', 'Process GPU memory (GiB)', 2**30),
        ('frame_age_p95_ms', 'Capture-return to hazard result p95 (ms)', 1),
        ('cpu_ms_per_processed_frame', 'Whole-pipeline CPU ms / decision', 1),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    labels = ['Python', 'Hybrid', 'Rust']
    for axis, (metric, label, divisor) in zip(axes.flat, panels, strict=True):
        values = [summary_data[b]['metrics'][metric] for b in BACKENDS]
        medians = [value['median'] / divisor for value in values]
        errors = [
            [(value['median'] - value['min']) / divisor for value in values],
            [(value['max'] - value['median']) / divisor for value in values],
        ]
        bars = axis.bar(
            labels, medians, yerr=errors, capsize=4,
            color=['#5079a5', '#d49a3b', '#7865a8'],
        )
        axis.bar_label(
            bars,
            labels=[f'{value:.2f}' for value in medians],
            padding=5,
        )
        axis.set_title(label, fontsize=10)
        axis.grid(axis='y', alpha=.2)
        axis.set_axisbelow(True)
        axis.set_ylim(
            0, max(
                value['max']
                for value in values
            ) / divisor * 1.3 or 1,
        )
    rounds = summary_data['rust']['rounds']
    healthy = cameras - settings.get('unavailable_cameras', 0)
    viewers = min(cameras, settings['viewers'])
    fig.suptitle(
        f'{healthy}/{cameras} active cameras, {viewers} viewed, '
        f'{settings["fps"]} FPS output setting\n'
        f'Median of {rounds} rounds with observed min and max',
        fontsize=13,
    )
    fig.text(
        .5, .012, 'All process-tree CPU/RAM included. '
        'GPU utilisation is not attributed. '
        'Detection latency excludes camera-to-decoder and browser latency.',
        ha='center', fontsize=9,
    )
    fig.tight_layout(rect=(0, .04, 1, .93))
    fig.savefig(output / 'comparison.png', dpi=160)
    fig.savefig(output / 'comparison.svg')
    plt.close(fig)


def load_run(directory):
    settings = json.loads((directory / 'settings.json').read_text())
    rows = json.loads((directory / 'results.json').read_text())
    cameras = {row['cameras'] for row in rows}
    expected = {(backend, index) for backend in BACKENDS
                for index in range(settings['rounds'])}
    actual = {(row['backend'], row['round']) for row in rows}
    if len(cameras) != 1 or actual != expected or len(rows) != len(expected):
        raise ValueError(
            f'{directory}: expected one camera count and one result per '
            'configured round for each of python, hybrid, and rust',
        )
    return settings, rows, cameras.pop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary', type=Path, required=True)
    parser.add_argument('--single', type=Path)
    parser.add_argument('--low-resolution', type=Path)
    parser.add_argument(
        '--output', type=Path, help='Report directory (default: PRIMARY/report)',
    )
    args = parser.parse_args()
    try:
        settings, rows, cameras = load_run(args.primary)
        result = {
            'settings': settings, 'primary': summary(rows), 'raw_trials': rows,
            'decoded_rate_diagnostics': capture_rates(args.primary),
            'latency_scope': 'Elapsed time from decoder return to hazard result.',
            'comparison_scope': (
                'End-to-end pipelines and backend configurations. '
                'Overload can affect completed work and decoded frame rates.'
            ),
        }
        for key, directory in (
            ('single_camera', args.single),
            ('low_resolution', args.low_resolution),
        ):
            if directory:
                extra_settings, extra_rows, _ = load_run(directory)
                result[key] = {
                    'settings': extra_settings,
                    'summary': summary(extra_rows),
                    'decoded_rate_diagnostics': capture_rates(directory),
                }
    except (OSError, ValueError, KeyError) as error:
        parser.error(str(error))

    output = args.output or args.primary / 'report'
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ) + '\n',
    )
    figure(result['primary'], settings, cameras, output)
    print(json.dumps(result['primary'], indent=2))


if __name__ == '__main__':
    main()
