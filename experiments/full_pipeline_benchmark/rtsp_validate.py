"""Compare finite decoded pixel sequences from both RTSP receivers."""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--url', required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    children = []
    try:
        for mode in ('libav', 'rust'):
            output = (args.output / f'{mode}.json').open('w')
            error = (args.output / f'{mode}.stderr').open('w')
            child = subprocess.Popen(
                [
                    str(args.probe.resolve()), args.url, str(args.frames),
                    str(args.width), str(args.height), mode,
                ], stdout=output, stderr=error,
            )
            children.append((child, output, error))
        for child, _, _ in children:
            if child.wait(timeout=30):
                raise RuntimeError('Receiver failed; see probe stderr')
    finally:
        for child, output, error in children:
            if child.poll() is None:
                child.kill()
                child.wait()
            output.close()
            error.close()
    data = {
        mode: json.loads((args.output / f'{mode}.json').read_text())
        for mode in ('libav', 'rust')
    }
    a, b = (data[mode]['frames'] for mode in ('libav', 'rust'))
    # Live subscriptions may start at adjacent IDRs. Compare aligned sequences,
    # requiring at least half the requested frames and every overlapping hash.
    matches = []
    for offset in range(-len(b) + 1, len(a)):
        pairs = [(i, i - offset)
                 for i in range(len(a)) if 0 <= i - offset < len(b)]
        if len(pairs) >= args.frames // 2 and all(
            a[i]['sha256'] == b[j]['sha256'] for i, j in pairs
        ):
            matches.append((len(pairs), offset))
    if not matches:
        raise AssertionError('No matching contiguous pixel sequence')
    count, offset = max(matches)
    pts_steps = {}
    for mode, frames in (('libav', a), ('rust', b)):
        pts = [f['pts'] for f in frames]
        assert all(p is not None for p in pts), mode
        steps = [y - x for x, y in zip(pts, pts[1:])]
        assert all(s > 0 for s in steps), mode
        pts_steps[mode] = {'min': min(steps), 'max': max(steps)}
    assert data['rust']['lost_packets'] == 0
    result = {
        'requested_frames': args.frames, 'matching_frames': count,
        'libav_minus_rust_offset': offset,
        'width': args.width, 'height': args.height,
        'all_overlapping_pixels_exact': True, 'pts_step_seconds': pts_steps,
        'ingress_frames': data['rust']['ingress_frames'],
        'keyframe_dropped': data['rust']['keyframe_dropped'],
        'lost_packets': data['rust']['lost_packets'],
    }
    destination = args.output / 'result.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
