"""Replay input timestamps through old/new capture gates; no camera or GPU."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.frame_pacer import CaptureCadence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for source_fps in [15, 25, 30, 60]:
        for jitter in [0., .001, .003]:
            times = [0.] + [
                i / source_fps + (-1 if i % 2 else 1) * jitter
                for i in range(1, source_fps * 30)
            ]
            last = -1.
            old = 0
            for now in times:
                # Legacy capture used wall-clock datetime, quantized to us.
                now = round(now, 6)
                if now - last >= 1 / 30:
                    old += 1
                    last = now
            gate = CaptureCadence()
            new = sum(gate.admit(now, 1 / 30) for now in times)
            rows.append({
                'source_fps': source_fps, 'jitter_ms': jitter * 1000,
                'old_admitted_fps': old / 30,
                'new_admitted_fps': new / 30,
            })
    result = {
        'scope': 'Synthetic 30-second timestamp replay, target 30 FPS. '
        'Shows rate-gate admission, not end-to-end production speed.',
        'cases': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
