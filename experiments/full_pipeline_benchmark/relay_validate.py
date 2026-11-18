"""Verify live passthrough pixels and explicit dimension-mismatch fallback."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import uuid
from pathlib import Path
from urllib.request import urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--height', type=int, required=True)
    parser.add_argument('--frames', type=int, default=64)
    parser.add_argument('--fallback', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    replay = args.output / 'replay.json'
    replay.write_text('[[]]')
    name = 'relay-validation-' + uuid.uuid4().hex
    target = 'rtsp://127.0.0.1:28554/' + name
    width = args.width // 2 if args.fallback else args.width
    height = args.height // 2 if args.fallback else args.height
    config = {
        'state_dir': str(args.output / 'state'), 'experimental_replay': True,
        'capture': {'host_yuv': True},
        'models': [{
            'key': 'fixture', 'artifact': 'unused', 'sha256': '0' * 64,
            'version': 'relay-validation', 'replay': str(replay),
        }],
        'streams': [{
            'id': 'one', 'site_id': 1, 'site': 'isolated',
            'name': 'isolated', 'source': args.source, 'model': 'fixture',
            'width': width, 'height': height, 'fps': 15,
            'output': {'clean': target, 'clean_passthrough': True},
        }],
    }
    config_path = args.output / 'config.json'
    config_path.write_text(json.dumps(config, indent=2) + '\n')
    children = []
    with (args.output / 'pipeline.stdout').open('w') as out, (
        args.output / 'pipeline.stderr'
    ).open('w') as err:
        process = subprocess.Popen(
            [
                str(args.binary.resolve()), 'run', '--config',
                str(config_path),
            ], stdout=out, stderr=err,
        )
        try:
            deadline = time.monotonic() + 30
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Pipeline exited; see pipeline.stderr')
                api = 'http://127.0.0.1:29997/v3/paths/list'
                with urlopen(api, timeout=5) as r:
                    paths = json.load(r)['items']
                if any(p['name'] == name and p['ready'] for p in paths):
                    break
                if time.monotonic() > deadline:
                    raise TimeoutError('Relay output did not become ready')
                time.sleep(.1)
            if args.fallback:
                info = subprocess.check_output(
                    [
                        'ffprobe', '-v', 'error', '-rtsp_transport', 'tcp',
                        '-select_streams', 'v:0', '-show_entries',
                        'stream=width,height',
                        '-of', 'json', target,
                    ], text=True,
                )
                stream = json.loads(info)['streams'][0]
                assert (stream['width'], stream['height']) == (width, height)
                result = {'fallback_dimensions': stream}
            else:
                sources = [('source', args.source), ('relay', target)]
                for key, source in sources:
                    output = (args.output / f'{key}.json').open('w')
                    error = (args.output / f'{key}.stderr').open('w')
                    child = subprocess.Popen(
                        [
                            str(args.probe.resolve()), source,
                            str(args.frames),
                            str(width), str(height), 'libav',
                        ], stdout=output, stderr=error,
                    )
                    children.append((child, output, error))
                for child, output, error in children:
                    assert child.wait(timeout=35) == 0
                    output.close()
                    error.close()
                a, b = (
                    json.loads(
                        (args.output / f'{key}.json').read_text(),
                    )['frames']
                    for key in ('source', 'relay')
                )
                matches = []
                for offset in range(-len(b) + 1, len(a)):
                    pairs = [(i, i - offset) for i in range(len(a))
                             if 0 <= i - offset < len(b)]
                    if all(a[i]['sha256'] == b[j]['sha256'] for i, j in pairs):
                        matches.append((len(pairs), offset))
                count, offset = max(matches, default=(0, 0))
                assert count >= args.frames // 2, (count, offset)
                result = {
                    'requested_frames': args.frames,
                    'matching_frames': count,
                    'source_minus_relay_offset': offset,
                }
        finally:
            for child, output, error in children:
                if child.poll() is None:
                    child.kill()
                    child.wait()
                output.close()
                error.close()
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    assert process.returncode == 0
    summary = json.loads((args.output / 'state/summary.json').read_text())
    camera = summary['cameras'][0]
    assert camera['relay_errors'] == 0
    if args.fallback:
        assert camera['relay_fallback'] > 0 and camera['encoded'] > 0
        assert camera['relayed_packets'] == 0
    else:
        assert camera['relayed_packets'] > 0 and camera['encoded'] == 0
    result['camera'] = camera
    destination = args.output / 'result.json'
    destination.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
