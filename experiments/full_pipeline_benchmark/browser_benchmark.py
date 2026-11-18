"""Real Chrome/WebRTC validation, separate from server-only capacity trials."""
from __future__ import annotations

import argparse
import base64
import json
import os
import signal
import statistics
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen

import psutil

ROOT = Path(__file__).resolve().parents[2]
PROXY_MODULE = 'experiments.full_pipeline_benchmark.browser_fixture'
AUTOPLAY = '--autoplay-policy=no-user-gesture-required'


def read(path):
    return json.loads(path.read_text())


def cdp(url, method, params):
    tool = Path(__file__).with_name('cdp.mjs')
    result = subprocess.check_output(
        [
            'node', str(tool), url, method, json.dumps(params),
        ], text=True, timeout=50,
    )
    response = json.loads(result)
    if 'error' in response:
        raise RuntimeError(response['error'])
    return response['result']


def evaluate(url, expression):
    result = cdp(
        url, 'Runtime.evaluate', {
            'expression': expression, 'awaitPromise': True,
            'returnByValue': True,
        },
    )
    if 'exceptionDetails' in result:
        raise RuntimeError(result['exceptionDetails'])
    return result['result'].get('value')


def stats(pid):
    processes = [psutil.Process(pid)]
    processes += processes[0].children(recursive=True)
    cpu, pss = {}, 0
    for process in processes:
        try:
            times = process.cpu_times()
            cpu[process.pid] = times.user + times.system
            pss += process.memory_full_info().pss
        except psutil.Error:
            pass
    return {'cpu': cpu, 'pss': pss}


def stop(process):
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        children = []
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    for child in children:
        try:
            child.terminate()
        except psutil.Error:
            pass


def main():
    from examples.streaming_web.media_paths import build_annotated_media_path
    from examples.streaming_web.media_paths import build_media_path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--before-binary', type=Path, required=True)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--modes', nargs='+', default=['before', 'after'])
    parser.add_argument('--seconds', type=float, default=20)
    parser.add_argument('--warmup', type=float, default=10)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ['CHD_BENCH_REDIS_URL'] = 'redis://127.0.0.1:26380/0'
    address = subprocess.check_output(
        [
            'docker', 'inspect', '-f',
            '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}',
            'chd-e2e-rtsp',
        ], text=True,
    ).strip()
    results = []
    for index in range(args.rounds):
        for mode in (args.modes if index % 2 == 0 else args.modes[::-1]):
            root = args.output / f'r{index}-{mode}'
            root.mkdir()
            config = read(args.config)
            config['state_dir'] = str(root / 'state')
            config['delivery']['enabled'] = False
            if mode == 'after':
                config.setdefault('capture', {})['host_yuv'] = True
            for number, stream in enumerate(config['streams']):
                stream['site'] = f'browser-{mode}-{index}'
                base = build_media_path(stream['site'], stream['name'])
                assert stream['source'].startswith('rtsp://127.0.0.1:28554/')
                stream['output'] = {
                    'clean': 'rtsp://127.0.0.1:28554/' + base,
                    'annotated': 'rtsp://127.0.0.1:28554/' +
                    build_annotated_media_path(base, 'en'),
                } if number < 4 or mode == 'after' else {}
                if mode == 'after':
                    stream['output'].update({
                        'clean_passthrough': True, 'client_overlay': True,
                        'on_demand': True,
                    })
            config_path = root / 'config.json'
            config_path.write_text(json.dumps(config, indent=2) + '\n')
            managed, handles = [], []

            def launch(command, name):
                log = (root / f'{name}.log').open('w')
                handles.append(log)
                p = subprocess.Popen(
                    command, cwd=ROOT, stdout=log, stderr=log,
                    start_new_session=True,
                )
                managed.append(p)
                return p

            with tempfile.TemporaryDirectory(prefix='chd-chrome-') as profile:
                try:
                    pipeline = launch(
                        [
                            str((
                                args.binary if mode == 'after' else
                                args.before_binary
                            ).resolve()),
                            'run', '--config', str(config_path),
                        ], 'pipeline',
                    )
                    proxy = launch(
                        [
                            os.sys.executable,
                            '-m',
                            PROXY_MODULE,
                            '--config',
                            str(config_path),
                            '--media-base',
                            f'http://{address}:8889',
                            *(['--overlay']
                              if mode == 'after' else []),
                        ],
                        'proxy',
                    )
                    chrome = launch(
                        [
                            '/opt/google/chrome/chrome',
                            '--headless=new',
                            '--no-sandbox',
                            '--no-first-run',
                            '--no-proxy-server',
                            '--disable-background-networking',
                            '--disable-extensions',
                            '--disable-breakpad',
                            '--remote-debugging-port=28779',
                            '--remote-debugging-address=127.0.0.1',
                            '--remote-allow-origins=*',
                            '--window-size=1280,900',
                            AUTOPLAY,
                            '--user-data-dir=' + profile,
                            'about:blank',
                        ],
                        'chrome',
                    )
                    deadline = time.monotonic() + 30
                    while True:
                        if any(p.poll() is not None for p in managed):
                            raise RuntimeError(
                                'A fixture exited; inspect logs',
                            )
                        try:
                            debug_api = 'http://127.0.0.1:28779/json/list'
                            with urlopen(debug_api) as r:
                                pages = json.load(r)
                                if not pages:
                                    raise OSError('Chrome page is starting')
                                target = pages[0]['webSocketDebuggerUrl']
                            with urlopen('http://127.0.0.1:28777/') as r:
                                assert r.status == 200
                            break
                        except OSError:
                            if time.monotonic() > deadline:
                                raise
                            time.sleep(.2)
                    # Wait for publishers, including demand acquisition.
                    while True:
                        media_api = 'http://127.0.0.1:29997/v3/paths/list'
                        with urlopen(media_api) as r:
                            paths = {
                                p['name']: p for p in json.load(r)[
                                    'items'
                                ]
                            }
                        wanted = [
                            s['output'][
                                'clean' if mode == 'after' else 'annotated'
                            ].rsplit('/', 1)[1] for s in config['streams'][:4]
                        ]
                        if all(paths.get(p, {}).get('ready') for p in wanted):
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError('Browser publishers not ready')
                        time.sleep(.2)
                    cdp(
                        target, 'Page.navigate', {
                            'url': 'http://127.0.0.1:28777/',
                        },
                    )
                    time.sleep(.5)
                    evaluate(
                        target, """new Promise((resolve,reject)=>{
                        const end=Date.now()+30000;const check=()=>{
                        if(window.failure)reject(Error(window.failure));
                        else if(window.ready)resolve(true);
                        else if(Date.now()>end)
                            reject(Error('Video not ready'));
                        else setTimeout(check,100);};check();})""",
                    )
                    time.sleep(args.warmup)
                    first = evaluate(target, 'window.sample()')
                    processes = {
                        'rust': pipeline.pid, 'chrome': chrome.pid,
                        'fixture_gateway': proxy.pid,
                    }
                    samples = []
                    began = time.time()
                    until = time.monotonic() + args.seconds
                    while True:
                        samples.append({
                            name: stats(pid)
                            for name, pid in processes.items()
                        })
                        if time.monotonic() >= until:
                            break
                        time.sleep(.5)
                    duration = time.time() - began
                    last = evaluate(target, 'window.sample()')
                    proof = evaluate(target, 'window.proof()')
                    (root / 'diagnostic.json').write_text(
                        json.dumps(
                            {'first': first, 'last': last, 'proof': proof},
                            indent=2,
                        ) + '\n',
                    )
                    assert len(last['videos']) == 4
                    assert all(
                        v['width'] == 1920 and v['height'] == 1080
                        for v in last['videos']
                    )
                    if mode == 'after':
                        if not any(n > 0 for n in proof):
                            evaluate(
                                target, '''new Promise((resolve,reject)=>{
                                const end=Date.now()+30000;const check=()=>{
                                const ready=Object.values(window.latest).some(
                                    v=>v.accepted&&v.snapshot.track_data.length);
                                if(ready)resolve(true);
                                else if(Date.now()>end)
                                    reject(Error('No drawable detections'));
                                else setTimeout(check,200);};check();})''',
                            )
                            time.sleep(.3)
                            proof = evaluate(target, 'window.proof()')
                        assert any(n > 0 for n in proof), proof
                        assert last['metadataFrames'] > first['metadataFrames']
                        evaluate(target, "window.setLanguage('zh-TW')")
                        time.sleep(.3)
                        screenshot = cdp(
                            target, 'Page.captureScreenshot', {
                                'format': 'png',
                                'captureBeyondViewport': False,
                            },
                        )
                        (root / 'preview.png').write_bytes(
                            base64.b64decode(
                                screenshot['data'],
                            ),
                        )
                    result = {
                        'round': index, 'mode': mode,
                        'window_seconds': duration,
                        'chrome_version': cdp(
                            target, 'Browser.getVersion', {},
                        ),
                        'client_first': first, 'client_last': last,
                        'canvas_nontransparent_pixels': proof,
                        'cpu_cores': {
                            name: sum(
                                max(
                                    0, value - samples[0][name]['cpu'].get(
                                        pid, value,
                                    ),
                                ) for pid, value in
                                samples[-1][name]['cpu'].items()
                            ) / duration for name in processes
                        },
                        'pss_mean_bytes': {
                            name: statistics.mean(
                                s[name]['pss']
                                for s in samples
                            )
                            for name in processes
                        },
                    }
                    for key in ('frames', 'dropped'):
                        result['video_' + key + '_per_second'] = sum(
                            b[key] - a[key] for a, b in
                            zip(first['videos'], last['videos'])
                        ) / duration
                    assert result['video_frames_per_second'] >= 40
                    result['rtp_bytes_per_second'] = sum(
                        sum(v.get('bytes', 0) for v in b) -
                        sum(v.get('bytes', 0) for v in a)
                        for a, b in zip(first['rtc'], last['rtc'])
                    ) / duration
                    (root / 'samples.json').write_text(json.dumps(samples))
                    destination = root / 'result.json'
                    destination.write_text(
                        json.dumps(result, indent=2) + '\n',
                    )
                    results.append(result)
                    evaluate(target, 'window.stop()')
                finally:
                    for process in reversed(managed):
                        stop(process)
                    for handle in handles:
                        handle.close()
            destination = args.output / 'results.json'
            destination.write_text(json.dumps(results, indent=2) + '\n')
            print(
                json.dumps(
                    {
                        k: result[k] for k in (
                            'mode',
                            'cpu_cores',
                            'video_frames_per_second',
                            'rtp_bytes_per_second',
                        )
                    },
                ),
                flush=True,
            )


if __name__ == '__main__':
    main()
