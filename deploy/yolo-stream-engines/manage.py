from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import pwd
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = Path('/home/changdar02/service-migration-phase7')
APP = BASE / 'app'
CONFIG = BASE / 'gpu-production.json'
UNIT = 'phase7-cameras.service'
USER = 'changdar02'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def run(*args, **kwargs):
    return subprocess.run(args, check=True, **kwargs)


def state():
    output = subprocess.check_output(
        [
            'systemctl',
            'show',
            UNIT,
            '-p',
            'MainPID',
            '-p',
            'ActiveState',
            '-p',
            'NRestarts',
        ],
        text=True,
    )
    return dict(line.split('=', 1) for line in output.splitlines())


def wait_ready():
    last, since = None, 0
    for _ in range(90):
        value = state()
        pid = int(value['MainPID'])
        children = Path(f"/proc/{pid}/task/{pid}/children")
        if (
            value['ActiveState'] == 'active'
            and children.exists()
            and len(children.read_text().split()) >= 2
        ):
            if value == last and time.monotonic() - since >= 15:
                return value
            if value != last:
                last, since = value, time.monotonic()
        time.sleep(2)
    raise RuntimeError(
        'Camera supervisor did not remain healthy with worker children',
    )


def websites():
    for host, path in [
        ('changdar-server.mooo.com', '/'),
        ('ai.changdar-server.mooo.com', '/'),
        ('wiki.changdar-server.mooo.com', '/healthz'),
    ]:
        run(
            'curl',
            '--noproxy',
            '*',
            '--fail',
            '--silent',
            '--show-error',
            '--max-time',
            '20',
            '--output',
            '/dev/null',
            'https://' + host + path,
        )


def install_bytes(data, target, owner=None):
    old = target.stat() if target.exists() else None
    fd, temporary = tempfile.mkstemp(
        prefix='.engine-update-',
        dir=target.parent,
    )
    try:
        with os.fdopen(fd, 'wb') as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
            if old:
                os.fchmod(output.fileno(), old.st_mode & 0o777)
                os.fchown(output.fileno(), old.st_uid, old.st_gid)
            else:
                os.fchmod(output.fileno(), 0o600)
                os.fchown(output.fileno(), owner.pw_uid, owner.pw_gid)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def runtime(script, *args):
    command = [str(BASE / 'gpu-runtime/bin/python'), str(ROOT / script), *args]
    if os.geteuid() == 0:
        command = ['runuser', '-u', USER, '--', *command]
    run(*command)


def check():
    value = json.loads((ROOT / 'code-manifest.json').read_text())
    for name, checksum in value['files'].items():
        if digest(ROOT / name) != checksum:
            raise RuntimeError('Bundle checksum differs: ' + name)
    for name in ('src/yolo_worker.py', 'src/yolo_runtime_artifacts.py'):
        target = APP / name
        if not target.exists():
            if name in value['before']:
                raise RuntimeError('Missing production module: ' + name)
            continue
        if digest(target) not in {
            value['files'][name],
            value['before'].get(name),
        }:
            raise RuntimeError(
                'Production source changed independently: ' + name,
            )
    config = json.loads(CONFIG.read_text())
    environment = config['environment']
    if (
        int(environment.get('YOLO_WORKER_BATCH_SIZE', '8')) != 1
        or int(environment.get('YOLO_WORKER_IMGSZ', '640')) != 640
    ):
        raise RuntimeError(
            'This engine profile requires batch=1 and imgsz=640',
        )
    if environment.get('YOLO_WORKER_ARTIFACT_MANIFEST') not in (
        None,
        '',
        str(ROOT / 'engines.json'),
    ):
        raise RuntimeError('A different stream engine manifest is configured')
    if state()['ActiveState'] != 'active':
        raise RuntimeError('Existing camera service is not active')
    websites()
    return value


def build():
    check()
    print(
        (
            'Pausing camera inference to release GPU memory; websites remain '
            'online.'
        ),
        flush=True,
    )
    run('systemctl', 'stop', UNIT)
    try:
        runtime('build.py')
    finally:
        run('systemctl', 'start', UNIT)
        wait_ready()
        websites()
        print(
            (
                'Original camera service resumed. Production model selection '
                'is unchanged.'
            ),
            flush=True,
        )


def restore(backup):
    record = json.loads((backup / 'backup.json').read_text())
    for target, saved in record.items():
        path = Path(target)
        if saved is None:
            path.unlink(missing_ok=True)
        else:
            install_bytes(
                (backup / saved).read_bytes(),
                path,
                pwd.getpwnam(USER),
            )


def apply():
    code = check()
    if (ROOT / 'applied.json').exists():
        receipt = json.loads((ROOT / 'applied.json').read_text())
        if (
            receipt['engine_manifest_sha256'] == digest(ROOT / 'engines.json')
            and receipt['configuration_sha256'] == digest(CONFIG)
            and all(
                digest(APP / name) == sha
                for name, sha in receipt['files'].items()
                if name.startswith('src/')
            )
        ):
            wait_ready()
            print(
                (
                    'This verified engine release is already active; original '
                    'backup retained.'
                ),
            )
            return
        raise RuntimeError(
            (
                'An applied release differs; inspect or roll back before '
                'replacing '
                'it'
            ),
        )
    manifest = json.loads((ROOT / 'engines.json').read_text())
    pins = json.loads((ROOT / 'inputs.json').read_text())
    if set(manifest['models']) != set(pins['models']):
        raise RuntimeError('Not all five engines have been built')
    for key, entry in manifest['models'].items():
        receipt = json.loads((ROOT / (key + '-build.json')).read_text())
        if (
            not receipt.get(
                'smoke_passed',
            )
            or receipt['inputs'] != pins['models'][key]
            or digest(Path(entry['artifact'])) != entry['sha256']
        ):
            raise RuntimeError(
                'Engine build receipt or checksum differs: ' + key,
            )
    backup = ROOT / ('production-backup-' + str(time.time_ns()))
    backup.mkdir(mode=0o700)
    targets = [
        APP / 'src/yolo_worker.py',
        APP / 'src/yolo_runtime_artifacts.py',
        CONFIG,
    ]
    record = {}
    for i, target in enumerate(targets):
        saved = str(i) + '.before' if target.exists() else None
        if saved:
            shutil.copy2(target, backup / saved)
        record[str(target)] = saved
    (backup / 'backup.json').write_text(json.dumps(record, indent=2) + '\n')
    original = state()
    run('systemctl', 'stop', UNIT)
    try:
        owner = pwd.getpwnam(USER)
        for name in ('src/yolo_worker.py', 'src/yolo_runtime_artifacts.py'):
            install_bytes((ROOT / name).read_bytes(), APP / name, owner)
        config = json.loads(CONFIG.read_text())
        config['environment'].update(
            YOLO_WORKER_PRECISION='int8',
            YOLO_WORKER_ARTIFACT_MANIFEST=str(ROOT / 'engines.json'),
        )
        install_bytes(
            (json.dumps(config, indent=2) + '\n').encode(),
            CONFIG,
            owner,
        )
        # Probe the installed worker before allowing any camera to write
        # events.
        runtime('probe.py')
        started = time.time()
        run('systemctl', 'start', UNIT)
        active = wait_ready()
        log = subprocess.check_output(
            [
                'journalctl',
                '--unit=' + UNIT,
                '--since=@' + str(started),
                '--no-pager',
                '--output=cat',
            ],
            text=True,
        )
        if any(
            error in log
            for error in (
                '[YOLO-Worker] batch request failed',
                'CUDA out of memory',
                'CUDA error:',
                'TensorRT model exported with a different version',
            )
        ):
            raise RuntimeError(
                (
                    'Camera workers reported inference errors after restart; '
                    'see private '
                    'journal'
                ),
            )
        websites()
    except BaseException:
        run('systemctl', 'stop', UNIT)
        restore(backup)
        run('systemctl', 'start', UNIT)
        wait_ready()
        print(
            (
                'Verification failed; original camera code and configuration '
                'restored.'
            ),
            flush=True,
        )
        raise
    (ROOT / 'applied.json').write_text(
        json.dumps(
            {
                'backup': str(backup),
                'before': original,
                'after': active,
                'files': code['files'],
                'engine_manifest_sha256': digest(ROOT / 'engines.json'),
                'configuration_sha256': digest(CONFIG),
                'database_modified': False,
                'image_api_changed': False,
            },
            indent=2,
        )
        + '\n',
    )
    print(
        (
            'Applied: mixed INT8/FP16 stream engines verified on B; camera '
            'workers stable.'
        ),
        flush=True,
    )
    print('Backup: ' + str(backup))


def rollback():
    receipt = json.loads((ROOT / 'applied.json').read_text())
    # Refuse to discard later source or operator configuration changes.
    for name, sha in receipt['files'].items():
        if name.startswith('src/') and digest(APP / name) != sha:
            raise RuntimeError(
                'Production source changed after deployment: ' + name,
            )
    config = json.loads(CONFIG.read_text())
    if config['environment'].get('YOLO_WORKER_ARTIFACT_MANIFEST') != str(
        ROOT / 'engines.json',
    ):
        raise RuntimeError(
            'Production engine selection changed after deployment',
        )
    if digest(CONFIG) != receipt['configuration_sha256']:
        raise RuntimeError('Production configuration changed after deployment')
    run('systemctl', 'stop', UNIT)
    try:
        restore(Path(receipt['backup']))
    finally:
        run('systemctl', 'start', UNIT)
    wait_ready()
    websites()
    (ROOT / 'applied.json').rename(
        ROOT / ('rolled-back-' + str(time.time_ns()) + '.json'),
    )
    print(
        (
            'Original camera modules and configuration restored; engine files '
            'retained.'
        ),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'action',
        choices=[
            'check',
            'build',
            'apply',
            'rollback',
        ],
    )
    action = parser.parse_args().action
    os.umask(0o077)
    if action == 'check':
        check()
        runtime('build.py', '--check')
        return
    if os.geteuid() != 0:
        raise SystemExit(
            'Run this action with sudo on B to manage the camera service.',
        )
    with (ROOT / 'manage.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        globals()[action]()


if __name__ == '__main__':
    main()
