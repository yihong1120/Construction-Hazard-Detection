from __future__ import annotations

import importlib.util
import json
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest


def _get_profile(
    profile,
    /,
):
    return profile


def _profile_accepts_successful_none_return_make_simplenamespace(_):
    return SimpleNamespace(name='images')


def _profile_accepts_successful_none_return_add(added, /, value):
    added.append(value)
    return 0


def _profile_rejects_unusable_configuration_add_optimization_profile(
    failure,
    /,
    _,
):
    return -1 if failure == 'add' else 0


def _return_none():
    return None


def _manager_callback():
    return {'ActiveState': 'active'}


def _failed_or_interrupted_build_resumes_append(commands, /, *args):
    return commands.append(args)


def _failed_or_interrupted_build_resumes_fail(failure, /, *args):
    raise failure


def _failed_activation_probe_restores_code_callback():
    return {'files': {}}


def _failed_activation_probe_restores_code_make_simplenamespace(
    owner,
    /,
    _,
):
    return SimpleNamespace(
        pw_uid=owner.st_uid,
        pw_gid=owner.st_gid,
    )


def _failed_activation_probe_restores_code_fail_probe(
    config,
    target,
    /,
    *args,
):
    assert target.read_bytes() == b'candidate'
    assert (
        json.loads(config.read_text())['environment']['YOLO_WORKER_PRECISION']
        == 'int8'
    )
    raise RuntimeError('engine load failed')


class _ProfileAcceptsSuccessfulNoneReturnProfile:
    def __init__(self, *, _test_setter_result):
        self._test_setter_result = _test_setter_result

    def set_shape(self, name, minimum, optimum, maximum):
        self.shapes = {name: [minimum, optimum, maximum]}
        return self._test_setter_result

    def get_shape(self, name):
        return self.shapes[name]


class _ProfileRejectsUnusableConfigurationProfile:
    def __init__(self, *, _test_failure):
        self._test_failure = _test_failure

    def set_shape(self, *args):
        return None

    def get_shape(self, name):
        return [(1, 3, 320 if self._test_failure == 'shape' else 640, 640)] * 3

    def __bool__(self):
        return self._test_failure != 'invalid'


@pytest.fixture
def builder_script():
    script = (
        Path(__file__).resolve().parents[2]
        / 'deploy/yolo-stream-engines/build.py'
    )
    spec = importlib.util.spec_from_file_location('engine_builder', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('setter_result', [None, True])
def test_profile_accepts_successful_none_return(builder_script, setter_result):
    """TRT 11 mutates the profile successfully without returning a bool."""
    Profile = partial(
        _ProfileAcceptsSuccessfulNoneReturnProfile,
        _test_setter_result=setter_result,
    )

    profile = Profile()
    builder = SimpleNamespace(
        create_optimization_profile=partial(
            _get_profile,
            profile,
        ),
    )
    network = SimpleNamespace(
        num_inputs=1,
        get_input=_profile_accepts_successful_none_return_make_simplenamespace,
    )
    added = []

    add = partial(_profile_accepts_successful_none_return_add, added)

    config = SimpleNamespace(add_optimization_profile=add)
    assert builder_script.add_batch_one_profile(builder, network, config) == 0
    assert profile.shapes['images'] == [(1, 3, 640, 640)] * 3
    assert added == [profile]


@pytest.mark.parametrize('failure', ['shape', 'invalid', 'add'])
def test_profile_rejects_unusable_configuration(builder_script, failure):
    Profile = partial(
        _ProfileRejectsUnusableConfigurationProfile,
        _test_failure=failure,
    )

    builder = SimpleNamespace(create_optimization_profile=Profile)
    network = SimpleNamespace(
        num_inputs=1,
        get_input=_profile_accepts_successful_none_return_make_simplenamespace,
    )
    config = SimpleNamespace(
        add_optimization_profile=partial(
            _profile_rejects_unusable_configuration_add_optimization_profile,
            failure,
        ),
    )
    with pytest.raises(RuntimeError, match='batch-1 profile'):
        builder_script.add_batch_one_profile(builder, network, config)


def test_metadata_does_not_override_runtime_device(builder_script):
    original = {
        'device': 'RTX 3050',
        'names': {'0': 'Person'},
        'batch': 16,
        'args': {'quantize': 8, 'batch': 16},
    }
    metadata = builder_script.batch_one_metadata(original)
    assert 'device' not in metadata
    assert metadata['batch'] == 1
    assert metadata['args']['batch'] == 1
    assert metadata['args']['quantize'] == 8
    assert original['args']['batch'] == 16
    assert original['device'] == 'RTX 3050'


def test_repairs_existing_header_without_rebuilding_plan(
    builder_script,
    tmp_path,
):
    path = tmp_path / 'model.engine'
    header = json.dumps({'device': 'RTX 3050', 'batch': 1}).encode()
    plan = b'calibrated-device-specific-tensorrt-plan'
    path.write_bytes(len(header).to_bytes(4, 'little') + header + plan)
    old_sha = builder_script.digest(path)
    receipt_path = tmp_path / 'build.json'
    receipt = builder_script.repair_device_metadata(
        path,
        receipt_path,
        {'sha256': old_sha, 'smoke_passed': True},
    )
    with path.open('rb') as stream:
        size = int.from_bytes(stream.read(4), 'little')
        assert 'device' not in json.loads(stream.read(size))
        assert stream.read() == plan
    assert receipt['sha256'] == builder_script.digest(path) != old_sha
    assert 'smoke_passed' not in receipt
    assert json.loads(receipt_path.read_text()) == receipt
    assert Path(receipt['metadata_backup']).exists()
    assert (
        builder_script.repair_device_metadata(path, receipt_path, receipt)
        == receipt
    )


def test_does_not_repair_unverified_candidate(builder_script, tmp_path):
    path = tmp_path / 'model.engine'
    path.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='checksum'):
        builder_script.repair_device_metadata(
            path,
            tmp_path / 'build.json',
            {'sha256': 'a' * 64},
        )
    assert path.read_bytes() == b'changed'


@pytest.fixture
def manager(tmp_path, monkeypatch):
    script = (
        Path(__file__).resolve().parents[2]
        / 'deploy/yolo-stream-engines/manage.py'
    )
    spec = importlib.util.spec_from_file_location('engine_manager', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'websites', _return_none)
    monkeypatch.setattr(module, 'wait_ready', _manager_callback)
    return module


@pytest.mark.parametrize(
    'failure',
    [RuntimeError('build failed'), KeyboardInterrupt()],
)
def test_failed_or_interrupted_build_resumes_camera_service(
    manager,
    monkeypatch,
    failure,
):
    commands = []
    monkeypatch.setattr(manager, 'check', _return_none)
    monkeypatch.setattr(
        manager,
        'run',
        partial(_failed_or_interrupted_build_resumes_append, commands),
    )

    fail = partial(_failed_or_interrupted_build_resumes_fail, failure)

    monkeypatch.setattr(manager, 'runtime', fail)
    with pytest.raises(type(failure)):
        manager.build()
    assert commands == [
        ('systemctl', 'stop', manager.UNIT),
        ('systemctl', 'start', manager.UNIT),
    ]


def test_failed_activation_probe_restores_code_config_and_absent_helper(
    manager,
    monkeypatch,
):
    root = manager.ROOT
    app = root / 'app'
    (app / 'src').mkdir(parents=True)
    (root / 'src').mkdir()
    config = root / 'config.json'
    config.write_text(json.dumps({'environment': {'UNCHANGED': 'keep'}}))
    original_config = config.read_bytes()
    target = app / 'src/yolo_worker.py'
    target.write_bytes(b'original')
    (root / 'src/yolo_worker.py').write_bytes(b'candidate')
    (root / 'src/yolo_runtime_artifacts.py').write_bytes(b'helper')
    engine = root / 'engine.engine'
    engine.write_bytes(b'engine')
    pin = {'source_sha256': 'a' * 64, 'version': 'v1'}
    manifest = {
        'models': {
            'detector': {
                'artifact': str(engine),
                'sha256': manager.digest(engine),
            },
        },
    }
    (root / 'engines.json').write_text(json.dumps(manifest))
    (root / 'inputs.json').write_text(
        json.dumps({'models': {'detector': pin}}),
    )
    (root / 'detector-build.json').write_text(
        json.dumps({'smoke_passed': True, 'inputs': pin}),
    )
    monkeypatch.setattr(manager, 'APP', app)
    monkeypatch.setattr(manager, 'CONFIG', config)
    monkeypatch.setattr(
        manager,
        'check',
        _failed_activation_probe_restores_code_callback,
    )
    monkeypatch.setattr(manager, 'state', _manager_callback)
    owner = target.stat()
    monkeypatch.setattr(
        manager.pwd,
        'getpwnam',
        partial(
            _failed_activation_probe_restores_code_make_simplenamespace,
            owner,
        ),
    )
    commands = []
    monkeypatch.setattr(
        manager,
        'run',
        partial(_failed_or_interrupted_build_resumes_append, commands),
    )

    fail_probe = partial(
        _failed_activation_probe_restores_code_fail_probe,
        config,
        target,
    )

    monkeypatch.setattr(manager, 'runtime', fail_probe)
    with pytest.raises(RuntimeError, match='engine load failed'):
        manager.apply()
    assert target.read_bytes() == b'original'
    assert config.read_bytes() == original_config
    assert not (app / 'src/yolo_runtime_artifacts.py').exists()
    assert commands[-1] == ('systemctl', 'start', manager.UNIT)
    assert not (root / 'applied.json').exists()
