"""Cross-process admission, crash recovery and encoder-child lease lifetime."""
from __future__ import annotations

import multiprocessing
import os
import signal
import subprocess
import sys
import time

import pytest

from src.gpu_resources import _python_acquire
from src.gpu_resources import try_acquire_resource


def hold_resource(directory, backend, connection):
    os.environ['GPU_RESOURCE_DIRECTORY'] = directory
    os.environ['POSTPROCESS_BACKEND'] = backend
    lease = try_acquire_resource('nvenc', 2)
    connection.send(lease is not None)
    connection.recv()
    if lease:
        lease.close()
    connection.close()


def orphan_encoder(directory, connection):
    os.environ['GPU_RESOURCE_DIRECTORY'] = directory
    lease = try_acquire_resource('nvenc', 1)
    child = subprocess.Popen(
        [sys.executable, '-c', 'import time; time.sleep(30)'],
        pass_fds=tuple(lease.filenos()),
    )
    connection.send(child.pid)
    connection.close()
    os._exit(0)


def forked_attempt(connection):
    lease = try_acquire_resource('nvenc', 1)
    connection.send(lease is not None)
    connection.recv()
    lease = try_acquire_resource('nvenc', 1)
    connection.send(lease is not None)
    if lease:
        lease.close()


def contend_resource(directory, backend, connection):
    os.environ['GPU_RESOURCE_DIRECTORY'] = directory
    os.environ['POSTPROCESS_BACKEND'] = backend
    connection.send('ready')
    connection.recv()
    lease = try_acquire_resource('nvenc', 6)
    connection.send(lease is not None)
    connection.recv()
    if lease:
        lease.close()


@pytest.fixture(autouse=True)
def resource_directory(monkeypatch, tmp_path):
    monkeypatch.setenv('GPU_RESOURCE_DIRECTORY', str(tmp_path / 'budget'))
    monkeypatch.setenv('POSTPROCESS_BACKEND', 'auto')


def test_python_and_native_leases_share_one_pool(monkeypatch):
    directory = os.environ['GPU_RESOURCE_DIRECTORY']
    parent = _python_acquire(directory, 'gpu0.nvenc', 2, 1)
    context = multiprocessing.get_context('spawn')
    connection, child = context.Pipe()
    process = context.Process(
        target=hold_resource, args=(
            directory, 'auto', child,
        ),
    )
    process.start()
    try:
        assert connection.poll(10) and connection.recv()
        assert try_acquire_resource('nvenc', 2) is None
        process.kill()
        process.join(timeout=10)
        recovered = try_acquire_resource('nvenc', 2)
        assert recovered is not None
        recovered.close()
    finally:
        parent.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        connection.close()
        child.close()


@pytest.mark.parametrize('backend', ['python', 'auto'])
def test_partial_reservation_rolls_back_and_active_capacity_is_consistent(
        monkeypatch, backend,
):
    monkeypatch.setenv('POSTPROCESS_BACKEND', backend)
    first = try_acquire_resource('memory', 3, 2)
    assert first is not None
    assert try_acquire_resource('memory', 3, 2) is None
    remaining = try_acquire_resource('memory', 3, 1)
    assert remaining is not None
    with pytest.raises(RuntimeError, match='differs'):
        try_acquire_resource('memory', 4, 1)
    first.close()
    remaining.close()
    changed = try_acquire_resource('memory', 4, 4)
    assert changed is not None
    changed.close()


@pytest.mark.parametrize('backend', ['python', 'auto'])
def test_corrupt_budget_and_invalid_resource_requests_fail_closed(
        monkeypatch, backend,
):
    from pathlib import Path

    monkeypatch.setenv('POSTPROCESS_BACKEND', backend)
    directory = Path(os.environ['GPU_RESOURCE_DIRECTORY'])
    directory.mkdir()
    (directory / 'gpu0.nvenc.control').write_text('0')
    with pytest.raises(RuntimeError, match='Corrupt'):
        try_acquire_resource('nvenc', 6)
    with pytest.raises(ValueError, match='kind'):
        try_acquire_resource('../escape', 6)
    with pytest.raises(ValueError, match='capacity'):
        try_acquire_resource('nvenc', 0)


def test_encoder_child_keeps_reservation_after_controller_crash():
    context = multiprocessing.get_context('spawn')
    connection, child = context.Pipe()
    process = context.Process(
        target=orphan_encoder, args=(
            os.environ['GPU_RESOURCE_DIRECTORY'], child,
        ),
    )
    process.start()
    encoder_pid = None
    try:
        assert connection.poll(10)
        encoder_pid = connection.recv()
        process.join(timeout=10)
        assert process.exitcode == 0
        assert try_acquire_resource('nvenc', 1) is None
        os.kill(encoder_pid, signal.SIGTERM)
        deadline = time.monotonic() + 5
        recovered = None
        while recovered is None and time.monotonic() < deadline:
            recovered = try_acquire_resource('nvenc', 1)
            if recovered is None:
                time.sleep(.01)
        assert recovered is not None
        recovered.close()
    finally:
        if encoder_pid is not None:
            try:
                os.kill(encoder_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        connection.close()
        child.close()


def test_fork_closes_inherited_leases_without_unlocking_parent():
    context = multiprocessing.get_context('fork')
    parent_lease = try_acquire_resource('nvenc', 1)
    connection, child = context.Pipe()
    process = context.Process(target=forked_attempt, args=(child,))
    process.start()
    try:
        assert connection.poll(10) and connection.recv() is False
        parent_lease.close()
        connection.send('retry')
        assert connection.poll(10) and connection.recv() is True
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        parent_lease.close()
        if process.is_alive():
            process.kill()
            process.join(timeout=10)
        connection.close()
        child.close()


def test_32_processes_share_six_slots_and_recover_after_crash():
    context = multiprocessing.get_context('fork')
    workers = []
    try:
        for index in range(32):
            connection, child = context.Pipe()
            process = context.Process(
                target=contend_resource, args=(
                    os.environ['GPU_RESOURCE_DIRECTORY'],
                    'python' if index % 2 else 'auto', child,
                ),
            )
            process.start()
            child.close()
            workers.append((process, connection))
        for _, connection in workers:
            assert connection.poll(10) and connection.recv() == 'ready'
        for _, connection in workers:
            connection.send('start')
        admitted = []
        for process, connection in workers:
            assert connection.poll(10)
            if connection.recv():
                admitted.append(process)
        assert len(admitted) == 6
        assert try_acquire_resource('nvenc', 6) is None
        admitted[0].kill()
        admitted[0].join(timeout=10)
        replacement = try_acquire_resource('nvenc', 6)
        assert replacement is not None
        assert try_acquire_resource('nvenc', 6) is None
        replacement.close()
    finally:
        for process, connection in workers:
            if process.is_alive():
                connection.send('stop')
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
            connection.close()
