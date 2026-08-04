from __future__ import annotations

import asyncio
from typing import Any

from src import gpu_rtsp_relay as relay_module


class _FakeProcess:
    """Controllable subprocess stand-in for relay lifecycle tests."""

    def __init__(self, returncode: int | None = None) -> None:
        """Create a process with an optional completed exit status."""
        self.returncode = returncode
        self.terminated = False
        self.killed = False
        self.wait_calls = 0

    def terminate(self) -> None:
        """Mark the process as terminated."""
        self.terminated = True
        self.returncode = 0

    def kill(self) -> None:
        """Mark the process as forcibly terminated."""
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int | None:
        """Return the configured exit status."""
        self.wait_calls += 1
        return self.returncode


def test_build_publish_url_hides_source_credentials(monkeypatch: Any) -> None:
    """The MediaMTX relay path never contains a camera username or password."""
    monkeypatch.setenv(
        'GPU_DECODE_RELAY_RTSP_BASE_URL',
        'rtsp://127.0.0.1:8554',
    )

    publish_url = relay_module._build_publish_url(
        'rtsp://admin:secret@camera.example/live',
    )

    assert publish_url.startswith('rtsp://127.0.0.1:8554/gpu-decode-')
    assert 'admin' not in publish_url
    assert 'secret' not in publish_url


def test_build_command_copies_rtsp_video_through_tcp(monkeypatch: Any) -> None:
    """The relay avoids CPU encoding and uses TCP on both RTSP ends."""
    monkeypatch.setenv('GPU_DECODE_RELAY_TIMEOUT_US', '1234')

    command = relay_module._build_command(
        '/bin/ffmpeg',
        'rtsp://camera.example/live',
        'rtsp://127.0.0.1:8554/gpu-decode-test',
    )

    assert command[:2] == ['/bin/ffmpeg', '-nostdin']
    assert command.count('-rtsp_transport') == 2
    assert command[command.index('-timeout') + 1] == '1234'
    assert command[command.index('-map') + 1] == '0:v:0'
    assert command[command.index('-c:v') + 1] == 'copy'
    assert command[-1] == 'rtsp://127.0.0.1:8554/gpu-decode-test'


def test_relay_starts_once_and_closes_process(monkeypatch: Any) -> None:
    """One camera owns one short-lived ffmpeg copy process."""
    process = _FakeProcess()
    calls: list[tuple[object, ...]] = []

    async def fake_create_subprocess_exec(
        *args: object,
        **_kwargs: object,
    ) -> _FakeProcess:
        """Record the command and return a live relay process."""
        calls.append(args)
        return process

    monkeypatch.setattr(relay_module, '_find_ffmpeg', lambda: '/bin/ffmpeg')
    monkeypatch.setattr(
        relay_module.asyncio,
        'create_subprocess_exec',
        fake_create_subprocess_exec,
    )

    async def run_case() -> relay_module.GpuRtspRelay:
        """Start twice then release the owned relay process."""
        relay = relay_module.GpuRtspRelay('rtsp://camera.example/live')
        await relay.start()
        await relay.start()
        assert relay.is_running
        await relay.close()
        return relay

    relay = asyncio.run(run_case())

    assert len(calls) == 1
    assert relay.is_running is False
    assert process.terminated
    assert process.wait_calls == 1


def test_input_timeout_defaults_when_invalid(monkeypatch: Any) -> None:
    """A malformed environment value retains a bounded connection timeout."""
    monkeypatch.setenv('GPU_DECODE_RELAY_TIMEOUT_US', 'not-a-number')

    assert relay_module._input_timeout_us() == 5_000_000
