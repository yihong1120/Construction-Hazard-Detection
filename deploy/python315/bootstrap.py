from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import sysconfig
import urllib.request
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ORT_COMMIT = '402979ac1bef89dcd55add41eaa7277c407599eb'
TRT_COMMIT = '16ed0636df41e4a27d46f82d9028fc50fadc7200'
PYBIND_COMMIT = '97bf890db679505a14dfe547a5e77bb2bd05dc90'
TRT_VERSION = '11.4.0.106'
ORT_VERSION = '1.31.0'
TRT_ASSET_URL = (
    'https://files.pythonhosted.org/packages/be/43/'
    '6fd7f0d80da5df9cf45373d76d35acff69e8946dff0e63724eef50e76f14/'
    'tensorrt_cu13_bindings-11.4.0.106-cp314-none-manylinux_2_28_x86_64.whl'
)
TRT_ASSET_SHA256 = (
    '233f5f02af7940b7cf15a5b526fc16b203d93181473a58588937bf401ca1d293'
)


def run(*args: str | Path, cwd: Path = REPO) -> None:
    """Run a build stage and stop immediately when it fails."""
    command = [str(arg) for arg in args]
    print('+', ' '.join(command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def checkout(url: str, destination: Path, commit: str) -> None:
    """Retain existing sources only when they match the pinned revision."""
    if not destination.exists():
        run('git', 'init', destination)
        run('git', 'remote', 'add', 'origin', url, cwd=destination)
        run('git', 'fetch', '--depth', '1', 'origin', commit, cwd=destination)
        run('git', 'checkout', '--detach', commit, cwd=destination)
    revision = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'],
        cwd=destination,
        text=True,
    ).strip()
    if revision != commit:
        raise RuntimeError(
            f'Unexpected source revision in {destination}: {revision}',
        )


def native_bindings_wheel(cache: Path, wheels: Path, python: Path) -> None:
    """Rebuild the native extension, preserving upstream Python helpers."""
    artifact = (
        wheels
        / (
            f'tensorrt_cu13_bindings-{TRT_VERSION}'
            '-1-cp315-cp315-linux_x86_64.whl'
        )
    )
    if artifact.exists():
        return
    source = cache / 'TensorRT'
    checkout('https://github.com/NVIDIA/TensorRT.git', source, TRT_COMMIT)
    run(
        'git',
        'submodule',
        'update',
        '--init',
        '--depth',
        '1',
        'parsers/onnx',
        cwd=source,
    )
    run(
        'uv',
        'pip',
        'install',
        '--python',
        python,
        '--no-deps',
        '--index-url',
        'https://pypi.nvidia.com',
        f'tensorrt-cu13-libs=={TRT_VERSION}',
    )
    site = Path(
        subprocess.check_output(
            [
                str(python),
                '-c',
                'import sysconfig; print(sysconfig.get_path("platlib"))',
            ],
            text=True,
        ).strip(),
    )
    cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda')) / 'include'
    if not (cuda / 'cuda_runtime_api.h').is_file():
        raise RuntimeError(f'CUDA development headers not found in {cuda}')
    build = cache / 'trt-bindings-cp315-build'
    run(
        python.parent / 'cmake',
        '-S',
        REPO / 'deploy/python315',
        '-B',
        build,
        '-G',
        'Ninja',
        '-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CXX_FLAGS_RELEASE=-O1 -DNDEBUG',
        f'-DPython3_EXECUTABLE={python}',
        f'-Dpybind11_DIR={site / "pybind11/share/cmake/pybind11"}',
        f'-DTRT_SOURCE={source}',
        f'-DTRT_LIBS={site / "tensorrt_libs"}',
        f'-DCUDA_INCLUDE={cuda}',
    )
    run(python.parent / 'cmake', '--build', build, '--parallel', '2')
    asset = cache / TRT_ASSET_URL.rsplit('/', 1)[-1]
    if not asset.exists():
        # Fixed HTTPS artifact URL; the pinned SHA256 is verified below.
        with urllib.request.urlopen(  # nosec B310
            TRT_ASSET_URL, timeout=120,
        ) as response:
            asset.write_bytes(response.read())
    if hashlib.sha256(asset.read_bytes()).hexdigest() != TRT_ASSET_SHA256:
        raise RuntimeError('TensorRT Python asset checksum mismatch')
    staging = cache / 'trt-wheel-staging'
    staging.mkdir(exist_ok=True)
    with zipfile.ZipFile(asset) as archive:
        archive.extractall(staging)
    # The cp314 native binary is replaced by the extension just built for
    # cp315.
    shutil.copy2(
        build / 'tensorrt.so',
        staging / 'tensorrt_bindings/tensorrt.so',
    )
    metadata = next(staging.glob('*.dist-info/WHEEL'))
    lines = [
        line
        for line in metadata.read_text().splitlines()
        if line.strip() and not line.startswith(('Tag:', 'Root-Is-Purelib:'))
    ]
    metadata.write_text(
        '\n'.join(
            [
                *lines,
                'Root-Is-Purelib: false',
                'Tag: cp315-cp315-linux_x86_64',
                '',
            ],
        ),
    )
    run(
        python,
        '-m',
        'wheel',
        'pack',
        staging,
        '--dest-dir',
        wheels,
        '--build-number',
        '1',
    )


def onnxruntime_wheel(
    cache: Path,
    wheels: Path,
    python: Path,
    jobs: int,
) -> None:
    """Build the CPU runtime used for calibration."""
    if list(wheels.glob('onnxruntime-1.31.0-*cp315*.whl')):
        return
    source = cache / 'onnxruntime'
    checkout(
        'https://github.com/microsoft/onnxruntime.git',
        source,
        ORT_COMMIT,
    )
    run(
        'git',
        'submodule',
        'update',
        '--init',
        '--recursive',
        '--depth',
        '1',
        cwd=source,
    )
    pybind = cache / 'pybind11'
    checkout('https://github.com/pybind/pybind11.git', pybind, PYBIND_COMMIT)
    root_option = ('--allow_running_as_root',) if os.geteuid() == 0 else ()
    run(
        python,
        source / 'tools/ci_build/build.py',
        '--build_dir',
        cache / 'ort-build',
        '--config',
        'Release',
        '--update',
        '--build',
        '--build_wheel',
        '--parallel',
        str(jobs),
        '--skip_tests',
        '--skip_submodule_sync',
        *root_option,
        '--compile_no_warning_as_error',
        '--cmake_generator',
        'Ninja',
        '--cmake_path',
        python.parent / 'cmake',
        '--cmake_extra_defines',
        'CMAKE_POLICY_VERSION_MINIMUM=3.5',
        'CMAKE_IGNORE_PREFIX_PATH=/opt/anaconda3',
        'FETCHCONTENT_TRY_FIND_PACKAGE_MODE=NEVER',
        f'FETCHCONTENT_SOURCE_DIR_PYBIND11={pybind}',
        'onnxruntime_BUILD_UNIT_TESTS=OFF',
    )
    built = list(
        (cache / 'ort-build/Release/dist').glob(
            'onnxruntime-1.31.0-*cp315*.whl',
        ),
    )
    if len(built) != 1:
        raise RuntimeError(
            f'Expected one ONNX Runtime cp315 wheel, found {built}',
        )
    shutil.copy2(built[0], wheels / built[0].name)


def main() -> None:
    """Prepare native artifacts before installing runtime dependencies."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cache-dir',
        type=Path,
        default=Path.home() / '.cache/construction-python315-build',
    )
    parser.add_argument(
        '--jobs',
        type=int,
        default=min(8, os.cpu_count() or 1),
    )
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument(
        '--cpu-only',
        action='store_true',
        help=(
            'Build/install calibration and test dependencies '
            'without TensorRT or CUDA headers.'
        ),
    )
    args = parser.parse_args()
    if sys.version_info[:2] != (
        3,
        15,
    ) or sysconfig.get_config_var('Py_GIL_DISABLED'):
        parser.error('Use standard CPython 3.15 (with the GIL enabled).')
    if platform.system() != 'Linux' or platform.machine() != 'x86_64':
        parser.error('This local build recipe targets Linux x86_64.')
    if args.jobs < 1:
        parser.error('--jobs must be positive')
    os.environ.setdefault('UV_HTTP_TIMEOUT', '300')
    os.environ.setdefault('UV_CONCURRENT_DOWNLOADS', '4')
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    wheels = REPO / 'wheels/python315'
    wheels.mkdir(parents=True, exist_ok=True)
    need_ort = not list(wheels.glob(f'onnxruntime-{ORT_VERSION}-*cp315*.whl'))
    need_trt = (
        not args.cpu_only
        and not (
            wheels
            / (
                f'tensorrt_cu13_bindings-{TRT_VERSION}'
                '-1-cp315-cp315-linux_x86_64.whl'
            )
        ).exists()
    )
    if need_ort or need_trt:
        build_python = cache / 'build-env/bin/python'
        if not build_python.exists():
            run('uv', 'venv', '--python', sys.executable, cache / 'build-env')
        # CMake launches Ninja as a subprocess; do not depend on a system
        # install.
        os.environ['PATH'] = (
            str(build_python.parent) + os.pathsep + os.environ.get('PATH', '')
        )
        run(
            'uv',
            'pip',
            'install',
            '--python',
            build_python,
            'numpy==2.5.4',
            'setuptools==84.2.0',
            'wheel==0.48.0',
            'cmake==4.4.4',
            'ninja==1.13.2',
            'packaging==26.3',
            'pybind11==3.1.0',
        )
        if not args.cpu_only:
            native_bindings_wheel(cache, wheels, build_python)
        onnxruntime_wheel(cache, wheels, build_python, args.jobs)
    manifest = {
        'python': sys.version.split()[0],
        'sources': {
            'onnxruntime': ORT_COMMIT,
            'tensorrt': TRT_COMMIT,
            'pybind11': PYBIND_COMMIT,
        },
        'sha256': {
            p.name: hashlib.sha256(
                p.read_bytes(),
            ).hexdigest()
            for p in wheels.glob('*.whl')
        },
    }
    (wheels / 'manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n',
    )
    if not args.build_only:
        # NVIDIA's index serves binary library wheels directly; PyPI provides
        # download stubs for these multi-GB artifacts.
        if not args.cpu_only:
            run(
                'uv',
                'pip',
                'install',
                '--python',
                sys.executable,
                '--find-links',
                wheels,
                '--index',
                'https://pypi.nvidia.com',
                f'tensorrt=={TRT_VERSION}',
            )
        run(
            'uv',
            'pip',
            'install',
            '--python',
            sys.executable,
            '--find-links',
            wheels,
            '-r',
            REPO / 'requirements.txt',
            f'onnxruntime=={ORT_VERSION}',
        )
        run('uv', 'pip', 'check', '--python', sys.executable)


if __name__ == '__main__':
    main()
