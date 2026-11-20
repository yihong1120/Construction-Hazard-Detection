"""Build the Rust GPU wheel against the repository's pinned TensorRT SDK."""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path

from deploy.python315.bootstrap import checkout
from deploy.python315.bootstrap import TRT_COMMIT
from deploy.python315.bootstrap import TRT_VERSION

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument(
        '--gpu-media', action='store_true',
        help='Include experimental FFmpeg CUDA/NVENC device-frame encoder',
    )
    parser.add_argument(
        '--cache', type=Path,
        default=Path('/tmp/native-inference'),
    )
    args = parser.parse_args()
    args.cache.mkdir(parents=True, exist_ok=True)
    try:
        distribution = importlib.metadata.distribution('tensorrt-cu13-libs')
    except importlib.metadata.PackageNotFoundError:
        subprocess.run(
            [
                'uv', 'pip', 'install', '--system', '--no-deps',
                '--index-url', 'https://pypi.nvidia.com',
                f'tensorrt-cu13-libs=={TRT_VERSION}',
            ], check=True,
        )
        distribution = importlib.metadata.distribution('tensorrt-cu13-libs')
    if distribution.version != TRT_VERSION:
        raise RuntimeError(
            f'Expected TensorRT {TRT_VERSION}; found {distribution.version}',
        )
    source = args.cache / 'TensorRT'
    checkout('https://github.com/NVIDIA/TensorRT.git', source, TRT_COMMIT)
    libraries = Path(str(distribution.locate_file('tensorrt_libs')))
    links = args.cache / 'lib'
    links.mkdir(exist_ok=True)
    for name in ('nvinfer', 'nvinfer_plugin'):
        link = links / f'lib{name}.so'
        target = libraries / f'lib{name}.so.11'
        if link.is_symlink():
            link.unlink()
        link.symlink_to(target)
    cuda = Path(
        os.environ.get(
            'CUDA_HOME', '/usr/local/cuda-13.0/targets/x86_64-linux',
        ),
    )
    env = os.environ.copy()
    env.update({
        'TRT_INCLUDE': str(source / 'include'), 'TRT_LIB': str(links),
        'CUDA_INCLUDE': str(cuda / 'include'), 'CUDA_LIB': str(cuda / 'lib'),
        'PYO3_PYTHON': sys.executable,
        'LIBRARY_PATH': ':'.join((
            str(libraries), str(cuda / 'lib'), str(cuda / 'lib/stubs'),
            env.get('LIBRARY_PATH', ''),
        )),
    })
    subprocess.run(
        [
            'uvx', '--from', 'maturin==1.16.0', 'maturin', 'build',
            '--release', '--locked', '--auditwheel', 'skip',
            '--manifest-path',
            str(ROOT / 'bindings/python/inference/Cargo.toml'),
            '--features', 'gpu-media' if args.gpu_media else 'tensorrt',
            '--interpreter', sys.executable,
            '--out', str(args.output),
        ], env=env, cwd=ROOT, check=True,
    )


if __name__ == '__main__':
    main()
