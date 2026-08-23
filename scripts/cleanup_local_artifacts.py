from __future__ import annotations

import argparse
import time
from collections.abc import Iterable
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGETS = (
    PROJECT_ROOT / '.history',
    PROJECT_ROOT / 'logs',
    PROJECT_ROOT / 'css-data',
    PROJECT_ROOT / 'examples' / 'YOLO_train' / 'runs',
)


def _iter_expired_files(
    targets: Iterable[Path],
    cutoff: float,
) -> Iterable[Path]:
    """Yield only regular files older than the requested retention cutoff."""
    for target in targets:
        if not target.exists():
            continue
        for candidate in target.rglob('*'):
            if candidate.is_file() and candidate.stat().st_mtime < cutoff:
                yield candidate


def _format_size(size: int) -> str:
    """Return a compact human-readable byte count."""
    value = float(size)
    for unit in ('B', 'KiB', 'MiB', 'GiB', 'TiB'):
        if value < 1024 or unit == 'TiB':
            return f'{value:.1f} {unit}'
        value /= 1024
    return '0 B'


def _remove_empty_parents(path: Path, stop_at: Path) -> None:
    """Remove now-empty directories without escaping an approved target root."""
    parent = path.parent
    while parent != stop_at and parent.is_relative_to(stop_at):
        try:
            parent.rmdir()
        except OSError:
            return
        parent = parent.parent


def main() -> None:
    """Print an artifact cleanup preview; delete only after ``--apply``."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--days',
        type=int,
        default=14,
        help='retain files modified within this many days (default: 14)',
    )
    parser.add_argument(
        '--apply',
        action='store_true',
        help='permanently delete the previewed local artifacts',
    )
    args = parser.parse_args()
    if args.days < 0:
        parser.error('--days must be non-negative')

    cutoff = time.time() - (args.days * 24 * 60 * 60)
    candidates = list(_iter_expired_files(DEFAULT_TARGETS, cutoff))
    reclaimed_bytes = sum(path.stat().st_size for path in candidates)
    action = 'Deleting' if args.apply else 'Would delete'
    print(
        f'{action} {len(candidates)} files '
        f'({_format_size(reclaimed_bytes)}) older than {args.days} days.',
    )
    for path in candidates:
        print(path.relative_to(PROJECT_ROOT))

    if not args.apply:
        print('Preview only. Re-run with --apply to permanently delete files.')
        return

    for path in candidates:
        target = next(root for root in DEFAULT_TARGETS if path.is_relative_to(root))
        path.unlink()
        _remove_empty_parents(path, target)
    print(f'Deleted {_format_size(reclaimed_bytes)} of local artifacts.')


if __name__ == '__main__':
    main()
