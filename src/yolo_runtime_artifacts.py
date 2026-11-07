from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=128)
def _digest(path: Path, fingerprint: tuple[int, ...]) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def stream_engine_path(
    manifest_path: Path,
    model_id: str,
    version: str,
    source_sha256: str,
) -> Path:
    """Require a pinned engine derived from the current catalog version.

    The service operator supplies this manifest. Catalog authorization and
    verification of the original checkpoint still happen before this call.
    Missing or stale variants fail explicitly instead of using another model.
    """
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if manifest.get('format') != 1:
        raise ValueError('Unsupported stream engine manifest format')
    entry = manifest['models'][model_id]
    candidates = [entry, *entry.get('previous', [])]
    selected = next(
        (
            candidate for candidate in candidates if (
                candidate['version'] == version
                and candidate['source_sha256'] == source_sha256
            )
        ), None,
    )
    if selected is not None:
        entry = selected
    if (
        entry['version'] != version
        or entry['source_sha256'] != source_sha256
    ):
        raise ValueError(
            f'Stream engine source differs from catalog: {model_id}',
        )
    path = Path(entry['artifact'])
    if not path.is_absolute() or path.suffix != '.engine':
        raise ValueError('Stream engine must be an absolute .engine path')
    stat = path.stat()
    fingerprint = (
        stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
    )
    if _digest(path, fingerprint) != entry['sha256']:
        raise ValueError(f'Stream engine checksum differs: {model_id}')
    return path


def retain_engine_history(manifest: dict, previous: dict) -> dict:
    """Keep immutable previous artifacts available for verified rollback."""
    models = dict(previous.get('models', {}))
    for key, entry in manifest['models'].items():
        old = models.get(key)
        history = [old, *old.get('previous', [])] if old else []
        selected = {(entry['version'], entry['source_sha256'])}
        retained = []
        for candidate in history:
            identity = (candidate['version'], candidate['source_sha256'])
            if identity not in selected:
                retained.append(
                    {k: v for k, v in candidate.items() if k != 'previous'},
                )
                selected.add(identity)
        models[key] = {
            **entry, **
            ({'previous': retained[:2]} if retained else {}),
        }
    return {'format': 1, 'models': models}
