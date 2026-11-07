from __future__ import annotations

import hashlib
import json

import pytest

from src.yolo_runtime_artifacts import retain_engine_history
from src.yolo_runtime_artifacts import stream_engine_path


@pytest.fixture
def variant(tmp_path):
    engine = tmp_path / 'model.engine'
    engine.write_bytes(b'engine')
    entry = {
        'artifact': str(engine),
        'sha256': hashlib.sha256(engine.read_bytes()).hexdigest(),
        'source_sha256': 'a' * 64,
        'version': 'published-v1',
    }
    path = tmp_path / 'manifest.json'
    manifest = {'format': 1, 'models': {'detector': entry}}
    path.write_text(json.dumps(manifest))
    return path, engine, manifest


def select(path):
    return stream_engine_path(path, 'detector', 'published-v1', 'a' * 64)


def test_selects_verified_engine_and_detects_replacement(variant):
    path, engine, _ = variant
    assert select(path) == engine
    engine.write_bytes(b'replaced')
    with pytest.raises(ValueError, match='checksum'):
        select(path)


@pytest.mark.parametrize(
    'field,value', [
        ('version', 'old-version'), ('source_sha256', 'b' * 64),
        ('artifact', 'relative.engine'), ('artifact', '/models/model.pt'),
    ],
)
def test_rejects_wrong_source_or_artifact(variant, field, value):
    path, _, manifest = variant
    manifest['models']['detector'][field] = value
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        select(path)


def test_unknown_model_does_not_fall_back(variant):
    path, _, manifest = variant
    manifest['models'].clear()
    path.write_text(json.dumps(manifest))
    with pytest.raises(KeyError):
        select(path)


def test_unknown_format_is_rejected(variant):
    path, _, manifest = variant
    manifest['format'] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='format'):
        select(path)


def test_publication_retains_verified_previous_engine_for_rollback(variant):
    path, old_engine, previous = variant
    new_engine = old_engine.with_name('new-version.engine')
    new_engine.write_bytes(b'new engine')
    published = {
        'format': 1, 'models': {
            'detector': {
                'artifact': str(new_engine),
                'sha256': hashlib.sha256(new_engine.read_bytes()).hexdigest(),
                'source_sha256': 'b' * 64, 'version': 'published-v2',
            },
        },
    }
    combined = retain_engine_history(published, previous)
    path.write_text(json.dumps(combined))
    assert select(path) == old_engine
    assert stream_engine_path(
        path, 'detector', 'published-v2', 'b' * 64,
    ) == new_engine
    old_engine.write_bytes(b'corrupted rollback engine')
    with pytest.raises(ValueError, match='checksum'):
        select(path)


def test_history_is_bounded_and_preserves_unselected_models(variant):
    _, _, previous = variant
    previous['models']['unselected'] = dict(previous['models']['detector'])
    for index in range(2, 8):
        entry = dict(previous['models']['detector'])
        entry.pop('previous', None)
        entry['version'] = f'published-v{index}'
        previous = retain_engine_history(
            {'format': 1, 'models': {'detector': entry}}, previous,
        )
    assert len(previous['models']['detector']['previous']) == 2
    assert previous['models']['unselected']['version'] == 'published-v1'
