from __future__ import annotations

from copy import deepcopy

import pytest

from deploy.dev import prepare_detection as prepare


@pytest.fixture
def model(tmp_path, monkeypatch):
    root = tmp_path / 'repo'
    state = tmp_path / 'dev'
    source = root / 'models/pt/example.pt'
    source.parent.mkdir(parents=True)
    source.write_bytes(b'weights')
    monkeypatch.setattr(prepare, 'ROOT', root)
    monkeypatch.setattr(prepare, 'STATE', state)
    return {
        'model_key': 'example',
        'definition': {
            'artifact': str(source),
            'sha256': 'a' * 64,
            'tenant_ids': ['production-tenant'],
            'site_ids': [1],
            'capabilities': ['image', 'stream'],
            'roles': ['admin', 'user'],
            'version': 'immutable-version',
            'classes': [{'id': 0, 'code': 'person'}],
            'hf_repo': 'published/model',
            'hf_filename': 'example.pt',
            'hf_revision': 'b' * 40,
        },
    }


def test_dev_catalog_retains_version_and_classes_with_only_local_grants(model):
    before = deepcopy(model)
    source, target, definition = prepare.dev_entry(model)
    assert definition['tenant_ids'] == [prepare.TENANT]
    assert definition['site_ids'] == [] and definition['capabilities'] == [
        'image',
    ]
    assert definition['classes'] == model['definition']['classes']
    assert definition['version'] == model['definition']['version']
    assert definition['sha256'] == model['definition']['sha256']
    assert definition['hf_repo'] is None
    assert target.is_relative_to(prepare.STATE / 'models/pt')
    assert target != source and model == before


@pytest.mark.parametrize(
    'field,value',
    [
        ('artifact', '/production/other.pt'),
        ('artifact', 'models/pt/../../secret.pt'),
        ('artifact', 'models/pt/model.engine'),
        ('sha256', '../bad'),
    ],
)
def test_refuses_nonlocal_or_unverified_model_paths(model, field, value):
    model['definition'][field] = value
    with pytest.raises(RuntimeError):
        prepare.dev_entry(model)


def test_refuses_model_key_path_traversal(model):
    model['model_key'] = '../production/model'
    with pytest.raises(RuntimeError):
        prepare.dev_entry(model)
