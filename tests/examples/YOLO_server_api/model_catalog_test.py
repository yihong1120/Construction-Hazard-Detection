from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.testclient import TestClient

from examples.YOLO_server_api import model_registry as catalog
from examples.YOLO_server_api import routers


@pytest.fixture
def client(tmp_path):
    artifact = tmp_path / 'model.pt'
    artifact.write_bytes(b'weights')
    model = dict(
        id='custom-detector',
        display_name='Custom',
        version='a' * 40,
        hf_repo='owner/models',
        hf_filename='weights.pt',
        hf_revision='a' * 40,
        artifact=str(artifact),
        sha256=hashlib.sha256(b'weights').hexdigest(),
        capabilities=['image', 'stream'],
        tenant_ids=['t1'],
        site_ids=[20],
        classes=[{'id': 7, 'code': 'vehicle', 'display_name': '車輛'}],
    )
    rows = [
        {'model_key': model['id'], 'definition': model, 'is_default': True},
    ]
    app = FastAPI()
    app.include_router(routers.detection_router)
    app.dependency_overrides[routers.jwt_access] = lambda: SimpleNamespace(
        subject={'role': 'admin', 'tenant_id': 't1'},
    )
    app.dependency_overrides[routers.rate_limiter_service] = lambda: 999
    with (
        patch.object(catalog, '_read_catalog_rows', return_value=rows),
        TestClient(app) as http,
    ):
        yield http, model, rows


def test_catalog_uses_database_without_loading_weights(client):
    http, model, _ = client
    with patch.object(routers.model_loader, 'get_registry_model') as load:
        result = http.get('/models')
    assert result.status_code == 200
    assert result.json()['models'][0]['classes'][0]['id'] == 7
    assert result.json()['models'][0]['version'] == model['hf_revision']
    load.assert_not_called()


def test_empty_and_disabled_catalog(client):
    http, model, rows = client
    model['enabled'] = False
    assert http.get('/models').json() == {
        'default_model_id': None,
        'models': [],
    }
    rows.clear()
    assert http.get('/models').status_code == 200


def test_database_outage_has_no_fallback(client):
    http, _, _ = client
    with patch.object(
        catalog,
        '_read_catalog_rows',
        side_effect=RuntimeError('database unavailable'),
    ):
        assert http.get('/models').status_code == 503


@pytest.mark.parametrize(
    'model,version', [('unknown', None), ('custom-detector', 'old')],
)
def test_unavailable_version_refused_before_inference(client, model, version):
    http, _, _ = client
    data = {'model': model}
    if version:
        data['model_version'] = version
    with patch.object(routers, 'run_detection_from_bytes') as infer:
        result = http.post(
            '/detect', data=data, files={'image': ('x.jpg', b'image')},
        )
    assert result.status_code == 409
    infer.assert_not_called()


def test_exact_version_and_original_response(client):
    http, model, _ = client
    with (
        patch.object(
            routers.model_loader, 'get_registry_model', return_value=Mock(),
        ),
        patch.object(
            routers,
            'run_detection_from_bytes',
            AsyncMock(
                return_value=(
                    [[1, 2, 3, 4, 0.9, 7]],
                    {'inference': 0, 'post': 0},
                ),
            ),
        ),
    ):
        result = http.post(
            '/detect',
            data={'model': model['id'], 'model_version': model['version']},
            files={'image': ('x.jpg', b'image')},
        )
    assert result.status_code == 200
    assert result.json() == [[1, 2, 3, 4, 0.9, 7]]


def test_missing_local_artifact_downloads_pinned_hf_revision(client, tmp_path):
    _, model, _ = client
    model['artifact'] = str(tmp_path / 'missing.pt')
    cached = tmp_path / 'cache.pt'
    cached.write_bytes(b'weights')
    with patch(
        'huggingface_hub.hf_hub_download', return_value=str(cached),
    ) as download:
        entry = catalog.require_model(
            model['id'], model['version'], 't1', 'admin', 'image',
        )
    download.assert_called_once_with(
        repo_id='owner/models', filename='weights.pt', revision='a' * 40,
    )
    assert entry.id == model['id']


def test_wrong_bytes_and_tenant_refused(client):
    _, model, _ = client
    with pytest.raises(HTTPException) as raised:
        catalog.require_model(model['id'], None, 't2', 'admin', 'image')
    assert raised.value.status_code == 403
    Path(model['artifact']).write_bytes(b'replaced')
    with pytest.raises(HTTPException) as raised:
        catalog.require_model(model['id'], None, 't1', 'admin', 'image')
    assert raised.value.status_code == 409
