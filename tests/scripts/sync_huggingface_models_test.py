from __future__ import annotations

import hashlib
from types import SimpleNamespace
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from scripts import sync_huggingface_models as sync


@pytest.mark.parametrize('configured_color', [None, '#ab47bc'])
def test_key_resolves_exact_commit_and_preserves_grants(
    tmp_path, configured_color,
):
    weights = tmp_path / 'models/pt/best_yolo26m.pt'
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b'weights')
    commit = 'c' * 40
    info = SimpleNamespace(
        sha=commit,
        siblings=[
            SimpleNamespace(
                rfilename='models/yolo26/pt/yolo26m.pt',
                lfs=SimpleNamespace(
                    sha256=hashlib.sha256(b'weights').hexdigest(),
                ),
            ),
        ],
    )
    api = Mock()
    api.model_info.return_value = info
    previous = dict(
        tenant_ids=['t1'],
        site_ids=[20],
        enabled=False,
        classes=[{'id': 0, 'code': 'vehicle', 'color': configured_color}],
    )
    with (
        patch.object(sync, 'ROOT', tmp_path),
        patch('huggingface_hub.HfApi', return_value=api),
        patch(
            'ultralytics.YOLO',
            return_value=SimpleNamespace(names={7: 'vehicle'}),
        ),
        patch('huggingface_hub.hf_hub_download') as download,
    ):
        entry = sync.resolve_model(
            'yolo26m', 'owner/models', 'main', None, previous,
        )
    assert entry.version == commit == entry.hf_revision
    assert entry.site_ids == [20]
    assert entry.tenant_ids == ['t1']
    assert not entry.enabled
    assert entry.classes[0].code == 'vehicle'
    assert entry.classes[0].id == 7
    assert entry.classes[0].color == (
        configured_color.upper() if configured_color else '#FFEB3B'
    )
    download.assert_not_called()


def test_unverified_remote_weights_are_never_loaded(tmp_path):
    bad = tmp_path / 'bad.pt'
    bad.write_bytes(b'wrong')
    info = SimpleNamespace(
        sha='d' * 40,
        siblings=[
            SimpleNamespace(
                rfilename='weights.pt',
                lfs=SimpleNamespace(sha256='a' * 64),
            ),
        ],
    )
    api = Mock()
    api.model_info.return_value = info
    with (
        patch.object(sync, 'ROOT', tmp_path),
        patch('huggingface_hub.HfApi', return_value=api),
        patch(
            'huggingface_hub.hf_hub_download', return_value=str(bad),
        ) as download,
        patch('ultralytics.YOLO') as load,
    ):
        with pytest.raises(ValueError, match='hash mismatch'):
            sync.resolve_model(
                'custom', 'owner/models', 'main', 'weights.pt', {},
            )
    download.assert_called_once_with(
        'owner/models', 'weights.pt', revision='d' * 40,
    )
    load.assert_not_called()
