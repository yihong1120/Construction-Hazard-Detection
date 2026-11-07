from __future__ import annotations

from copy import deepcopy

import pytest

from deploy.dev.database_parity import compare
from deploy.dev.database_parity import QUERIES


@pytest.fixture
def baseline():
    schema = {kind: [{'name': kind}] for kind in QUERIES}
    schema['columns'] = [
        {
            'schema': 'public',
            'relation': 'events',
            'name': 'id',
            'default': None,
        },
        {
            'schema': 'construction_hazard_detection',
            'relation': 'events',
            'name': 'id',
            'default': '1',
        },
    ]
    return {
        'schema': schema,
        'catalog': [{'feature_name': 'file_manage', 'description': 'Files'}],
        'operator': {
            'role': 'admin',
            'features': ['file_manage'],
            'max_allowed_streams': 99,
        },
        'migrations': [{'version': 'baseline', 'checksum': 'verified'}],
    }


def test_schema_names_are_preserved_in_parity(baseline):
    result = compare(baseline, deepcopy(baseline))
    assert result['matched'] and result['dev_counts']['tables'] == 2


@pytest.mark.parametrize('kind', QUERIES)
def test_missing_schema_objects_fail_parity(baseline, kind):
    changed = deepcopy(baseline)
    changed['schema'][kind].pop()
    assert kind in compare(baseline, changed)['differences']


def test_default_changes_and_authorization_limits_fail_parity(baseline):
    changed = deepcopy(baseline)
    changed['schema']['columns'][1]['default'] = '2'
    changed['operator']['max_allowed_streams'] = 1
    changed['catalog'] = []
    changed['migrations'][0]['checksum'] = 'different'
    result = compare(baseline, changed)
    assert not result['matched']
    assert set(result['differences']) == {
        'columns',
        'operator',
        'catalog',
        'migrations',
    }
