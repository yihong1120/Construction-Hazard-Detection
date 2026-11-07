from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import configure_exit as x


class ExitAccessTests(unittest.TestCase):
    def setUp(self):
        self.peer = {
            'id': 'employee-device',
            'ip': '10.203.252.213',
            'user_id': 'employee',
            'approval_required': False,
        }
        self.user = {
            'id': 'employee',
            'is_blocked': False,
            'pending_approval': False,
            'idp_id': 'company-connector',
        }

    def api(self, path, **kwargs):
        return {'peers': [self.peer], 'users': [self.user]}[path]

    def select(self):
        with patch.object(x.d, 'api', side_effect=self.api), patch.object(
            x.s,
            'provider_for',
            return_value={'id': 'company-connector'},
        ):
            return x.active_client('private-fixture-token', '10.203.252.213')

    def test_approved_company_device(self):
        self.assertEqual(self.select()['id'], 'employee-device')

    def test_blocked_pending_and_other_provider_are_denied(self):
        for field, denied, original in [
            ('is_blocked', True, False),
            ('pending_approval', True, False),
            (
                'idp_id',
                'local-provider',
                'company-connector',
            ),
        ]:
            with self.subTest(field=field):
                self.user[field] = denied
                with self.assertRaises(RuntimeError):
                    self.select()
                self.user[field] = original

    def test_setup_key_server_cannot_be_added_as_employee(self):
        self.peer['user_id'] = ''
        with self.assertRaises(RuntimeError):
            self.select()

    def test_unknown_device_cannot_be_added(self):
        self.peer['ip'] = '10.203.0.99'
        with self.assertRaises(RuntimeError):
            self.select()

    def test_api_omits_empty_acl_and_expands_group_peers(self):
        self.assertEqual(
            x.comparable(
                {'peers': [{'id': 'a'}]},
                ['peers'],
            ),
            {'peers': ['a']},
        )
        self.assertEqual(
            x.comparable({}, ['access_control_groups']),
            {
                'access_control_groups': [],
            },
        )

    def test_manually_modified_group_is_not_overwritten(self):
        with patch.object(
            x.d,
            'api',
            return_value={
                'name': x.CLIENT_GROUP,
                'peers': [{'id': 'unexpected-device'}],
            },
        ) as api:
            with self.assertRaises(RuntimeError):
                x.ensure(
                    'groups',
                    'client_group_id',
                    {
                        'name': x.CLIENT_GROUP,
                        'peers': ['employee-device'],
                    },
                    {'client_group_id': 'managed-group'},
                    'fixture-token',
                )
            self.assertEqual(api.call_count, 1)

    def test_add_device_preserves_existing_devices_and_records_scope(self):
        state = {
            'complete': True,
            'client_group_id': 'managed-group',
            'client_peer_ids': ['existing-device'],
            'client_ips': ['10.203.0.3'],
        }
        with tempfile.TemporaryDirectory() as tmp, patch.object(
            x,
            'STATE',
            Path(tmp) / 'state.json',
        ), patch.object(x, 'verify'), patch.object(
            x,
            'active_client',
            return_value=self.peer,
        ), patch.object(
            x.d,
            'api',
        ) as api:
            x.add_device('fixture-token', state, self.peer['ip'])
            api.assert_called_once_with(
                'groups/managed-group',
                'PUT',
                {
                    'name': x.CLIENT_GROUP,
                    'peers': [
                        'existing-device',
                        'employee-device',
                    ],
                },
                'fixture-token',
            )
            self.assertEqual(
                json.loads(x.STATE.read_text())['client_peer_ids'],
                ['existing-device', 'employee-device'],
            )


if __name__ == '__main__':
    unittest.main()
