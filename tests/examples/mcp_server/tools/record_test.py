from __future__ import annotations

import base64
from datetime import datetime
from unittest.mock import AsyncMock
from unittest.mock import patch

import httpx
import pytest

from examples.mcp_server.tools.record import RecordTools


def upload(**overrides):
    return {
        'image_base64': base64.b64encode(b'image').decode(),
        'detections': [[0, 0, 1, 1, .9, 0, 42]],
        'warnings': {'no_helmet': {'count': 1}}, 'site': 'site',
        'stream_name': 'camera', 'model_id': 'model', 'model_version': 'v1',
        'timestamp': '2026-10-04T10:00:00+08:00', **overrides,
    }


@pytest.mark.anyio
async def test_upload_real_multipart_contract_and_close():
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(200, json={'violation_id': '42'})

    tool = RecordTools()
    with patch('examples.mcp_server.tools.record.ViolationSender') as factory:
        from src.violation_sender import ViolationSender
        sender = ViolationSender()
        sender.token_manager = AsyncMock()
        sender.token_manager.get_valid_token.return_value = 'oidc-token'
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        sender._client = client
        factory.return_value = sender
        result = await tool.send_violation(**upload())
        assert result == {'success': True, 'record_id': '42'}
        request = requests[0]
        assert request.headers['Authorization'] == 'Bearer oidc-token'
        assert request.url.path == '/upload'
        for field, value in [
            ('model_id', 'model'), ('model_version', 'v1'),
            ('site', 'site'), ('stream_name', 'camera'),
        ]:
            assert f'name="{field}"\r\n\r\n{value}'.encode() in request.content
        assert b'"no_helmet"' in request.content
        await tool.close()
        assert client.is_closed
        assert tool._violation_sender is None
    await tool.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    'overrides', [
        {'image_base64': '%invalid'}, {'image_base64': ''}, {
            'image_base64': 'data:broken',
        },
        {'image_base64': 'data:image/jpeg;base64,aW1hZ2U='},
        {'model_id': ''}, {'model_version': ''}, {'timestamp': 'invalid'},
        {'site': ''}, {'detections': [{'bbox': [0, 0, 1, 1]}]}, {
            'detections': [[0, 0, 1, 1, .9, 0]],
        }, {'warnings': {'message': 'legacy'}},
    ],
)
async def test_validation_before_network(overrides):
    tool = RecordTools()
    tool._violation_sender = AsyncMock()
    tool._violation_sender.send_violation.return_value = '1'
    result = await tool.send_violation(**upload(**overrides))
    if overrides.get('image_base64', '').startswith('data:image/'):
        assert result['success']
    else:
        assert not result['success']
        tool._violation_sender.send_violation.assert_not_awaited()


@pytest.mark.anyio
async def test_size_limit_and_timezone_and_backend_failure(monkeypatch):
    monkeypatch.setattr('examples.mcp_server.tools.record._MAX_IMAGE_BYTES', 1)
    tool = RecordTools()
    assert not (await tool.send_violation(**upload()))['success']
    monkeypatch.setattr('examples.mcp_server.tools.record._MAX_IMAGE_BYTES', 2)
    assert not (await tool.send_violation(**upload(image_base64='aW1h')))[
        'success'
    ]
    monkeypatch.setattr(
        'examples.mcp_server.tools.record._MAX_IMAGE_BYTES', 1000,
    )
    sender = AsyncMock()
    tool._violation_sender = sender
    sender.send_violation.side_effect = RuntimeError('private-token')
    result = await tool.send_violation(**upload())
    assert not result['success'] and 'private-token' not in result['message']
    assert sender.send_violation.call_args.kwargs[
        'detection_time'
    ] == datetime.fromisoformat(
        upload()['timestamp'],
    )
    sender.send_violation.side_effect = None
    sender.send_violation.return_value = None
    assert not (await tool.send_violation(**upload()))['success']
    await tool.close()
    sender.close.assert_awaited_once()


@pytest.mark.anyio
async def test_batch_does_not_abort_after_malformed_record():
    tool = RecordTools()
    sender = AsyncMock()
    sender.send_violation.return_value = '42'
    tool._violation_sender = sender
    result = await tool.batch_send_violations([upload(), {}, upload()])
    assert result['successful'] == 2 and result['failed'] == 1
    assert len(result['results']) == 3
    assert sender.send_violation.await_count == 2
    for batch in ([], [upload()] * 101):
        with pytest.raises(ValueError):
            await tool.batch_send_violations(batch)
