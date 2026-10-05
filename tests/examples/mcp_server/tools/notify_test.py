from __future__ import annotations

from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest

from examples.mcp_server.tools.notify import NotifyTools


@pytest.mark.anyio
async def test_fcm_sender_reused_and_closed():
    with patch('examples.mcp_server.tools.notify.FCMSender') as factory:
        sender = AsyncMock()
        factory.return_value = sender
        tool = NotifyTools()
        sender.send_fcm_message_to_site.return_value = True
        assert (
            await tool.fcm_send(
                'site', 'camera', {'no_helmet': {}}, 'image.jpg', 42,
            )
        )['success']
        sender.send_fcm_message_to_site.assert_awaited_once_with(
            'site', 'camera', {'no_helmet': {}}, 'image.jpg', 42,
        )
        sender.send_fcm_message_to_site.return_value = False
        assert not (await tool.fcm_send('site', 'camera', {'no_helmet': {}}))[
            'success'
        ]
        factory.assert_called_once()
        await tool.close()
        sender.close.assert_awaited_once()
        assert tool._sender is None
        await tool.close()


@pytest.mark.anyio
@pytest.mark.parametrize(
    'site,stream,warnings',
    [('', 'cam', {'w': {}}), ('site', ' ', {'w': {}}), ('site', 'cam', {})],
)
async def test_missing_notification_fields_rejected_before_network(
    site, stream, warnings,
):
    tool = NotifyTools()
    with pytest.raises(ValueError):
        await tool.fcm_send(site, stream, warnings)
    assert tool._sender is None
