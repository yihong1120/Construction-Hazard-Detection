from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock
from unittest.mock import patch

import pytest
from mcp import ClientSession
from mcp import StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.shared.memory import create_connected_server_and_client_session

from examples.mcp_server import main


@pytest.mark.anyio
async def test_stdio_protocol_starts_without_model_dependencies():
    """A real SDK client must initialize, list tools and call the stdio
    server.
    """
    script = """
import sys, importlib.abc, asyncio
class BlockModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {
            'torch', 'cv2', 'numpy', 'sklearn', 'shapely', 'ultralytics',
        }:
            raise ImportError(
                'MCP startup imported optional dependency: ' + fullname
            )
sys.meta_path.insert(0, BlockModels())
from examples.mcp_server.main import run_server
asyncio.run(run_server())
"""
    parameters = StdioServerParameters(
        command=sys.executable, args=['-c', script],
        env={
            'MCP_TRANSPORT': 'invalid-use-stdio',
            'PYTHON_DOTENV_DISABLED': '1',
        },
        cwd=Path(__file__).resolve().parents[3],
    )
    async with stdio_client(parameters) as (read, write):
        async with ClientSession(
            read, write, read_timeout_seconds=timedelta(seconds=15),
        ) as client:
            initialized = await client.initialize()
            assert (
                initialized.serverInfo.name == 'construction-hazard-detection'
            )
            listed = await client.list_tools()
            names = {tool.name for tool in listed.tools}
            assert len(names) == 22
            assert {
                'notify_fcm_send',
                'inference_detect_frame',
                'record_send_violation',
            } <= names
            assert {
                'notify_line_push',
                'notify_telegram_send',
                'notify_messenger_send',
                'notify_wechat_send',
                'notify_broadcast_send',
            } <= names
            assert 'streaming_start_detection' not in names
            upload_schema = next(
                t.inputSchema
                for t in listed.tools
                if t.name == 'record_send_violation'
            )
            assert {
                'site',
                'stream_name',
                'model_id',
                'model_version',
                'warnings',
            } <= set(
                upload_schema['required'],
            )
            result = await client.call_tool(
                'utils_calculate_polygon_area',
                {'polygon_points': [[0, 0], [2, 0], [0, 3]]},
            )
            assert not result.isError
            assert result.structuredContent['area'] == 3
            rejected = await client.call_tool(
                'record_send_violation', {'image_base64': 'bad'},
            )
            assert rejected.isError


@pytest.mark.anyio
async def test_protocol_uses_modern_upload_contract_and_fcm():
    records = AsyncMock()
    records.send_violation.return_value = {'success': True, 'record_id': '42'}
    notify = AsyncMock()
    notify.fcm_send.return_value = {'success': True}
    with (
        patch.object(main, 'record_tools', records),
        patch.object(main, 'notify_tools', notify),
    ):
        async with create_connected_server_and_client_session(
            main.mcp,
        ) as client:
            result = await client.call_tool(
                'record_send_violation',
                {
                    'image_base64': 'aW1hZ2U=',
                    'detections': [],
                    'warnings': {'no_helmet': {}},
                    'site': 'site',
                    'stream_name': 'camera',
                    'model_id': 'model',
                    'model_version': 'v1',
                },
            )
            assert not result.isError
            assert result.structuredContent['record_id'] == '42'
            assert (
                records.send_violation.call_args.kwargs['model_version']
                == 'v1'
            )
            result = await client.call_tool(
                'notify_fcm_send',
                {
                    'site': 'site',
                    'stream_name': 'camera',
                    'warnings': {'no_helmet': {}},
                },
            )
            assert not result.isError and result.structuredContent['success']
        records.close.assert_awaited_once()
        notify.close.assert_awaited_once()
