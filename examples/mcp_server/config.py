from __future__ import annotations

from dotenv import load_dotenv

from examples.mcp_server.schemas import TransportConfig
from examples.mcp_server.settings import McpTransportSettings

load_dotenv()


def get_transport_config() -> TransportConfig:
    config = McpTransportSettings()
    return TransportConfig(
        transport=config.transport,
        host=config.host,
        port=config.port,
        path=config.path,
        sse_path=config.sse_path,
        debug=config.debug,
    )
