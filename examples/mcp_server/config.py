from __future__ import annotations

import os

from dotenv import load_dotenv

from examples.mcp_server.schemas import TransportConfig

# ---------------------------------------------------------------------------
# Load environment variables from .env for local development.
# ---------------------------------------------------------------------------
load_dotenv()


# ---------------------------------------------------------------------------
# Default configuration values for fallback
# ---------------------------------------------------------------------------
DEFAULT_ENV: dict[str, str] = {
    'MCP_TRANSPORT': 'streamable-http',
    'MCP_HOST': '0.0.0.0',
    'MCP_PORT': '8092',
    'MCP_PATH': '/mcp',
    'MCP_SSE_PATH': '/sse',
    'MCP_DEBUG': 'false',
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_env_var(key: str, default: str | None = None) -> str:
    """Return the value of an environment variable or a safe default."""
    return os.getenv(key, default or DEFAULT_ENV.get(key, ''))


def get_env_bool(key: str, default: bool = False) -> bool:
    """Interpret an environment variable as a boolean."""
    value = get_env_var(key, '1' if default else '').lower()
    return value in ('1', 'true', 'yes', 'on')


def get_env_int(key: str, default: int = 0) -> int:
    """Interpret an environment variable as an integer."""
    val = get_env_var(key, str(default))
    return (
        int(val)
        if val.isdigit() or (val.startswith('-') and val[1:].isdigit())
        else default
    )


def get_env_float(key: str, default: float = 0.0) -> float:
    """Interpret an environment variable as a floating point number."""
    try:
        return float(get_env_var(key, str(default)))
    except ValueError:
        return default


def get_env_list(
    key: str,
    sep: str = ',',
    default: list[str] | None = None,
) -> list[str]:
    """Interpret an environment variable as a delimited list of strings."""
    value = get_env_var(key)
    if not value:
        return default or []
    return [x.strip() for x in value.split(sep) if x.strip()]


# ---------------------------------------------------------------------------
# Transport configuration builder
# ---------------------------------------------------------------------------

def get_transport_config() -> TransportConfig:
    """Construct and return the transport configuration."""
    valid_transports = ['stdio', 'streamable-http', 'sse']
    transport = get_env_var(
        'MCP_TRANSPORT', DEFAULT_ENV['MCP_TRANSPORT'],
    ).lower()
    if transport not in valid_transports:
        print(
            f"Warning: Invalid transport '{transport}', "
            "falling back to 'stdio'.",
        )
        transport = 'stdio'

    config: TransportConfig = {
        'transport': transport,
        'host': get_env_var('MCP_HOST', DEFAULT_ENV['MCP_HOST']),
        'port': get_env_int('MCP_PORT', int(DEFAULT_ENV['MCP_PORT'])),
        'path': get_env_var('MCP_PATH', DEFAULT_ENV['MCP_PATH']),
        'sse_path': get_env_var('MCP_SSE_PATH', DEFAULT_ENV['MCP_SSE_PATH']),
        'debug': get_env_bool(
            'MCP_DEBUG',
            DEFAULT_ENV['MCP_DEBUG'].lower() in ('1', 'true', 'yes'),
        ),
    }

    return config
