"""Construction tool contracts discovered by clients over MCP."""
from __future__ import annotations

from typing import Annotated
from typing import Any

from mcp.server.mcpserver import Context
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.server.mcpserver.tools import Tool
from mcp.types import ToolAnnotations
from pydantic import ConfigDict
from pydantic import create_model
from pydantic import Field
from pydantic import ValidationError

from examples.mcp_server.agent_reads.schemas import GetViolation
from examples.mcp_server.agent_reads.schemas import MySites
from examples.mcp_server.agent_reads.schemas import SearchViolations
from examples.mcp_server.agent_reads.service import query


async def perform(
    ctx: Context,
    operation: str,
    args: MySites | SearchViolations | GetViolation,
) -> dict[str, Any]:
    request = ctx.request_context.request
    if request is None:
        raise ToolError('工安工具需要本人已驗證的 HTTP 請求。')
    verified = getattr(request.state, 'agent_read_identity', None)
    db = getattr(request.state, 'agent_read_database', None)
    if verified is None or db is None:
        raise ToolError('工安工具需要本人有效的 OIDC 登入。')
    return await query(request, db, verified, operation, args)


async def construction_my_sites(ctx: Context) -> dict[str, Any]:
    """列出本人目前可存取的工地；唯讀。取得真實 site_id 後再查違規。"""
    return await perform(ctx, 'construction_my_sites', MySites())


async def construction_search_violations(
    ctx: Context,
    site_id: Annotated[int | None, Field(gt=0, strict=True)] = None,
    keyword: Annotated[
        str | None, Field(min_length=1, max_length=200),
    ] = None,
    start_time: str | None = None,
    end_time: str | None = None,
    limit: Annotated[int, Field(ge=1, le=100, strict=True)] = 20,
    cursor: Annotated[
        str | None, Field(min_length=1, max_length=2000),
    ] = None,
) -> dict[str, Any]:
    """依工地、關鍵詞及成對帶時區的起訖時間搜尋違規；唯讀。

    回傳一頁與 next_cursor／has_more；單頁筆數不是完整總數。
    """
    try:
        args = SearchViolations.model_validate(
            {
                'site_id': site_id,
                'keyword': keyword,
                'start_time': start_time,
                'end_time': end_time,
                'limit': limit,
                'cursor': cursor,
            },
        )
    except ValidationError:
        raise ToolError('工安查詢參數無效；時間須成對、帶時區且順序正確。') from None
    return await perform(ctx, 'construction_search_violations', args)


async def construction_get_violation(
    ctx: Context,
    violation_id: Annotated[int, Field(gt=0, strict=True)],
) -> dict[str, Any]:
    """以搜尋取得的真實 violation_id 讀單筆違規，仍依本人工地權限檢查。

    唯讀；不下載圖片、不推論、不通知或修改紀錄。
    """
    return await perform(
        ctx, 'construction_get_violation', GetViolation(
            violation_id=violation_id,
        ),
    )


def build_server() -> MCPServer:
    annotation = ToolAnnotations(
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    )
    tools = []
    for handler in (
        construction_my_sites,
        construction_search_violations,
        construction_get_violation,
    ):
        tool = Tool.from_function(handler, annotations=annotation)
        tool.fn_metadata.arg_model = create_model(
            handler.__name__ + 'Input',
            __base__=tool.fn_metadata.arg_model,
            __config__=ConfigDict(extra='forbid'),
        )
        tool.parameters = tool.fn_metadata.arg_model.model_json_schema()
        tools.append(tool)
    return MCPServer('construction-authorized-reads', tools=tools)
