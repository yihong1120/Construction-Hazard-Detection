"""Reuse verified identity and domain services without self-HTTP."""
from __future__ import annotations

import json
from datetime import datetime
from datetime import timezone
from typing import Any

from fastapi import HTTPException
from fastapi import Request
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import BaseModel
from pydantic import ValidationError

from examples.auth.jwt_config import JwtAuthorizationCredentials
from examples.mcp_server.agent_reads.evidence import provenance
from examples.mcp_server.agent_reads.projection import SiteRead
from examples.mcp_server.agent_reads.projection import ViolationDetail
from examples.mcp_server.agent_reads.projection import ViolationPage
from examples.mcp_server.agent_reads.schemas import GetViolation
from examples.mcp_server.agent_reads.schemas import MySites
from examples.mcp_server.agent_reads.schemas import SearchViolations
from examples.violation_records import violation_query_service
from examples.violation_records import violation_review_service
from examples.violation_records.violation_queries import ViolationFilters

MAX_RESPONSE_BYTES = 1_048_576
READ_LIMITATION = '僅查詢目前登入帳號可見的資料；未讀取圖片或執行修改。'
PAGE_LIMITATION = (
    '此結果只涵蓋一頁；has_more 與 next_cursor 決定是否仍有下一頁，'
    '沒有完整筆數。'
)


def json_value(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode='json')
    return value


def projection(operation: str, value: Any, args: BaseModel) -> Any:
    value = json_value(value)
    if operation == 'construction_my_sites':
        if not isinstance(value, list):
            raise ValueError('Invalid site response')
        return [
            SiteRead.model_validate(json_value(row)).model_dump()
            for row in value
        ]
    if operation == 'construction_search_violations':
        page = ViolationPage.model_validate(value)
        if (
            not isinstance(args, SearchViolations)
            or len(page.items) > args.limit
            or page.has_more != bool(page.next_cursor)
            or (page.has_more and not page.items)
            or (
                page.next_cursor is not None
                and (
                    not page.next_cursor.strip()
                    or len(page.next_cursor) > 2000
                )
            )
        ):
            raise ValueError('Invalid page response')
        return page.model_dump()
    detail = ViolationDetail.model_validate(value)
    if not isinstance(args, GetViolation) or detail.id != args.violation_id:
        raise ValueError('Invalid record identity')
    return detail.model_dump()


async def query(
    request: Request,
    db: Any,
    credentials: JwtAuthorizationCredentials,
    operation: str,
    args: MySites | SearchViolations | GetViolation,
) -> dict[str, Any]:
    """Run one existing domain read with the current request identity."""
    try:
        if operation == 'construction_my_sites' and isinstance(args, MySites):
            value = await violation_query_service.get_my_sites(db, credentials)
        elif operation == 'construction_search_violations' and isinstance(
            args, SearchViolations,
        ):
            value = await violation_query_service.get_violations(
                request,
                db,
                credentials,
                filters=ViolationFilters(
                    site_id=args.site_id,
                    keyword=args.keyword,
                    start_time=args.start_time,
                    end_time=args.end_time,
                    cursor=args.cursor,
                ),
                limit=args.limit,
            )
        elif operation == 'construction_get_violation' and isinstance(
            args, GetViolation,
        ):
            value = await violation_review_service.get_single_violation(
                args.violation_id, request, db, credentials,
            )
        else:
            raise ToolError('未提供此工安操作。')
    except HTTPException as failure:
        messages = {
            401: '工安登入已失效，請重新登入。',
            403: '目前帳號無權讀取此工安資料。',
            404: '工安紀錄不存在或不在目前帳號可見範圍。',
            409: '工安部署設定已變更，請重新登入。',
            429: '工安服務目前受到速率限制，請稍後再查詢。',
        }
        raise ToolError(
            messages.get(failure.status_code, '工安服務暫時無法完成查詢。'),
        ) from None
    try:
        data = projection(operation, value, args)
        if len(json.dumps(data, ensure_ascii=False).encode()) > (
            MAX_RESPONSE_BYTES
        ):
            raise ToolError('工安回應超過讀取上限，請縮小查詢範圍。')
    except (TypeError, ValueError, ValidationError):
        raise ToolError('工安資料格式不完整，未將資料當作有效結果。') from None
    limitations = [READ_LIMITATION]
    if isinstance(args, SearchViolations):
        limitations.append(PAGE_LIMITATION)
    envelope = {
        'source': str(request.url.replace(query=None)),
        'operation': operation,
        'queried_at': datetime.now(timezone.utc).isoformat(),
        'query': args.model_dump(mode='json', exclude_none=True),
        'data': data,
        'limitations': limitations,
    }
    envelope['provenance'] = provenance(envelope)
    return envelope
