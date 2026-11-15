"""Construction-specific provenance exported with each authenticated result."""
from __future__ import annotations

from typing import Any

MAX_IDENTIFIERS = 100


def provenance(envelope: dict[str, Any]) -> dict[str, Any]:
    result = {
        key: envelope[key]
        for key in (
            'source', 'operation', 'queried_at', 'query', 'limitations',
        )
    }
    operation, data = envelope['operation'], envelope['data']
    if operation == 'construction_my_sites':
        rows, identifier_key = data, 'site_ids'
    elif operation == 'construction_search_violations':
        rows, identifier_key = data['items'], 'violation_ids'
        result.update(
            has_more=data['has_more'], next_cursor=data['next_cursor'],
        )
    else:
        rows, identifier_key = [data], 'violation_ids'
    result.update(
        returned_count=len(rows),
        **{identifier_key: [row['id'] for row in rows[:MAX_IDENTIFIERS]]},
    )
    if len(rows) > MAX_IDENTIFIERS:
        result['ids_truncated'] = True
    return result
