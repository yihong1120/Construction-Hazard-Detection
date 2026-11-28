🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Read-only construction safety MCP for individual users

This package defines the names, descriptions, input schemas, response fields, pagination and query provenance for site and violation tools. `ai-gateway` handles agent and LLM execution, conversations and usage through its general MCP connector. This repository owns safety queries, OIDC verification and tenant and site permissions.

## Start the service

Use the project's existing database, Redis, OIDC and deployment settings. Install the MCP extra and start the application:

```bash
uv sync --locked --extra mcp
uv run --no-sync uvicorn examples.mcp_server.agent_reads.app:create_app \
  --factory --host 127.0.0.1 --port 8093
```

The local endpoint is `/mcp`. The read-only service is optional and runs separately. The ordinary violations HTTP API can run without MCP. To mount it in an ASGI application with an existing database, Redis and OIDC lifespan, use `install(app, path='/mcp')`. The default service startup and public Nginx routes do not install it automatically.

Configure a registered public HTTPS origin for production. The trusted request scheme and Host identify the safety deployment. Preserve the trusted forwarding rules at the reverse proxy. A loopback URL is insufficient to identify a public deployment. Every MCP HTTP request, including discovery and tool listing, passes through `jwt_access`. Supply the user's bearer token. The endpoint rejects browser session cookies and has no fallback to a worker token or `MCP_STATIC_BEARER`.

## Tools

| Server tool | Data source |
| --- | --- |
| `construction_my_sites` | `violation_query_service.get_my_sites` |
| `construction_search_violations` | `violation_query_service.get_violations` |
| `construction_get_violation` | `violation_review_service.get_single_violation` |

Tools call the existing domain services directly and share their data models and access checks. Each request retains its own verified identity and database session until the SDK completes the response. All tools declare read-only, non-destructive behaviour and accept only the parameters in their public schemas. Notifications, reviews, uploads, image downloads and live inference are outside their scope.

Search accepts a site ID, keywords, paired time-zone-aware start and end times, a limit from 1 to 100, and a cursor. The response contains `items`, `next_cursor` and `has_more`. The page length describes that page only. Results omit image addresses, internal paths and unrelated fields. The service generates `provenance` containing the source, query time, filters, page scope, up to 100 identifiers and limitations. It flags truncated identifier lists. The gateway forwards these fields without interpreting them.

## Gateway configuration

Add this entry to the gateway's existing Company Agent JSON configuration, using the registered MCP URL:

```json
{
  "mcp_servers": [
    {
      "name": "safety",
      "url": "https://construction.example/agent-mcp",
      "allowed_tools": [
        "construction_my_sites",
        "construction_search_violations",
        "construction_get_violation"
      ]
    }
  ]
}
```

The gateway builds `external_safety_*` tools from the server's descriptions and schemas. For each request it obtains the individual's current OIDC token. The issuer, JWKS and audience must match this service. The `(provider, sub)` pair must identify an active account in the tenant with access to the requested site. Signing in to the WebUI alone does not grant safety permissions. Identity comes from the verified token, never a name, email address or LiteLLM key. This integration preserves the login and session contracts and requires no identity table migration.

The `examples.mcp_server.main` worker on stdio or port 8092 uses its configured server identity. It remains a separate entry point and cannot serve as a fallback for queries from multiple individual users.

## Verification

```bash
python -m pytest tests/examples/mcp_server/agent_reads_test.py \
  tests/examples/violation_records/agent_read_contract_test.py -q
```

The tests use a database double and verified test identities over the actual MCP SDK transport. They cover discovery, schemas, tool calls, anonymous rejection, concurrent identities and permission errors. They require neither a live model nor production site data.
