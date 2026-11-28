🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Tenant and deployment management

The Deployment Registry reads its authoritative configuration from `tenants` and `deployments`. A native app exchanges a single-use company activation code for a `deployment_id`, then fetches and verifies the Registry's Ed25519-signed configuration. The `api_base_url` is the canonical HTTPS **API root**, for example `https://api.example.com/hazard/api`. It excludes service paths such as `/db_management`, query strings, fragments and credentials. A route prefix must not give clients another way to select a tenant.

The following administrative APIs require authorisation.

- `POST /admin/tenants`
- `PATCH /admin/tenants/{tenant_id}`
- `POST /admin/deployments`
- `PATCH /admin/deployments/{deployment_id}`
- `GET /admin/deployments`

Only platform super-admins can use these endpoints. Ordinary users, Flutter apps and FCM device registration cannot select a tenant or user by supplying a `tenant_id` or `user_id`.

Login, refresh and BFF session responses include the following deployment contract.

```json
{
  "deployment": {
    "deployment_id": "<UUID>",
    "tenant_id": "<UUID>",
    "config_revision": 7
  }
}
```

If any value differs, the client must discard its tokens and log in again using the verified Registry configuration. It must not create or keep the session. Apps must reject tenant, deployment and API root overrides from deep links, queries or headers.

## Changing a deployment

1. Create or update the deployment through the administrative API. A change to `api_base_url`, `tenant_id` or `status` increments `config_revision`.
2. A trusted operator creates, revokes or delivers the single-use company activation code. The app exchanges it for a `deployment_id` before downloading the Registry configuration.
3. Ask users to log in again with the new signed configuration. An issuer/audience, deployment or revision mismatch on an old access token, refresh token or BFF session returns `409` with `deployment_configuration_changed`.

The reverse proxy must pass the external HTTPS scheme to ASGI correctly. The server looks up `deployments` using the HTTPS origin that actually received the request. It does not trust tenant or deployment headers supplied by clients.

## Signed Deployment Registry

A managed native app can first exchange a single-use company activation code for a deployment ID.

```text
POST /hazard/api/deployment-registry/v1/enrollments/exchange
```

The request body must use `Content-Type: application/json` and contain only `enrollment_code`. A client or edge proxy may add `Accept` and `Cache-Control`. Neither header is required to validate the Registry response. This endpoint does not accept Bearer tokens, cookies, CSRF tokens or refresh tokens. Redis limits attempts by IP and code verifier. A PostgreSQL transaction marks the code as used under a row lock. The database stores only the verifier derived using HMAC with `DEPLOYMENT_ENROLLMENT_CODE_PEPPER`. The raw activation code must never be stored or logged. A successful response has the following form.

```json
{"deployment_id":"<uuid>"}
```

Before login, the app then uses the following endpoint.

```text
GET /hazard/api/deployment-registry/v1/deployments/{deployment_id}
```

Use these values in the Flutter build defines.

```text
DEPLOYMENT_REGISTRY_URL=https://<host>/hazard/api/deployment-registry
```

The Registry has its own FastAPI router and signing service, independent of BFF, login, cookie, CSRF and JWT middleware. A successful response contains exactly nine fields.

```json
{
  "schema_version": 1,
  "deployment_id": "<uuid>",
  "tenant_id": "<uuid>",
  "api_base_url": "https://<host>/hazard/api",
  "config_revision": 1,
  "issued_at": 1735689600,
  "expires_at": 1735693200,
  "key_id": "registry-ed25519-2026-01",
  "signature": "<base64url-ed25519-signature>"
}
```

Inject the Ed25519 PKCS#8 PEM for `DEPLOYMENT_REGISTRY_ED25519_PRIVATE_KEY` through a KMS, a secret manager or a protected runtime environment variable. Keep it out of Git, app configuration and JSON responses. Set `DEPLOYMENT_REGISTRY_TTL_SECONDS` between 1 and 86400. If the key is missing or malformed, or the TTL is invalid, the endpoint returns `503` without issuing an unsigned document. Supply `DEPLOYMENT_ENROLLMENT_CODE_PEPPER` from the same protected secret store, with at least 32 bytes. A missing pepper, Redis outage or database outage causes exchange to return a sanitised `503`. An expired, revoked or already redeemed code returns `410`, an unknown or malformed code returns `403` and rate-limited requests return `429`.

Create codes using the trusted private operator tooling. The migration and this tooling are distributed separately from the application source. Only after a successful commit may the raw code be delivered through an approved company channel. Keep it out of tickets, logs and source control.

## Managing device invitations after login

Tenant admins and platform super-admins can create, list and revoke device invitations **for the current login's deployment**. The tenant and deployment come from the deployment-bound JWT or BFF session. Request bodies must not contain `tenant_id`, `deployment_id`, API or Registry URLs, or signing keys.

Native apps use a Bearer token with the following routes under the API root.

```text
POST   /hazard/api/db_management/deployment-enrollment-codes
GET    /hazard/api/db_management/deployment-enrollment-codes
DELETE /hazard/api/db_management/deployment-enrollment-codes/{id}
```

POST only accepts a body of the form `{"expires_in_minutes":30}`, in the range 1–1440. On creation, the response includes `enrollment_code` **once**. Subsequent GET, audit and error responses expose neither the raw code nor the verifier. The database never exposes these through the management interface. The list returns the canonical lower-case UUID, `expires_at` and an `active`, `redeemed`, `expired` or `revoked` status. DELETE is idempotent and returns `204`.

The web client uses the following routes without holding a Bearer token directly.

```text
/bff/db_management/deployment-enrollment-codes
```

The BFF stores the server-side token in an HttpOnly session and requires a trusted Origin and `X-CSRF-Token` on POST and DELETE. Access and refresh tokens stay on the server.

After applying `20260821_add_deployment_enrollment_code_management_postgres.sql`, `deployment_enrollment_codes.public_id` holds the public UUID used by the management API. The `deployment_enrollment_code_audit_logs` table records the actor, tenant, deployment and time for creation and revocation, excluding codes and hashes. The app must use its built-in or securely configured key ring to find the public key by `key_id` and verify the signature. Sign the following seven fields as UTF-8 JSON in lexicographical key order without whitespace. Exclude `key_id` and `signature`. The signed bytes are separate from the HTTP body.

```json
{"api_base_url":"...","config_revision":1,"deployment_id":"...","expires_at":123,"issued_at":123,"schema_version":1,"tenant_id":"..."}
```

Place the HTTP and HTTPS locations from `nginx.deployment-registry.conf.example` in their respective server blocks, with the HTTPS location ahead of the generic `/hazard/api/` fallback. The route accepts HTTPS only and rejects HTTP directly without redirecting. The example adds `Cache-Control: no-store` at the edge. Clients must accept a valid Registry response without requiring this header.

## Direct access during local development

Production must use the registered HTTPS API root. Only for direct local access to an OIDC-authenticated API such as `http://127.0.0.1:8005/me` may the following be set in the protected local `.env`.

```dotenv
LOCAL_DEVELOPMENT_AUTH_ENABLED=true
LOCAL_DEVELOPMENT_DEPLOYMENT_ID=<registered-deployment-uuid>
```

This exception requires both the socket peer and the `Host` to be loopback (`127.0.0.1`, `::1` or `localhost`). The server environment always supplies the deployment. Clients cannot use an arbitrary origin or tenant.
