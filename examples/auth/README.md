🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Shared authentication

API services verify OIDC access tokens through `jwt_config.py` and `oidc.py`.
`oidc_identity.py` maps the verified provider subject to local tenant, role,
group, site and feature permissions. HTTP, WebSocket and playback requests
share deployment binding and Redis revocation checks.

## Configuration

Configure `OIDC_ENABLED=true`, `OIDC_ISSUER_URL`, `OIDC_JWKS_URL`,
`OIDC_AUDIENCE`, `OIDC_IDENTITY_PROVIDER` and `OIDC_ACCOUNT_URL` consistently
across services. See `.env.example` for the BFF authorisation-code settings.
A service without an OIDC verifier rejects authenticated requests.

The BFF requires an independent `BFF_TOKEN_ENCRYPTION_KEY` to encrypt tokens
stored in Redis. Web clients receive an opaque HttpOnly session cookie;
native clients use Keycloak Authorization Code + PKCE. Workers use dedicated
Keycloak service accounts and the client-credentials grant.

## Account management

Keycloak owns passwords, recovery and social sign-in. Administrator account
creation and password resets use the Keycloak Admin API and retain local
business authorisation checks. Existing users require immutable Keycloak
identity links; username matching cannot create those links automatically.

Local password login, JWT signing, local refresh grants, password recovery
and the one-time password migration bridge have been removed. Apply
`scripts/migrate_remove_local_passwords.sql` to an existing database when
rolling out this backend; new databases use the updated initialisation SQL.

Redis stores BFF sessions, media capabilities, revocation entries and rate
limits. Token refresh runs against the configured identity provider.

## Cross-repository authentication changes

Coordinate authentication changes with `ai-gateway` and
`visionnaire-flutter-dev`. This backend remains Keycloak/OIDC-only: the gateway's
local password-recovery token table and identity migration do not apply here.
Keep the shared login and session behaviour consistent across repositories.

Native social begin and API quotas share `redis_counters.py`: a single Lua
operation increments the counter and repairs missing expiry without resetting
the window. Authentication and retired-route regression tests must remain part
of any coordinated update.

The gateway groups its account/session/provider core under `examples.identity`.
This backend keeps `examples.auth` and Keycloak as its authority; the internal
package move does not change login, refresh, cookie or deployment contracts.
Shared counter behaviour still matches the gateway's `examples.identity.redis_counters`;
do not copy its local password issuer or recovery migrations into this backend.
