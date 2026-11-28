🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Keycloak deployment

Replace the example domains, addresses and identifiers with the values for your deployment.

This Compose configuration uses the host's PostgreSQL instance. Give Keycloak a dedicated `keycloak` database. Never point it at Visionnaire's `construction_hazard_detection` database or its tables.

When PostgreSQL listens only on `127.0.0.1`, this configuration runs Keycloak with host networking and binds its HTTP listener only to `127.0.0.1:8081`. All external traffic must enter through the existing Nginx HTTPS proxy at `/keycloak/`.

Before deployment, configure the following variables in `.env`.

- `KEYCLOAK_ADMIN_USERNAME` and `KEYCLOAK_ADMIN_PASSWORD`
- `KEYCLOAK_DB_USERNAME` and `KEYCLOAK_DB_PASSWORD`
- `KEYCLOAK_USER_LINKER_CLIENT_SECRET`
- All `OIDC_*` values
- `HCAPTCHA_SITE_KEY` and `HCAPTCHA_SECRET_KEY`

The database account must be able to connect to the dedicated `keycloak` database. A PostgreSQL administrator can create the database once during initial deployment. Neither Keycloak nor the Visionnaire application needs additional superuser privileges.

## Session lifetimes

Visionnaire Web, the development site and native apps use these lifetimes.

- The idle timeout is 7 days (604800 seconds) and renews while the session is in use.
- A login lasts at most 90 days (7776000 seconds), including periods of continuous use.
- Access tokens last 5 minutes. A regular refresh token renews them automatically. Clients do not request offline tokens.

The BFF's `BFF_SESSION_TTL_SECONDS` remains 7 days. Authenticated requests renew the idle timeout in Redis and the browser cookie. Keycloak enforces the 90-day maximum. The shared realm's SSO idle and maximum lifetimes are 7 and 90 days respectively. Visionnaire clients set the same values explicitly via `client.session.idle.timeout` and `client.session.max.lifespan`. Other clients keep the realm's existing client defaults of 8 hours idle and 10 hours maximum.

For an existing deployment, apply the realm, client and startup settings together. If the BFF already uses a 7-day idle timeout, this change to the maximum lifetime does not require a BFF restart. Editing the realm import template alone does not update an existing realm.

## Login protection

The image includes the custom `visionnaire-hcaptcha` Keycloak authenticator and the `visionnaire` login theme. At container startup, the Keycloak Admin API creates and binds the `visionnaire-browser` flow. Users with a valid Keycloak SSO cookie can continue their sessions. New password logins authenticate the credentials first, then require hCaptcha. The hCaptcha secret lives only in the container environment and stays out of realm JSON, Keycloak administrative settings and the Flutter frontend.

The web BFF must use regular OIDC refresh tokens and must not request `offline_access`. Regular refresh extends the online SSO session shared by Visionnaire and Account Console. Offline tokens support unattended work and would keep an API session alive after browser SSO expires, unexpectedly requiring another login when the user opens Account Console.

The provider sends the server-side secret, single-use challenge token and expected site key to hCaptcha's `siteverify`. If the service fails, the token is invalid or the site key differs, verification fails closed. Set the site key's **Domain Allowlist** in the hCaptcha Dashboard to `example.org`, which covers `app.example.org`. Do not use the `hostname` returned by `siteverify` to restrict domains. This value is browser-derived telemetry.

Existing Visionnaire users may have no email address, given name or surname. Deployment disables the Keycloak `VERIFY_PROFILE` required action so these users can still log in directly after the OIDC switch. They can complete their profile in Keycloak Account Console after login.

The realm's `forceExpiredPasswordChange(180)` policy expires passwords 180 days after the Keycloak credential was last created or updated. Users must change the password at login once it expires. Account creation and administrative password resets use non-temporary passwords, without triggering an extra first-login change.

Set `frame-src` in the Keycloak realm's `browserSecurityHeaders.contentSecurityPolicy` to allow both `https://hcaptcha.com` and `https://*.hcaptcha.com`. The hCaptcha iframe uses this response header. Setting it only in the login theme's `theme.properties` is insufficient.

### Username and password verification

Password login uses Keycloak's standard `auth-username-password-form` followed by hCaptcha. The legacy Visionnaire password migration provider has been removed. The startup script removes any leftover migration execution from the existing browser forms flow and re-enables the standard password authenticator.

## Google and Apple sign-in

Keycloak Identity Broker handles Google and Apple sign-in for the web BFF. At startup the Identity Provider Redirector is placed before the password form in the browser flow. The Google and Apple buttons open third-party sign-in and only the password branch proceeds to hCaptcha.

Providers are disabled by default. When setting `KEYCLOAK_GOOGLE_ENABLED=true` or `KEYCLOAK_APPLE_ENABLED=true`, supply the corresponding client ID and secret. Otherwise startup disables the provider to keep an unusable button out of the UI. For external callback URLs, Apple JWT secret rotation and acceptance tests on all three platforms, see the [Keycloak social sign-in deployment specification](../../docs/zh/keycloak_social_login.md).

Flutter on iOS and Android can also use the official Google or Apple SDK. The SDK does not issue Visionnaire JWTs and the flow does not use Keycloak's retired external Token Exchange v1. The Visionnaire API verifies the provider assertion and nonce. A Keycloak custom authenticator then consumes a PKCE-bound, single-use proof using loopback HMAC, returning to the standard Authorization Code + PKCE flow. For the API, account linking transaction and frontend contract, see the [native social credential exchange specification](../../docs/zh/native_social_exchange.md).

## Native Flutter client

Startup also creates the `visionnaire-mobile` public OIDC client. It allows only Authorization Code + PKCE (S256), disables implicit and password/direct grants and uses no client secret. Its callback is fixed at `com.example.visionnaire:/oauthredirect`. Access tokens include the `visionnaire-api` audience. The Flutter app must not hold a client secret or hCaptcha secret.

## Source of login policy settings

Edit `src/settings/session_policy.json` and run the following command.

```bash
python -m deploy.keycloak.session_policy
python -m deploy.keycloak.session_policy --check
```

CI checks consistency across the generated realm template, `session-policy.env` and `.env.example`. BFF defaults and the development setup tool read the same policy directly. The Keycloak startup script reads the generated shell settings, so include `session-policy.env` when rebuilding the image. Startup updates existing clients' lifetimes, reads them back for verification and preserves other attributes. Development clients that have not been created may be skipped according to policy.

For an existing realm, check the policy before explicitly applying it. Routine API startup does not modify the shared Keycloak instance.

```bash
python -m deploy.keycloak.apply_session_policy --server https://HOST/keycloak --realm visionnaire --credentials /private/admin.json
python -m deploy.keycloak.apply_session_policy --server https://HOST/keycloak --realm visionnaire --credentials /private/admin.json --apply --backup /private/new-session-backup.json
```

The credentials file accepts `username` and `password`, or the existing `KC_BOOTSTRAP_ADMIN_USERNAME` and `KC_BOOTSTRAP_ADMIN_PASSWORD`. Applying updates preserves other client attributes, saves a private backup, writes the changes and reads them back for verification. A partial failure restores entries already changed. Use a new backup path to preserve the previous recovery point.
