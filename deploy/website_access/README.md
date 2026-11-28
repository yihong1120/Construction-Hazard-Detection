🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Website IP access management

Replace the example domains, addresses and identifiers with the values for your deployment.

Website IP policies are managed separately from construction `Site` records. Global administrators can manage IPv4/IPv6 addresses and CIDR ranges for `main`, `ai` and `wiki`. Ordinary users and group administrators cannot change the network boundary for the websites. The frontend uses `can_manage_website_access` in `GET /admin/capabilities` to display its management menu.

Web calls use the existing BFF session and CSRF protection at `/bff/db_management/admin/website-access`. Native API calls use the existing OIDC Bearer authentication. Never store an API bearer token in the browser.

For a deployment at `app.example.org`, the native Flutter app uses `https://app.example.org/hazard/api/db_management/admin/website-access` as the base URL and sends `Authorization: Bearer <access_token>` with every request. GET on the base URL returns a JSON **array** containing exactly three entries, `main`, `ai` and `wiki`. Each contains `site_key`, `hostname`, `restricted`, `version`, `updated_at` and `rules`. Parse the array directly, without a `{ "policies": [...] }` wrapper. `GET /context` returns `{ "client_ip": "...", "can_manage_website_access": true }`. `GET /gateway` returns `online`, `checked_at` and `versions`. If the API returns 200 but the screen reports no policies, record the status code and JSON shape before checking the app's response model.

| Method | API path relative to `/admin/website-access` | Purpose |
| --- | --- | --- |
| GET | Empty path | List the three websites with their rules and versions |
| GET | `/context` | Show the source IP observed by the service |
| GET | `/gateway` | Check the Nginx synchronisation service and the applied versions |
| PUT | `/{site_key}` | Enable or disable restrictions |
| POST | `/{site_key}/rules` | Add a named IP address or network |
| PUT | `/{site_key}/rules/{uuid}` | Edit the address, name or enabled status |
| DELETE | `/{site_key}/rules/{uuid}?expected_version=N` | Delete a rule |
| GET | `/audit?site_key=ai&limit=50` | View changes, up to 200 entries |

Supply the complete payload when creating or editing a rule.

```json
{"network":"192.0.2.10","label":"Company exit / VPN","enabled":true,"expected_version":0}
```

Rule responses normalise single IP addresses to `/32` or `/128`. Each operation returns the full website policy and new `version`. Supply it as `expected_version` on the next change. On a 409 version conflict, fetch the policy again. Enable restrictions with `{"restricted":true,"expected_version":1}`. New policies are unrestricted. Adding an IP does not automatically enable restrictions. Each website supports up to 256 rules.

If enabling, changing, disabling or deleting a rule would leave the current administrator's source IP outside the effective allowlist, the transaction rolls back with `website_access_would_lock_you_out`. Empty allowlists and `/0` are prohibited. To allow all traffic, disable restrictions. All changes and their before/after audit entries commit together. The source IP comes from the ASGI client. Forms and arbitrary forwarding headers cannot specify it.

The lockout check runs when saving a policy and does not add administrators to an allowlist or exempt them. Once restrictions are enabled and applied at the gateway, every role follows the same rules. Even a signed-in global administrator receives 403 on new HTTP requests from an unlisted network.

## How the gateway enforces policies

Using the application account, `snapshot.py` exports all policies in a PostgreSQL READ ONLY, REPEATABLE READ transaction. The independent root synchronisation service accepts only the three known websites, booleans and numeric CIDRs. It never writes names or other database text into Nginx configuration. It updates `geo` allowlists every five seconds, reloading only after `nginx -t` passes and retaining the last successful configuration on failure. The UI distinguishes a successful API save from application at Nginx. It shows ‘Applied’ only when `online=true` and `versions[site_key]` in `/gateway` matches the policy's `version`. Nginx continues to enforce the last successful rules if the database or synchronisation service goes offline.

The HTTPS server block checks `if ($vn_website_<site>_allowed = 0) { return 403; }`, covering static files, API calls, WebSocket handshakes and routes with an existing media `auth_request`. HTTP serves only ACME challenges and HTTPS redirects. Rule changes affect new HTTP requests and handshakes. Established WebSocket and stream connections remain open.

Nginx uses the actual `$remote_addr`. The application trusts only the local gateway proxy. Before introducing a CDN or another reverse proxy, configure real-IP rules restricted to trusted sources. Never trust a client's `X-Forwarded-For` directly. The gateway sees the user's public IP for traffic outside the VPN exit route. Traffic sent through the company exit node arrives with the company's public IP.

All three sites still enforce their existing OIDC login and business permissions. AI and Wiki logins depend on Keycloak on the main site. Before restricting main, make sure everyone and every backend service that needs to log in can reach it through an allowed network. Test IPv4 and IPv6 separately, as the rules match them independently.

## Deployment

1. Run `deploy/website_access/migration.sql` against the development database first. The API does not run DDL on startup. New databases also need this migration to create the catalogue.
2. Update and restart the management API. Production and development each use their own database.
3. Run `python -m deploy.website_access.package <output-directory>` from the repository root to create a credential-free deployment bundle. Transfer it to `~/website-access-management/bundle/` on the gateway host. Run `python3 ~/website-access-management/bundle/deploy.py --check`, then `sudo python3 ~/website-access-management/bundle/deploy.py`. Deployment builds on the existing phase5 production gateway configuration. It backs up the runtime code, Nginx configuration and new feature configuration before installing. All existing policies must initially be unrestricted. After activating the gateway, administrators can add allowlists and enable restrictions through the API.
4. Local development can set the three development hostnames explicitly with `WEBSITE_ACCESS_HOSTS_JSON`. The synchronisation service and management API must use exactly the same hosts. Never apply development policies to the production gateway.

The synchronisation service writes status to `/run/website-access/status.json` by default. The management API reads it. To lift restrictions in an emergency, stop `website-access-agent.service` from the gateway host's terminal, set all three geo blocks in `/etc/nginx/conf.d/website-access.conf` to `default 1;`, remove their CIDR lines, run `nginx -t` and reload. Correct the database policies before restarting the synchronisation service.

## Rust management API and synchronisation service

Enabling `account_management` in the Rust API reuses the management routes, policy versions, audit records and `/gateway` status format. When local Nginx serves the management API, overwrite `X-Real-IP` with `$remote_addr` in the forwarding location and set `RUST_API_WEBSITE_ACCESS_TRUST_PROXY=true` for the Rust API. Direct development connections use the TCP peer address by default.

The Rust release includes `construction-website-access`, which accepts the existing synchronisation configuration fields `hosts`, `reader`, `output`, `last_good` and `status`. Configure `reader` to run `construction-website-access snapshot` as the application account. Supply `RUST_API_DATABASE_URL`, the actual `RUST_API_DATABASE_SCHEMA` and the website hosts. Check the generated geo configuration with `construction-website-access render --config <config> --snapshot <snapshot>`, then have the service run `construction-website-access sync --config <config>`. It starts from the existing `last_good` file and records the applied versions once the Nginx syntax check and reload succeed.
