🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Nginx for the BFF, native OAuth and media sessions

Edit the HTTPS `server {}` block in the configuration Nginx actually loads. The examples use `app.example.org` and a checkout in `/srv/Construction-Hazard-Detection`. Substitute your hostname and checkout path.

## 1. Start and connect the web BFF

The BFF and db_management share a process on port `8005`. Start it with the following command.

```bash
uvicorn examples.db_management.app:app \
  --host 127.0.0.1 \
  --port 8005 \
  --workers 4 \
  --timeout-graceful-shutdown 10
```

Add the following include to the HTTPS `server {}` block.

```nginx
include /srv/Construction-Hazard-Detection/deploy/bff/nginx.bff.conf;
```

Web login and sessions use `/bff/*`, which Nginx forwards directly to port `8005`. Flutter Web sends playback control requests to `/bff/db_management/api/playback/*` so the BFF handles the HttpOnly session and CSRF protection. Native clients use the existing `/hazard/api/db_management/api/playback/*` facade. The backend adds a short-lived `mt` media token to HLS URLs. Native OAuth, `/me` and native playback all use the existing db_management Nginx route.

## 2. Replace legacy HLS media authentication

Remove the following three locations from the active server block.

```text
location = /hazard/api/media-auth
location ^~ /hazard/media/webrtc/
location ^~ /hazard/media/
```

Add the following include in their place.

```nginx
include /srv/Construction-Hazard-Detection/examples/streaming_web/nginx.hazard-media.conf;
```

Also remove the legacy media authentication cache and query-token configuration outside the server block.

```text
proxy_cache_path /tmp/hazard_media_auth_cache ...
map $arg_token $hazard_media_auth_cookie { ... }
map $request_uri $hazard_media_auth_cache_uri { ... }
```

Media sessions require immediate revocation. Remove `proxy_cache` and the rule that converts `?token=` into a `hazard_access_token` cookie.

## 3. Validate and reload Nginx

```bash
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl status nginx --no-pager
```

Run the following smoke tests while signed out.

```bash
curl -i https://app.example.org/bff/auth/session
curl -i https://app.example.org/hazard/api/db_management/me
curl -i https://app.example.org/hazard/api/media-auth
```

The first two login-related endpoints should return `401`. Direct external access to `media-auth` should return `404` because the location is marked `internal`.
