# Nginx：BFF、Native OAuth 與 Media Session

本機目前實際載入 `/etc/nginx/sites-enabled/default`；它不是
`sites-available/default` 的 symlink。以下設定都必須放在
`changdar-server.mooo.com` 的 HTTPS `server {}` 內。

## 1. 啟動並加入 Web BFF

BFF module 與 db_management 共用 port `8005`，啟動同一個 process：

```bash
uvicorn examples.db_management.app:app \
  --host 127.0.0.1 \
  --port 8005 \
  --workers 4
```

在 HTTPS `server {}` 內加入：

```nginx
include /home/changdar/Documents/Construction-Hazard-Detection/deploy/bff/nginx.bff.conf;
```

Web 登入/session 仍使用 `/bff/*`，Nginx 直接轉到 port `8005`。Flutter
Web/iOS/Android 直播統一使用既有 `/hazard/api/db_management/` base path 下的
`/api/playback/*` facade；HLS URL 由後端加上短效 `mt` media token。Native
OAuth、`/me` 與 playback facade 都走同一條既有 db_management Nginx route。

## 2. 取代舊 HLS media auth

從 active server block 刪除原本三個 location：

```text
location = /hazard/api/media-auth
location ^~ /hazard/media/webrtc/
location ^~ /hazard/media/
```

在相同位置改成：

```nginx
include /home/changdar/Documents/Construction-Hazard-Detection/examples/streaming_web/nginx.hazard-media.conf;
```

同時刪除 server block 外的舊 media auth cache/query-token 設定：

```text
proxy_cache_path /tmp/hazard_media_auth_cache ...
map $arg_token $hazard_media_auth_cookie { ... }
map $request_uri $hazard_media_auth_cache_uri { ... }
```

獨立 media session 必須即時撤銷，因此不可保留 `proxy_cache`；也不可再把
`?token=` 轉成 `hazard_access_token` cookie。

## 3. 驗證與 reload

```bash
sudo nginx -t
sudo systemctl reload nginx
sudo systemctl status nginx --no-pager
```

未登入 smoke test：

```bash
curl -i https://changdar-server.mooo.com/bff/auth/session
curl -i https://changdar-server.mooo.com/hazard/api/db_management/me
curl -i https://changdar-server.mooo.com/hazard/api/media-auth
```

預期前兩個登入相關 endpoint 回 `401`；`media-auth` 因為是 Nginx `internal`
location，外部直接存取應回 `404`。
