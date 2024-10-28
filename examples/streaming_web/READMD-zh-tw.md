
🇬🇧 [English](./README.md) | 🇹🇼 [繁體中文](./README-zh-tw.md)

# 即時串流網頁範例

此部分提供一個即時串流網頁應用程式的實作範例，設計目的是為了實現即時的攝影機串流畫面更新。本指南提供如何使用、配置及瞭解此應用程式的特點與功能。

## 使用方法

1. **啟動伺服器：**
    ```sh
    python app.py
    ```

    或者使用 Gunicorn 啟動應用程式，並設定非同步工作執行緒：
    ```sh
    gunicorn -w 1 -k eventlet -b 127.0.0.1:8000 "examples.streaming_web.app:app"
    ```

2. **訪問應用程式：**
   打開網頁瀏覽器，並導向以下網址：
    ```sh
    http://localhost:8000
    ```

## 功能

- **即時串流**：顯示即時的攝影機畫面，每 5 秒自動更新。
- **WebSocket 整合**：使用 WebSocket 進行高效的即時通訊。
- **動態內容加載**：自動更新攝影機圖片，無需重新整理頁面。
- **響應式設計**：適應不同螢幕尺寸，提供無縫的使用者體驗。
- **可自定義的佈局**：透過 CSS 調整佈局和樣式，以符合個人需求。

## 配置和檔案概覽

此應用程式可透過以下關鍵檔案進行自訂和配置：

- **app.py**：啟動伺服器並定義路由的主要應用程式檔案。
- **routes.py**：定義網頁路由及其相應的處理器。
- **sockets.py**：管理 WebSocket 連接和事件。
- **utils.py**：包含應用程式使用的實用工具函式。
- **index.js**：處理主頁面中攝影機圖片的動態更新。
- **camera.js**：管理攝影機畫面的更新。
- **label.js**：處理 WebSocket 通訊和基於標籤的更新。
- **styles.css**：包含網頁應用程式的樣式，確保響應式和可存取的設計。

請務必根據環境需求檢查並調整這些檔案中的配置設定。

## Nginx 配置範例

若要使用 Nginx 作為此 FastAPI 應用程式的反向代理，可以參考以下關鍵配置部分。完整的範例配置檔案請參見 `config/` 目錄中的 `nginx_config_example.conf`。

1. **HTTP 重定向至 HTTPS**：將所有 HTTP 請求重定向到 HTTPS，確保安全通訊。
    ```nginx
    server {
        listen 80;
        server_name yourdomain.com;
        location / {
            return 301 https://$server_name$request_uri;
        }
    }
    ```

2. **HTTPS 配置**：啟用 SSL 憑證並代理靜態文件和 WebSocket 請求。
    ```nginx
    server {
        listen 443 ssl;
        server_name yourdomain.com;

        # SSL 憑證路徑
        ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
        ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

        # 靜態文件
        location /upload/ {
            alias /home/youruser/Documents/Construction-Hazard-Detection/static/uploads/;
            autoindex on;
            allow all;
        }

        # WebSocket 配置
        location /ws/ {
            proxy_pass http://127.0.0.1:8000;
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
            proxy_buffering off;
            # 傳遞額外的客戶端資訊標頭
        }

        # 一般 HTTP 代理
        location / {
            proxy_pass http://127.0.0.1:8000;
            # 傳遞客戶端及 SSL 狀態的標頭
        }
    }
    ```

3. **SSL 憑證設定**

   若要使用 SSL 保護伺服器，可以使用 Let's Encrypt 提供的免費 SSL 憑證。建議步驟如下：

   - **安裝 Certbot**：使用 Certbot 來自動處理 SSL 憑證的安裝和續期。
   - **取得 SSL 憑證**：使用您的網域名稱運行 Certbot 以創建 SSL 憑證：
     ```sh
     sudo certbot --nginx -d yourdomain.com
     ```
   - **設置自動續期**：Certbot 會自動處理憑證的續期，您可以新增 Cron 排程定期檢查：
     ```sh
     0 12 * * * /usr/bin/certbot renew --quiet
     ```

此配置確保 Nginx 伺服器的安全性，並自動管理 SSL 憑證。

## 其他注意事項

欲進一步自訂應用程式，請參考 `examples/streaming_web` 資料夾並根據專案需求調整檔案。此程式碼具有模組化設計，允許您更新或替換組件，以適應擴展性和維護性需求。
