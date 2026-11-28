🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 前端交付規範

Flutter 公開原始碼由 [visionnaire-flutter](https://github.com/yihong1120/visionnaire-flutter) 儲存庫維護。

`web/` 存放已建置的部署檔案，由 `deploy/mobile-deep-links/nginx.mobile-deep-links.conf` 提供服務。Git 會忽略這個目錄，前端變更應在原始碼儲存庫內進行。

Flutter/Web 的建置工作必須發布含版本資訊的成品，其中包含 `index.html`、`assets/` 及建置所用的修訂版本。部署時會取得成品、驗證檢查碼並解壓縮到 `frontend/web/`。移除 API 路由前，必須先發布已停止呼叫該路由的前端版本。

後端儲存庫沒有另一份前端原始碼。`frontend/web/` 是本機部署所用的檔案，Git 和 Docker 建置環境都會排除此目錄。

## 後端相容性

Flutter 開發人員可先閱讀[後端整合交接文件](../docs/flutter-backend-handoff.md)，瞭解目前工作目錄需要的資料庫遷移、網頁版與原生應用程式的身分驗證、請求範例、錯誤處理及發布驗收項目。

建置及部署的 Flutter 版本必須支援下列後端介面。

- Flutter Web 以 `/bff/detect/models` 查詢模型，以 `/bff/detect/v2/detect` 進行影像推論。BFF 的 `detect` 與 `detection` 都會連到同一個需要身分驗證的 YOLO 上游服務。
- Flutter Web 經由同源的 BFF，將播放控制請求送至 `/bff/db_management/api/playback/*`。
- 原生應用程式驗證部署設定檔簽章後，將播放控制請求送至 `/hazard/api/db_management/api/playback/*`。
- 即時標註請選用 `overlay` 並指定語言。後端會直接在影片上繪製物件框、區域和文字，Flutter 再播放回傳的 HLS 網址。
- Registry 用戶端須驗證 HTTPS、重新導向、JSON 大小及 Ed25519 簽章。若回應其餘檢查皆通過，中介服務未提供 `Cache-Control: no-store` 這一點不應導致拒絕回應。

公開路徑或 Registry 文件規格有變更時，請一起部署對應的前端成品及後端版本。

## 功能退役版本

查詢串流設定時，請依環境選擇 BFF 或經簽章的原生部署 API 前綴，呼叫 `GET /sites/{site_id}/stream-config`。舊別名 `GET /list_stream_configs?site_id=...` 已移除。這兩條路由原本共用工地／群組權限服務，回傳相同的陣列結構。

本機的 `/auth/verify-email` 和 `/auth/resend-verification` 路由也已移除。帳號驗證及電子郵件驗證畫面皆由 Keycloak 處理。既有停用帳號須由管理員另行處理，路由退役不會將電子郵件自動標記為已驗證。

目前部署的 Web 成品仍含舊的串流清單路由，並由 Git 忽略。重新載入此後端版本前，請先從 Flutter 原始碼建置並發布使用現行路由的版本。請勿直接修改 `main.dart.js`。
