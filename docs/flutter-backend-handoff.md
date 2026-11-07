# Flutter 與新版後端整合交接

核對日期：2026-11-07。對象：Flutter Web、Android、iOS 工程師與聯調人員。

本文件依目前後端工作目錄及本機 Flutter 原始碼核對；後端仍有未提交修改，
不能以目前 HEAD 代表整批改版，也不代表正式機已經套用。交付前請填妥下表，
並以同一個 release 的 OpenAPI、schema 與測試案例作為驗收依據。

| 交付資訊 | 需要填寫的內容 |
| --- | --- |
| 後端版本 | 包含本批修改的 commit SHA／release tag |
| Flutter 版本 | 對應的 commit SHA、App 版本及 Web build revision |
| 聯調環境 | 實際 Web origin、Native Registry URL、OIDC 設定 |
| API 契約 | 該後端版本各服務的 OpenAPI JSON；BFF proxy 路徑另依本文對照 |
| 測試資料 | 一般使用者、群組管理員、全域管理員，以及可用工地、鏡頭、模型 |
| 發布安排 | 前後端相容組合、上線順序與可回復版本 |

本文 JSON、ID、URL 與 token 都是示例，不能當成可直接呼叫的正式設定。
改版中的 worker、交易、快取、工作池及函式拆分由後端管理；Flutter 對接的是
以下 HTTP 契約，不需要複製後端內部參數或狀態。

## 1. Flutter 修改清單與優先度

| 優先度 | 工作 | 完成條件 |
| --- | --- | --- |
| P0 | 替換串流列表舊路由 | 不再呼叫 `/list_stream_configs`，改用 `/sites/{site_id}/stream-config` |
| P0 | 移除舊信箱驗證流程 | 不再呼叫 `/auth/verify-email`、`/auth/resend-verification`；帳號驗證交由 Keycloak |
| P0 | 同步 Web 播放 base path | 預設值、範例、所有 build 設定使用 `/bff/db_management/api/playback` |
| P0 | 遷移 Native 登入與 refresh | 使用 Keycloak PKCE 與 OIDC refresh，不再送 management `/login`、`/refresh` |
| P0 | 統一認證與錯誤分類 | Web 使用 BFF Cookie／CSRF；暫時錯誤不清登入、不無限重送修改操作 |
| P1 | 依模型目錄解析偵測 | 移除固定類別編號對照，使用模型版本及 `classes`／`code` |
| P1 | 共用即時與歷史 Overlay DTO | 使用正規化座標及後端 `object_id`，保留原始回饋對應 |
| P1 | 完整播放生命週期 | 正確建立、續租、釋放；續租成功不重建播放器 |
| P1 | 處理帳號刪除進行中 | 識別 `account_deletion_pending`，顯示處理中並更新列表 |
| P2 | 新增網站存取管理 UI | 若本次要提供此功能，依 `/admin/capabilities` 顯示管理入口 |

目前檢視到的 Flutter 待調整位置：

- `lib/services/management_api_service.dart`：串流列表、舊信箱驗證與重寄。
- `lib/config/bff_config.dart`、環境範例：播放預設路徑仍為 `/bff/playback`。
  本機 dev build 設定已覆寫正確路徑，但其他 build 不能依賴這個偶然條件。
- `lib/services/detection_api_service.dart`、`lib/pages/hazard_detection/detection_page.dart`
  及 `lib/widgets/detection_painter.dart`：目前仍以固定類別編號解讀結果。
- `lib/pages/managements/user_management_page.dart`：刪除中的操作提示與結果更新。
- `lib/services/auth_failure_policy.dart`、`lib/providers/unified_auth_provider.dart`
  已有共用認證與 session 維護邏輯；其他頁面應整合使用，避免另外用錯誤字串判斷。
- Native 的 `ManagementAPIService.refreshToken()` 目前仍 POST management `/refresh`，
  需要替換成 OIDC SDK／受信任 Keycloak token endpoint 的 refresh 流程。
  Web session 維護已有基礎，不代表 Native 的登入、續期已完成新版整合。

## 2. Web 與 Native 的入口

| 功能 | Flutter Web | Native |
| --- | --- | --- |
| 管理服務 | 同源 `/bff/db_management` | 已驗證部署的 management base；本部署為 `/hazard/api/db_management` |
| 偵測服務 | 同源 `/bff/detect`，`detection` 別名亦相容 | 已驗證部署的 detection base；本部署為 `/hazard/api/detect` |
| 播放控制 | `/bff/db_management/api/playback` | management base + `/api/playback` |
| 違規資料 | 同源 `/bff/violations` | 已驗證部署的 violations base |
| 登入憑證 | 瀏覽器自動攜帶 HttpOnly Cookie | OIDC access token，使用 `Authorization: Bearer ...` |

Web 不送自行取得的 Bearer token，不讀取或保存 BFF 的 access／refresh token。
同源請求必須保留 Cookie；POST、PUT、PATCH、DELETE 需要 CSRF header。
一般 JSON DTO 應容忍新增欄位，並驗證實際使用欄位的型別；Registry 簽章文件則
依其版本化契約驗證，不能任意改動簽署內容。

### Web 登入、還原與登出

1. 頂層導向 `GET /bff/auth/oidc/login?return_to=%2Fviolations`。
   `return_to` 使用 App 內路徑；Keycloak 處理帳密、驗證與 MFA。
2. 回到 App 後呼叫 `GET /bff/auth/session` 還原使用者及功能權限。
3. 呼叫 `GET /bff/auth/csrf`，將 `csrf_token` 放入後續修改請求的
   `X-CSRF-Token` header。瀏覽器提供正常同源 `Origin`。
4. 登出先呼叫 `POST /bff/auth/logout`，再頂層導向回應中的
   `global_logout_url`，完成 Keycloak SSO 登出；不要用 background fetch 跟隨它。
5. 帳號中心及個人密碼設定由 `GET /bff/auth/account` 開啟。

Web 的 Google／Apple 品牌按鈕可使用同一登入入口加上 `idp_hint=google` 或
`idp_hint=apple`；不再呼叫舊的 BFF social-token 交換端點。

Session 回應示例：

```json
{
  "authenticated": true,
  "user": {
    "id": 123,
    "username": "example_operator",
    "display_name": "示例使用者",
    "role": "admin",
    "group_id": 5,
    "status": "active"
  },
  "feature_names": []
}
```

目標政策為閒置 7 天、單次線上 SSO 最多 90 天、access token 5 分鐘並自動更新。
實際可用性由伺服器回應決定，停用／撤銷仍會使登入失效。Web 不自行計算或要求
JWT；Native 依 token 到期與正常 refresh 流程處理。App 可見且持續輪詢 session
也會形成活動，不能把「7 天閒置」解讀成固定在登入後第 7 天登出。

### Native 初始化

先兌換公司啟用碼，再取得並驗證 Registry 文件：

```text
POST /hazard/api/deployment-registry/v1/enrollments/exchange
GET  /hazard/api/deployment-registry/v1/deployments/{deployment_id}
```

兌換只回傳 `deployment_id`。Registry 文件含 `schema_version`、`deployment_id`、
`tenant_id`、`api_base_url`、`config_revision`、`issued_at`、`expires_at`、`key_id`、
`signature` 九個欄位。驗證 HTTPS、允許的導向、大小、期限及 Ed25519 簽章；
不能只因中介層缺少 `Cache-Control: no-store` 就拒絕其他驗證均通過的文件。
Registry 不簽發登入 token，也不是 Web BFF 的登入入口。

Native 登入使用受信任部署設定中的 Keycloak Authorization Code + S256 PKCE；
以 `GET /me` 取得本機業務身分與權限。不要回頭使用已移除的本機帳密／refresh API。
refresh 由 OIDC SDK 向該受信任 issuer 的 token endpoint 處理，更新安全儲存中的
token pair；不能只保留新 access token 而丟掉回應中的新 refresh token。
原生 Google／Apple SDK 如要保留，另依既有 Native Social Exchange 契約整合，
不能直接把第三方 token 當成業務 API 的 access token。

## 3. 路由遷移對照

以下管理路徑都相對於 Web `/bff/db_management` 或 Native management base。

| 舊呼叫 | 新呼叫／處理方式 | 相容狀態 |
| --- | --- | --- |
| `GET /list_stream_configs?site_id=123` | `GET /sites/123/stream-config` | 舊路由已移除；新路由仍回傳陣列 |
| `POST /auth/verify-email` | Keycloak 驗證流程 | 舊路由已移除 |
| `POST /auth/resend-verification` | Keycloak 帳號／登入流程 | 舊路由已移除 |
| management `POST /login`、`POST /refresh` | Native Keycloak PKCE／OIDC token refresh | 本機簽發路由已移除 |
| management `/auth/google`、`/auth/apple` | Native Social Exchange；Web 改用 Keycloak provider hint | 舊的直接換發業務 JWT 流程已移除 |
| Web `/bff/playback/*` | Web `/bff/db_management/api/playback/*` | 不存在獨立 playback BFF service |
| Detection `POST /detect` | 可相容使用；新整合採 `POST /v2/detect` | 舊介面保留六欄陣列，v2 回傳具名 Overlay envelope |

Web build 設定片段，請合併至既有環境設定，保留其他必要設定：

```json
{
  "BFF_BASE_PATH": "/bff",
  "BFF_PLAYBACK_PATH": "/bff/db_management/api/playback"
}
```

## 4. 模型選擇與圖片偵測

先呼叫 `GET /bff/detect/models`，不要把預設模型或類別編號寫死。

```json
{
  "default_model_id": "example-detector",
  "models": [
    {
      "id": "example-detector",
      "display_name": "示例模型",
      "version": "example-v1",
      "classes": [
        {"id": 42, "code": "helmet", "display_name": "安全帽", "color": "#457B9D"}
      ]
    }
  ]
}
```

處理 `default_model_id: null` 及空清單；沒有可用模型時停用送出操作並顯示原因。
選擇後保留該項的 `id`、`version` 與 class metadata。新請求使用 multipart：

```text
POST /bff/detect/v2/detect
X-CSRF-Token: <目前 csrf_token>

image: <圖片 bytes>
model: <選定的 models[i].id>
model_version: <同一項的 models[i].version>
view: objects
compute_regions: false
```

若要顯示錐筒／電桿區域，改送 `view=regions`、`compute_regions=true`。
`view=regions` 搭配 `compute_regions=false` 會得到 422。

v2 回應示例（節錄；正式回應另含樣式與顯示控制欄位）：

```json
{
  "overlay_schema_version": 1,
  "model_id": "example-detector",
  "model_version": "example-v1",
  "view": "objects",
  "image_size": {"width": 1920, "height": 1080},
  "overlay_coordinates": {
    "space": "normalized",
    "bbox_format": "xywh",
    "polygon_format": "xy",
    "origin": "top_left",
    "reference": "original_image"
  },
  "class_metadata": [
    {"id": 42, "code": "helmet", "display_name": "安全帽", "color": "#457B9D"}
  ],
  "overlay_objects": [
    {
      "object_id": "det_1",
      "code": "helmet",
      "display_name": "安全帽",
      "color": "#457B9D",
      "confidence": 0.97,
      "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
    }
  ],
  "overlay_regions": [],
  "region_status": {"cone": "not_computed", "pole": "not_computed"}
}
```

繪圖與回饋規則：

- bbox 是原圖上的正規化 `x,y,w,h`，不是像素 `x1,y1,x2,y2`；依實際圖片顯示範圍
  換算，計入縮放與留白。舊 `/detect` 六欄陣列仍是像素端點格式，兩種 DTO 分開解析。
- 類別依 `code` 對應本地化，沒有翻譯時使用 `display_name`；顏色與顯示開關使用回應值。
  不再假設 0 是安全帽、2 是未戴安全帽。
- 即時與歷史圖片共用 Overlay envelope；歷史 `image_size` 或 `overlay_objects`
  可能為 `null`，無資料時仍需能顯示紀錄。
- `object_id` 在該份結果／紀錄內識別物件。即使只有 `det_1`，也不能改編為 `det_0`；
  不把它當成跨圖片、跨鏡頭的全域 ID。
- 回饋的 `target_detection_id` 原樣送回後端。不要在過濾無效框後自行用陣列索引產生 ID。
- `warnings` 是供本地化使用的結構化資料；不要解析顯示用 `warning_text` 來判斷類別。
- 使用回傳的證據／縮圖 URL，不依工地名稱、日期或固定 `.png` 副檔名自行拼路徑。

## 5. 播放生命週期

以下路徑相對於 Web 或 Native 各自的 playback base：

| 方法與路徑 | 用途 |
| --- | --- |
| `POST /sessions` | 建立單鏡頭 detail 播放 |
| `POST /walls` | 建立多鏡頭 preview 牆 |
| `POST /sessions/renew` | 續租既有公開播放 ID |
| `DELETE /sessions/{id}` | 釋放播放；成功 204，沒有 JSON body |

單鏡頭 request：

```json
{"site": "Example Site", "camera": "Cam 1", "profile": "overlay", "language": "zh-TW", "transport": "hls"}
```

多鏡頭 request：

```json
{"site": "Example Site", "cameras": ["Cam 1", "Cam 2"], "profile": "overlay", "language": "zh-TW", "transport": "hls"}
```

公開 management facade 的明確 `cameras` 清單最多 **16 個不重複鏡頭**，
不能傳 wildcard。省略 `cameras` 使用站點預設選擇，數量另受上游額度與權限限制；
UI 依回傳的 `items`、`max_streams` 顯示。不要把內部 streaming 服務的較大限制
當成明確清單可超過 16 個的依據，也不要把 site、camera 字串欄位擅自改成數字 ID。

`profile=overlay` 使用後端繪製的辨識畫面，`clean` 關閉畫面 Overlay。
detail／preview 由入口決定，前端不傳自訂 FPS；來源 FPS 維持原設定。
語言選項依 streaming 的 `GET /overlay-languages` 載入，不自行假設所有語言皆已啟用。

單鏡頭以回應的 `id`、`hls_url`、`expires_in` 管理生命週期；多鏡頭使用同一個公開
`id` 及 `items[*].preview_hls_url`。這些 URL 已包含播放所需的短效 `mt`，保留所有
query 參數並交給播放器，不再先呼叫舊的 media-session API。

續租 request 與回應示例：

```json
{"id": "example-public-media-session"}
```

```json
{
  "id": "example-public-media-session",
  "mode": "single",
  "expires_in": 600,
  "renew_endpoint": "/hazard/api/db_management/api/playback/sessions/renew",
  "renewed": true,
  "hls_urls_changed": false
}
```

`600` 僅為示例；實際依回應的期限提早續租。續租只延長原能力，URL 不變，
不可因此重建播放器。Web 即使收到 Native 形式的 `renew_endpoint`，仍維持走
Web 的 BFF playback base。切換工地、關閉鏡頭牆或離開頁面時釋放資源。
公開 media `id` 與 `items[*].session_id` 是不同層級，續租／釋放使用前者。

鏡頭識別優先保留回應的穩定 `stream_id`，不要用可能重複的來源 URL 當作索引。
Metadata SSE 是最新狀態快照，不保證每幀一筆，不作為來源 FPS 的量測依據；
訂閱隨頁面取消並清理，重新連線需有退避。

## 6. 帳號管理、權限與推播

管理 UI 先取管理服務的 `GET /admin/capabilities`，依 `can_create_users`、
`can_delete_users`、`can_manage_groups`、`can_manage_website_access` 等能力顯示操作。
後端仍逐次驗權；前端不能只憑登入快照中的 role 自行放行管理功能。

刪除 request 保持原介面：

```text
DELETE /bff/db_management/delete_user
X-CSRF-Token: <目前 csrf_token>
Content-Type: application/json
```

```json
{"user_id": 123}
```

成功才回 200 與確認訊息。若回 503：

```json
{"detail": {"code": "account_deletion_pending"}}
```

代表本機已記錄待刪除，後端會繼續補同步，不能顯示「刪除成功」或假設已完整回復。
顯示「刪除處理中」，阻止重複點擊並重新查詢列表／提供稍後確認。此介面目前沒有
公開 operation-status 查詢端點或承諾 202；不要自行依想像新增呼叫。
`account_reconciliation_required` 顯示暫時無法完成並交由管理維護處理。

Web 推播在登入與取得 CSRF 後呼叫 `PUT /bff/fcm/devices`：

```json
{"device_token": "example-fcm-token", "device_lang": "zh-TW", "platform": "web"}
```

不要傳 owner/user ID，擁有者由登入身分決定；換帳號後依目前身分重新註冊。
FCM token 所有權、版本、Redis 索引及通知去重由後端管理，前端不傳這些內部欄位。
Native 使用既有 FCM resource API 與 Bearer 認證，保持平台欄位正確。

## 7. 統一錯誤處理

錯誤 body 可能是 `{"detail":"..."}`、`{"detail":{"code":"..."}}` 或 FastAPI
422 的 validation array。先讀取型別，再取 code；不要以訊息包含 `invalid`
就刷新 token，也不要假設所有 `detail` 都是字串。

| HTTP／code | Flutter 行為 |
| --- | --- |
| 401、已確認登入失效 | Web 重新驗證 BFF session，Native 依 refresh 流程處理；確認永久失效後才清登入 |
| 401 `expired_media_session` | 保留有效登入，重新協商播放資源；不要直接登出 |
| 403 一般權限不足 | 顯示無權限，不刷新 token、不清登入 |
| 403 `invalid_csrf_token` | 重新取 CSRF；確認此拒絕尚未執行操作後最多重試一次，不能泛化重送所有修改請求 |
| 409 `MODEL_UNAVAILABLE` | 重新取得模型目錄並請使用者重新選擇，不反覆提交同一舊版本 |
| 409／503 `account_deletion_pending` | 顯示處理中並更新資料，不當作一般可無限重送的 503 |
| 400／413／422 輸入錯誤 | 顯示格式、尺寸或驗證原因，不自動刷新 token／重試同一內容 |
| 429 `overlay_language_limit_reached` | 調整語言或釋放既有需求後再試，不把語言額度視為單純網路重試 |
| 429 | 保留登入，依 `Retry-After` 退避；沒有 header 時採有限次數退避 |
| 503 `INFERENCE_BUSY`、`INFERENCE_MEMORY_BUSY`、`OVERLAY_BUSY`、`session_refresh_busy` | 保留登入，以有限重試／稍後再試處理 |
| 其他 5xx、斷線、逾時 | 保留登入；修改請求可能已提交，先查結果，不能一律自動重送 |

`Retry-After` 可能是秒數或 HTTP date。每個操作應有總等待期限與重試上限，
使用者離開頁面時取消排程；原生推論已開始後，後端仍會持有工作額度直到完成。

## 8. 驗收與交付

### 必測情境

- [ ] Web 登入、重新整理、關閉再開還原；登出完成 BFF 與 Keycloak 的導向流程。
- [ ] Native 啟用碼、Registry 驗簽、PKCE、token refresh 及 `/me` 權限還原。
- [ ] 模擬 token 更新的 429／503／斷線仍保留登入，真正的 session 失效才要求登入。
- [ ] 新串流列表正常，網路請求不含三個已移除路由。
- [ ] 模型清單為空、沒有預設、版本更新 409、不同模型重用不同類別編號，皆正確處理。
- [ ] v2 與歷史 Overlay 在不同長寬比下框線正確；`det_1` 回饋 ID 不被重新編號。
- [ ] 單鏡頭、明確清單 16 支、17 支的 422、站點預設清單、重複鏡頭及無權限。
- [ ] `clean/overlay`、語言切換、續租不重建播放器、離頁釋放及過期後重新建立。
- [ ] 刪除立即成功與待補同步兩條路徑；503 時不誤報成功或重複送出。
- [ ] 同一裝置切換帳號後重新註冊推播；舊帳號不再收到通知。
- [ ] 一般使用者、群組管理員、全域管理員與工地權限變更的 UI／API 行為一致。

### 發布產物

Flutter 交付 source commit、Web build revision／App 版本、採用的後端版本、
建置設定清單、artifact checksum 與上述驗收結果。Web artifact 包含 `index.html`、
`assets/`、必要啟動檔及版本資訊；從獨立 Flutter repo 建置，不手改 `main.dart.js`。

路由移除前需要相容的前端版本。先在指定聯調環境確認版本組合，再協調上線順序；
本文與後端單元測試不是正式環境 Web／Android／iOS 端到端驗收的替代品。

### 契約來源

- [前端產物與部署契約](../frontend/README.md)
- [Web BFF、Cookie、CSRF 與登出](../examples/bff/README-zh-tw.md)
- [管理服務與 Native／Registry](../examples/db_management/README-zh-tw.md)
- [模型目錄及偵測 API](../examples/YOLO_server_api/DETECTION_MODEL_API.md)
- [公開播放 request schema（包含 16 鏡頭限制）](../examples/db_management/schemas/playback.py)
- [共用 Overlay schema](../examples/shared/overlay_schemas.py)
- [SSO 政策單一來源](../src/settings/session_policy.json)

各服務 FastAPI 的 `/openapi.json` 描述其原生路由；對外前綴須依本文件及實際部署
映射。BFF 通用 proxy、OIDC redirect 與部分媒體路徑未列入 OpenAPI，不能只看
Swagger 就假設不存在或忽略 Cookie／CSRF／導向規則。
