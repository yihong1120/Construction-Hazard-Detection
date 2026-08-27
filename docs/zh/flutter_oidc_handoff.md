# Flutter OIDC 整合規格

Visionnaire 現在由 Keycloak 統一管理登入、密碼、真人驗證與後續 MFA。Flutter Web
維持 Keycloak broker + BFF；Flutter iOS／Android 可使用官方 Google／Apple SDK，但只可透過
後端的「一次性原生憑證交換」回到 Keycloak 的 Authorization Code + PKCE，不能直接取得
Visionnaire JWT 或呼叫舊 `/auth/google`、`/auth/apple`。完整請交付工程師
[原生社群憑證交換規格](native_social_exchange.md)。

## 固定 OIDC 資訊

| 項目 | 值 |
| --- | --- |
| Issuer | `https://changdar-server.mooo.com/keycloak/realms/visionnaire` |
| Discovery | `https://changdar-server.mooo.com/keycloak/realms/visionnaire/.well-known/openid-configuration` |
| Native client ID | `visionnaire-mobile` |
| Native redirect URI | `com.changdar.visionnaire:/oauthredirect` |
| API audience | `visionnaire-api` |
| API public base URL | `https://changdar-server.mooo.com/hazard/api` |

Native client 是 public client：只開啟 Authorization Code + PKCE (S256)，沒有也不得加入 client secret；implicit flow 與 password/direct grant 都已停用。

## Flutter Web

Web 使用 BFF，不在瀏覽器內保存或讀取 OAuth token。

1. 未登入時直接導向 `https://changdar-server.mooo.com/login`；若要保留頁面，使用 `https://changdar-server.mooo.com/bff/auth/oidc/login?return_to=/violations`。
2. Keycloak 會顯示帳密與已啟用的 Google／Apple 按鈕。帳密登入正確後再顯示 hCaptcha；社群登入由第三方回到 Keycloak，不需也不應在 Flutter 內執行 hCaptcha。
3. 回到 App 後呼叫 `GET /bff/auth/session`；若回傳 `401`，重新導向 `/login`。
4. Web API 一律使用 `/bff/{service}/...`，並以 `credentials: include` 傳送 HttpOnly session cookie。例如違規列表走 `/bff/violations/...`。
5. 每個變更狀態的 BFF request 先呼叫 `GET /bff/auth/csrf`，再傳送 `X-CSRF-Token`。不要加 `Authorization` header。
6. 使用者改密碼或設定 MFA 時，使用**同一個頂層瀏覽器視窗**導向 `GET /bff/auth/account`；這會開啟 Keycloak Account Console。它和 Visionnaire 共用 Keycloak 的瀏覽器 SSO session，正常情況下不會再次要求帳密。Visionnaire 已不再提供密碼修改頁。
7. App 登出必須是全域登出：以 CSRF token 呼叫 `POST /bff/auth/logout`。回應為 `{ "global_logout_url": "/bff/auth/oidc/logout?state=..." }` 時，立刻用 `window.location.assign(global_logout_url)`（不可用 `fetch` 跟隨 redirect，也不可自行組 Keycloak logout URL）。此一次性 URL 會先清除 BFF cookie，再由後端以 Keycloak 自己的 HttpOnly SSO cookie 導向 RP-initiated logout，結束瀏覽器 SSO，最後回到網站根目錄；access、refresh、ID token 都不會進入 Flutter Web。回應為 `null` 僅代表遷移期的舊本機 session；照常清除前端狀態即可。

若產品需要保留 Visionnaire Web 的「使用 Google／Apple 繼續」按鈕，按鈕只能導向下列 BFF URL，不能使用 Google Sign-In、Apple Sign In SDK 或將第三方 token POST 給 Visionnaire：

```text
/bff/auth/oidc/login?return_to=/&idp_hint=google
/bff/auth/oidc/login?return_to=/&idp_hint=apple
```

後端只接受 `google`、`apple` 兩個 allow-list hint，並轉成 Keycloak 的 `kc_idp_hint`。若該 provider 尚未由 Keycloak 啟用，Flutter 不得顯示對應按鈕。

必須移除的 Web 程式碼：帳號密碼表單送往 `/login` 或 `/bff/auth/login`、hCaptcha Flutter／JavaScript 元件、`X-HCaptcha-Bypass-Key`、將 OAuth token 存入 LocalStorage／IndexedDB，以及舊 Google、Apple、refresh-token **直接**登入入口。

## Visionnaire 使用者與群組管理

Keycloak Account Console 是**個人自助**頁面，只能修改自己的密碼、MFA、個人資料與已連結
帳號；它不是 Visionnaire 的使用者管理後台。Flutter 的 `/users` 必須保留為 Visionnaire
自己的管理頁面，絕不可從 App 直接呼叫 Keycloak Admin API。Web 走 BFF；iOS／Android 使用
自己的 OIDC bearer token 直接呼叫 Visionnaire API。

進入管理頁後，Web 先呼叫：

```http
GET /bff/db_management/admin/capabilities
```

iOS／Android 呼叫等價的：

```http
GET /hazard/api/db_management/admin/capabilities
Authorization: Bearer <access token>
```

回應是後端判定的 UI 能力，例如：

```json
{
  "scope": "all_groups",
  "managed_group_id": null,
  "can_create_users": true,
  "can_reset_passwords": true,
  "can_suspend_users": true,
  "can_delete_users": true,
  "can_manage_groups": true,
  "can_manage_group_features": true,
  "can_assign_group_admins": true
}
```

不得在 Flutter 以帳號名稱或 Keycloak realm role 推斷 ChangDar 身分。前端只依此能力顯示
控制項，後端仍會在每一個請求再次授權。

| 操作人 | 可見範圍與可執行操作 |
| --- | --- |
| ChangDar（super admin） | 所有群組、使用者、工地範圍與功能權限；可建立／修改群組，並在指定群組建立或任命 `admin`。 |
| 群組 admin | 僅自己的群組與其中一般成員；可建立、修改、停用及重設該群組一般成員帳號，但不可任命、降級或修改任何 `admin`，也不可管理群組定義或功能權限。 |

管理頁不得顯示 `Read-only in this client` 這類全域唯讀提示。若能力不足，只隱藏或停用該
單一操作並說明範圍限制；例如群組 admin 不應看到「管理所有群組」或「任命群組管理員」。

使用者清單使用 `GET /bff/db_management/admin/users`（cursor 分頁）；群組清單使用
`GET /bff/db_management/list_groups`。Web 的每個變更操作都先取得 CSRF token，並用
`X-CSRF-Token` 呼叫下列 BFF 路徑；iOS／Android 將 `/bff` 替換為
`/hazard/api`、使用 bearer token，且不傳 CSRF header：

```text
POST   /bff/db_management/add_user
PUT    /bff/db_management/admin_update_password_userid
PUT    /bff/db_management/update_user_role
PUT    /bff/db_management/update_user_group
PUT    /bff/db_management/set_user_status
PUT    /bff/db_management/update_user_profile
DELETE /bff/db_management/delete_user

POST   /bff/db_management/create_group
PUT    /bff/db_management/update_group
DELETE /bff/db_management/delete_group
POST   /bff/db_management/update_group_feature
```

建立使用者時請送完整 `profile`（email、given name、family name）、初始 `password`、
`group_id` 與 `role`。新欄位 `force_password_change` 預設為 `true`；應提供管理者核取方塊，
讓初始密碼或管理員重設密碼後，使用者在 Keycloak 下次登入時必須更新。密碼只會經 HTTPS
送至 Visionnaire BFF，再由後端寫入 Keycloak；Flutter 不得保存、記錄或重送密碼。

`role: "admin"` 的選項只在 `can_assign_group_admins=true` 時顯示。所有群組、角色、工地
範圍與功能權限仍由 Visionnaire 資料庫管理；Keycloak 只管理 identity、登入、密碼、MFA 與
SSO session。

## Flutter iOS／Android

Native 的帳密／MFA 登入仍使用系統瀏覽器與 Authorization Code + PKCE。不要使用
WebView，也不要將密碼交給 App；因此 Keycloak 頁面中的 hCaptcha、密碼政策與 MFA 對
iOS、Android 和 Web 一致。

建議採用 [`flutter_appauth`](https://pub.dev/packages/flutter_appauth) 與 [`flutter_secure_storage`](https://pub.dev/packages/flutter_secure_storage)：

```dart
final result = await appAuth.authorizeAndExchangeCode(
  AuthorizationTokenRequest(
    'visionnaire-mobile',
    'com.changdar.visionnaire:/oauthredirect',
    issuer: 'https://changdar-server.mooo.com/keycloak/realms/visionnaire',
    scopes: const ['openid', 'profile', 'email'],
  ),
);
```

Native UI 的 Google／Apple 按鈕改用官方 provider SDK，但不能直接將 provider token
換成 Visionnaire token。實作順序固定為：

1. App 產生 `state` 與 PKCE S256 verifier/challenge，呼叫
   `POST /auth/native-social/exchanges` 取得 `transaction_id`、`nonce`。
2. 將 nonce 傳入 Google Sign-In 或 Sign in with Apple SDK，取得 Google `idToken` 或
   Apple `authorizationCode`（建議同時送 `identityToken`）。
3. 呼叫 `POST /auth/native-social/exchanges/complete`；以 `url_launcher` 開回傳的
   `authorization_url`。
4. Keycloak callback 回到 App 後，驗證 state，使用**同一個** PKCE verifier 交換普通
   Keycloak token。

若第 3 步回傳 HTTP `409` 且 `detail.code == 'account_link_required'`，代表供應商已驗證的
email 唯一命中既有 Visionnaire 帳號；這仍**不是**自動合併。App 將
`detail.link_transaction_id` 暫存在記憶體，立刻以正常 Keycloak Authorization Code + PKCE
重新登入，並加入 `prompt=login`、`max_age=0`。例如 `flutter_appauth` request 加上：

```dart
additionalParameters: const {'prompt': 'login', 'max_age': '0'},
```

以 callback 換得、已更新到 secure storage 的 fresh access token 呼叫：

```http
POST /hazard/api/db_management/auth/native-social/email-link-confirmations/complete
Authorization: Bearer <fresh access token>

{ "transaction_id": "<link_transaction_id>" }
```

成功後 Google／Apple `sub`、Keycloak `sub` 與 Visionnaire user 已永久連結；再次使用同一個
Google／Apple 或既有帳密會直接登入同一筆資料。若重新登入的是不同帳號、交易過期或已使用，
清除記憶體中的 transaction，回到社群登入起點；不得在 App 端以 email 選擇或合併帳號。

完整 request/response、Apple one-use code 重試規則與帳號連結實作在
[原生社群憑證交換規格](native_social_exchange.md)，工程師須逐項遵守。

- access token 只保留在記憶體；每個 API request 加上 `Authorization: Bearer <access-token>`。
- 若要維持登入，把 refresh token 和 `idToken` 放在 iOS Keychain／Android Keystore（例如 `flutter_secure_storage`），絕不可存 SharedPreferences、檔案或 log。這是**一般 online refresh token**，不得要求 `offline_access`；它會隨 Keycloak 的 online SSO policy 到期。
- access token 到期時以 refresh token 更新；更新失敗或 API 回傳 `401` 時清除本機 token 並重新登入。
- Native 直接呼叫 `/hazard/api/...`，例如 `POST /hazard/api/db_management/api/playback/sessions`；不要使用 Web BFF cookie 路徑。
- Android Manifest 與 iOS URL Types 都需註冊 scheme `com.changdar.visionnaire`，讓 `com.changdar.visionnaire:/oauthredirect` 回到 App。
- Native 登出順序固定為：停止 API request → 用系統瀏覽器開啟 discovery document 的 `end_session_endpoint`，帶 `id_token_hint`、`post_logout_redirect_uri=com.changdar.visionnaire:/oauthredirect` 與 `client_id=visionnaire-mobile` → callback 回 App 後清除 memory 與 secure storage。不得以 WebView 或手刻 Cookie 登出；若未取得 `idToken`，先清除本機資料並強制下一次完整登入。
- 顯式帳號連結、以及 `account_link_required` 的確認，都必須再做一次 Keycloak
  `prompt=login`、`max_age=0`；後端只接受 `auth_time` 不超過五分鐘的 bearer token。已驗證
  email 僅能安全地發現唯一候選帳號，不能在 App 端自動判定或合併帳號。

## 真人驗證、授權與驗收

登入次序固定為：

```text
帳密路徑：系統瀏覽器 → Keycloak 帳密 → hCaptcha → OIDC callback → Visionnaire API

Web 社群路徑：系統瀏覽器 → Keycloak Google／Apple → 第三方登入 → Keycloak → OIDC callback → Visionnaire API

Native 社群路徑：官方 SDK（nonce）→ Visionnaire one-use proof → Keycloak → OIDC callback → Visionnaire API
```

hCaptcha token 只能由 Keycloak Provider 在伺服器端驗證，並會驗證 hCaptcha 的成功狀態、site key 與 `changdar-server.mooo.com` hostname。它保護 Keycloak 本機帳密流程；Google／Apple 已各自在其登入流程驗證使用者，Flutter 不需要、也不應看見 hCaptcha 或第三方 OAuth secret。

若 API 回傳 `401`，Native 先嘗試 token refresh；仍失敗則回登入。若登入後回傳 `OIDC identity is not linked to a local user`，應顯示「此帳號尚未獲得 Visionnaire 存取權」並通知管理員；不得以 username 或 email 在前端自動綁定。

- Web、iOS、Android 的最終 token 都由同一個 Keycloak issuer 簽發；帳密 hCaptcha 只出現在 Keycloak 路徑。
- Flutter Web 的開發工具與 LocalStorage 中沒有 OAuth access/refresh token。
- Native client 沒有 client secret，登入請求有 `code_challenge_method=S256`。
- Native Google／Apple network log 不會出現 Flutter 對舊 `/auth/google`、`/auth/apple` 的請求；只允許新的 nonce-bound `/auth/native-social/*` API。
- API access token 的 `iss` 為上述 issuer，`aud` 包含 `visionnaire-api`。
- 在 Keycloak Account Console 改密碼後，三個平台都以新密碼登入；Web Account Console 與 Visionnaire 在同一個 online SSO session 期間不應重複詢問帳密。
- 舊 Visionnaire 密碼登入 API 與 `/bff/auth/login` 帳密 POST 一律收到 `409 login_managed_by_identity_provider`。
