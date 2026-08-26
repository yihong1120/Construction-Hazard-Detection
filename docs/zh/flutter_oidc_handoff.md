# Flutter OIDC 整合規格

Visionnaire 現在由 Keycloak 統一管理登入、密碼、真人驗證與後續 MFA。Flutter 不得再直接呼叫 Visionnaire 的帳密登入、Google／Apple 登入、refresh-token 或 hCaptcha API。

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
2. Keycloak 會顯示帳密登入頁，正確帳密後再顯示 hCaptcha。Flutter Web 不得嵌入 hCaptcha widget、site key 或傳送 `hcaptcha_token`。
3. 回到 App 後呼叫 `GET /bff/auth/session`；若回傳 `401`，重新導向 `/login`。
4. Web API 一律使用 `/bff/{service}/...`，並以 `credentials: include` 傳送 HttpOnly session cookie。例如違規列表走 `/bff/violations/...`。
5. 每個變更狀態的 BFF request 先呼叫 `GET /bff/auth/csrf`，再傳送 `X-CSRF-Token`。不要加 `Authorization` header。
6. 使用者改密碼或設定 MFA 時，導向 `GET /bff/auth/account`；這會開啟 Keycloak Account Console。Visionnaire 已不再提供密碼修改頁。
7. App 登出：以 CSRF token 呼叫 `POST /bff/auth/logout`，清掉本 App 的 BFF session。若產品要求全域登出，另依 discovery document 的 `end_session_endpoint` 執行 Keycloak 登出。

必須移除的 Web 程式碼：帳號密碼表單送往 `/login` 或 `/bff/auth/login`、hCaptcha Flutter／JavaScript 元件、`X-HCaptcha-Bypass-Key`、將 OAuth token 存入 LocalStorage／IndexedDB，以及舊 Google、Apple、refresh-token 登入入口。

## Flutter iOS／Android

Native 使用系統瀏覽器與 Authorization Code + PKCE。不要使用 WebView，也不要將密碼交給 App；因此 Keycloak 頁面中的 hCaptcha、密碼政策與 MFA 對 iOS、Android 和 Web 一致。

建議採用 [`flutter_appauth`](https://pub.dev/packages/flutter_appauth) 與 [`flutter_secure_storage`](https://pub.dev/packages/flutter_secure_storage)：

```dart
final result = await appAuth.authorizeAndExchangeCode(
  AuthorizationTokenRequest(
    'visionnaire-mobile',
    'com.changdar.visionnaire:/oauthredirect',
    issuer: 'https://changdar-server.mooo.com/keycloak/realms/visionnaire',
    scopes: const ['openid', 'profile', 'email', 'offline_access'],
  ),
);
```

- access token 只保留在記憶體；每個 API request 加上 `Authorization: Bearer <access-token>`。
- 若要維持登入，把 refresh token 放在 iOS Keychain／Android Keystore（例如 `flutter_secure_storage`），絕不可存 SharedPreferences、檔案或 log。
- access token 到期時以 refresh token 更新；更新失敗或 API 回傳 `401` 時清除本機 token 並重新登入。
- Native 直接呼叫 `/hazard/api/...`，例如 `POST /hazard/api/db_management/api/playback/sessions`；不要使用 Web BFF cookie 路徑。
- Android Manifest 與 iOS URL Types 都需註冊 scheme `com.changdar.visionnaire`，讓 `com.changdar.visionnaire:/oauthredirect` 回到 App。
- Native 登出需清除 secure storage，並用 discovery document 的 `end_session_endpoint` 和 `id_token_hint` 執行 Keycloak 全域登出。

## 真人驗證、授權與驗收

登入次序固定為：

```text
系統瀏覽器 → Keycloak 帳密 → hCaptcha → OIDC callback → Visionnaire API
```

hCaptcha token 只能由 Keycloak Provider 在伺服器端驗證，並會驗證 hCaptcha 的成功狀態、site key 與 `changdar-server.mooo.com` hostname。Flutter 不需要、也不應看見 hCaptcha secret。

若 API 回傳 `401`，Native 先嘗試 token refresh；仍失敗則回登入。若登入後回傳 `OIDC identity is not linked to a local user`，應顯示「此帳號尚未獲得 Visionnaire 存取權」並通知管理員；不得以 username 或 email 在前端自動綁定。

- Web、iOS、Android 都從同一個 Keycloak 頁面登入，hCaptcha 只出現一次。
- Flutter Web 的開發工具與 LocalStorage 中沒有 OAuth access/refresh token。
- Native client 沒有 client secret，登入請求有 `code_challenge_method=S256`。
- API access token 的 `iss` 為上述 issuer，`aud` 包含 `visionnaire-api`。
- 在 Keycloak Account Console 改密碼後，三個平台都以新密碼登入。
- 舊 Visionnaire 密碼登入 API 與 `/bff/auth/login` 帳密 POST 一律收到 `409 login_managed_by_identity_provider`。
