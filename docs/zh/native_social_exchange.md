# 原生 Google／Apple 憑證交換與帳號連結規格

本規格讓 Flutter iOS／Android 使用官方 Google Sign-In 與 Sign in with Apple
SDK，同時仍由 Keycloak 核發最終的 OIDC access、refresh 與 ID token。它**不**啟用
Keycloak 已淘汰的 external Token Exchange v1，也不會把第三方 token、Apple code 或
Keycloak Admin token 存進 Redis。

## 登入交換

```text
Flutter ──(PKCE, state)──> Visionnaire begin
        <── transaction_id, nonce ──
Flutter ── 官方 Google / Apple SDK（nonce）──> Provider
Flutter ── provider assertion ──> Visionnaire complete
        <── Keycloak authorization URL ──
system browser ── one-use HMAC redeem ──> Keycloak
Keycloak ── normal authorization code ──> Flutter redirect URI
Flutter ── PKCE verifier ──> Keycloak token endpoint
```

1. Flutter 以安全亂數產生 `state` 與 PKCE verifier/challenge（S256）；state 必須在
   callback 做固定時間比對。
2. 呼叫 `POST /hazard/api/db_management/auth/native-social/exchanges`：

   ```json
   {
     "provider": "google",
     "client_id": "visionnaire-mobile",
     "redirect_uri": "com.changdar.visionnaire:/oauthredirect",
     "code_challenge": "<S256 challenge>",
     "code_challenge_method": "S256",
     "state": "<opaque state>"
   }
   ```

   回應的 `transaction_id`、`nonce` 有效期 90 秒，兩者都是一次性值。
3. 將回傳的 `nonce` 原樣交給官方 SDK。Google 完成後傳 `idToken`；Apple 必須傳
   `authorizationCode`，並應同時傳 `identityToken`。不要把 email、姓名或 provider
   `sub` 當作可信 input。
4. 呼叫 `POST /auth/native-social/exchanges/complete`：

   ```json
   {
     "transaction_id": "<from begin>",
     "id_token": "<Google ID token or Apple identity token>",
     "authorization_code": "<Apple authorization code only>"
   }
   ```

   Google 請只送 `id_token`；Apple 請送 `authorization_code`，`id_token` 可選。交易被
   consume 後不可重試同一份 Apple code，失敗應從 begin 重新開始。
5. 立刻用 `url_launcher` 開回傳的 `authorization_url`，不可自行改動 query string。
   Keycloak 的 custom authenticator 會以 loopback HMAC 拿一次性 proof，逐項比對
   `client_id`、`redirect_uri`、PKCE challenge，依 Google／Apple **不可變 `sub`** 查詢
   已連結的 Keycloak federated identity，之後才簽發一般 authorization code。
6. 接到 callback 後驗證原始 `state`，再以原始 PKCE verifier 對 Keycloak token endpoint
   換 token。access token 放記憶體；refresh token 僅放 iOS Keychain／Android Keystore。

若回應 `native_social_account_not_linked`（或 Keycloak 顯示通用登入失敗），不可註冊新
帳號、不可依 email 合併，應顯示「請先以既有 Visionnaire 帳號連結此社群帳號」。

## 安全帳號連結

帳號連結是高風險動作，登入中的 session 不足以直接執行。Flutter 必須先執行一次正常的
Keycloak Authorization Code + PKCE，額外帶入 `prompt=login` 與 `max_age=0`；使用者
完成密碼／MFA／hCaptcha 或既有社群驗證後，才會取得帶有新 `auth_time` 的 access token。

1. 帶 `Authorization: Bearer <fresh Keycloak access token>` 呼叫
   `POST /auth/native-social/links`：

   ```json
   { "provider": "apple" }
   ```

   後端只接受 issuer 為 Visionnaire Keycloak、`aud` 含 `visionnaire-api`、`auth_time`
   不超過五分鐘的 token；它會把 transaction 綁定 Keycloak `sub` 及 session ID。
2. 使用回傳 `nonce` 呼叫官方 SDK，然後以**同一個** bearer token 呼叫
   `POST /auth/native-social/links/complete`：

   ```json
   {
     "transaction_id": "<from links>",
     "id_token": "<provider identity token>",
     "authorization_code": "<Apple code when provider is apple>"
   }
   ```

3. 後端驗證 issuer、JWKS 簽章、audience、expiry、nonce（Apple 也會 server-to-server
   換驗 authorization code），再以 Keycloak Admin API 的官方 federated-identity
   endpoint 寫入 `provider + sub`。client body 從來沒有 target user ID；email 不參與
   連結。已連到另一帳號會得到 `409 provider_identity_already_linked`；同一帳號重試回
   `already_linked`。

連結成功後建議主動以 `prompt=login` 重新取得 token；不必也不應把社群 token 或 Apple
code 留在 App、Crash report、analytics、log 或資料庫。

## Flutter 實作界線

- iOS：使用 Apple 官方 Sign in with Apple 能力／受維護 Flutter wrapper，將 API 的
  nonce 傳給 native authorization request；Google 使用官方 Google Sign-In wrapper。
- Android：Google 使用 Google Identity Services／Credential Manager 的受維護 Flutter
  wrapper，並要求回傳 ID token 的 nonce；Apple 按鈕可只在 Apple 可用的平台顯示。
- Web：維持現有 BFF + Keycloak broker，因為 HttpOnly cookie 不會把 refresh token 暴露
  給瀏覽器 JavaScript。若日後產品明確要 Flutter Web 使用 Google Identity Services／
  Apple JS，必須新增獨立 public Keycloak client、精確 HTTPS redirect URI、嚴格 CSP，並
  將它加入 `NATIVE_SOCIAL_ALLOWED_CLIENTS_JSON`；不得重用 BFF confidential client。
- 所有平台不得把 `NATIVE_SOCIAL_EXCHANGE_SHARED_SECRET`、Keycloak User Linker client
  secret、Apple private key/client secret、hCaptcha secret 或 provider authorization code
  hard-code 到 App。

## 設定與營運

`NATIVE_SOCIAL_EXCHANGE_ENABLED=true` 時必須設定獨立的至少 32 字元
`NATIVE_SOCIAL_EXCHANGE_SHARED_SECRET`。此值只存在 Visionnaire API 與 Keycloak container
environment；Keycloak 以 `KC_NATIVE_SOCIAL_EXCHANGE_SHARED_SECRET` 收到同一值，並只可呼叫
`http://127.0.0.1:8005/auth/native-social/keycloak/redeem`。

`visionnaire-user-linker` 是 confidential service client，啟動程序只賦予
`realm-management` 的 `view-users` 與 `manage-users`，且其 secret 只存在 API。它用來呼叫
Keycloak 官方 federated identity Admin API；不能被 Flutter、Open WebUI 或一般 browser
取得。帳號連結事件應保留既有 API audit/request ID，並監控 `409`、nonce mismatch、HMAC
failure 與 Keycloak 503，但不得記錄 token、code、nonce 或 provider subject。
