# Keycloak 社群登入部署規格

Google／Apple 在 Flutter Web BFF 作為 Keycloak Identity Broker；Flutter iOS／Android
則可使用官方 SDK 加上 Visionnaire 的一次性原生憑證交換。兩條路徑的最終 access／refresh
token 都由同一個 Keycloak issuer 核發。Visionnaire 保留本機的 tenant、角色、群組、site
與 feature 授權資料，且不依 email 自動連結帳號。這個界線適用 Flutter Web、iOS、Android
與後續 Open WebUI。

## 三平台流程

```text
Flutter Web ───── BFF Authorization Code ────────────────┐
Flutter iOS ─┐                                            │
Flutter Android ─┴─ provider SDK → one-use proof → Keycloak
                                                        ├─ 本機帳密 → hCaptcha
                                                        ├─ Web broker Google → Google
                                                        └─ Web broker Apple  → Apple
                                                                  ↓
                                              Keycloak OIDC token (aud=visionnaire-api)
                                                                  ↓
                                                      Visionnaire local authorisation
```

Web 仍使用 BFF cookie；iOS／Android 仍直接持有 Keycloak access/refresh token。原生
provider assertion 的詳盡 nonce、PKCE、HMAC redemption 與帳號連結流程在
[原生社群憑證交換規格](native_social_exchange.md)。

## 第三方後台的一次性設定

### Google Cloud Console

建立新的 **Web application** OAuth 2.0 client。不得重用 Flutter iOS 或 Android client ID，redirect URI 必須精確設為：

```text
https://changdar-server.mooo.com/keycloak/realms/visionnaire/broker/google/endpoint
```

將新的 client ID 與 client secret 放入主機 `.env`：

```dotenv
KEYCLOAK_GOOGLE_ENABLED=true
KEYCLOAK_GOOGLE_CLIENT_ID='...apps.googleusercontent.com'
KEYCLOAK_GOOGLE_CLIENT_SECRET='...'
```

### Apple Developer

在既有 Service ID `com.changdar.visionnaire.signin` 開啟 Sign in with Apple，並將網域與 Return URL 改為：

```text
網域：changdar-server.mooo.com
Return URL：https://changdar-server.mooo.com/keycloak/realms/visionnaire/broker/apple/endpoint
```

Apple 的 `client_secret` 是最多 180 天有效的 ES256 JWT。使用主機上的 `.p8` 私鑰產生，輸出是 secret，不要貼入 Git、終端紀錄或 Flutter：

```bash
uv run python scripts/generate_apple_client_secret.py \
  --team-id "$APPLE_TEAM_ID" \
  --key-id "$APPLE_KEY_ID" \
  --client-id 'com.changdar.visionnaire.signin' \
  --private-key-file config/secrets/apple/AuthKey_NGC4QBS7ZY.p8
```

把輸出存入部署 secret／`.env`，然後啟用 provider：

```dotenv
KEYCLOAK_APPLE_ENABLED=true
KEYCLOAK_APPLE_CLIENT_ID='com.changdar.visionnaire.signin'
KEYCLOAK_APPLE_CLIENT_SECRET='generated-jwt'
```

至少每 150 天重新產生並更新 `KEYCLOAK_APPLE_CLIENT_SECRET`，再重新部署 Keycloak。Keycloak 使用內建 generic OIDC broker 對接 Apple，私鑰不會進入容器。

## 啟用與驗收

完成各自後台設定與 `.env` 後執行：

```bash
docker compose --env-file .env -f deploy/keycloak/docker-compose.yml up -d --build
```

Keycloak 只會在 `*_ENABLED=true` 且必需 credential 齊全時顯示 provider；設定不完整時會停用該 provider，避免向使用者顯示壞掉的按鈕。

登入頁應顯示 Google／Apple。以下 URL 也應分別直接跳轉至相應第三方：

```text
https://changdar-server.mooo.com/bff/auth/oidc/login?return_to=/&idp_hint=google
https://changdar-server.mooo.com/bff/auth/oidc/login?return_to=/&idp_hint=apple
```

## 帳號與權限

Keycloak 的 `sub` 才是 Visionnaire 授權對應的不可變識別；Google／Apple 的 `sub` 由
Keycloak 保存為 federated identity。不要以 email 自動連結，Apple Private Relay 特別容易
造成誤綁。原生 SDK 路徑僅在使用者已用近期 Keycloak reauthentication 明確執行連結後，才
會把 `provider + sub` 寫入 Keycloak；不存在 first-login 自動註冊或 email 合併。

既有使用者應先由管理員建立／關聯 Keycloak 帳號並用 `scripts/keycloak_link_users.py` 對應到 Visionnaire `user_identities(provider=keycloak)`。之後使用者可在 Keycloak Account Console 將 Google 或 Apple 加為登入方法。尚未連結的 Keycloak 使用者即使第三方驗證成功，也不會獲得 Visionnaire API 權限。

Keycloak 的 [Identity Brokering 官方文件](https://www.keycloak.org/docs/latest/server_admin/#_identity_broker)說明 brokered login 與 First Login Flow；本專案保留它的預設 first-broker flow，並以 Visionnaire 的本機權限對應作最終拒絕判斷。
