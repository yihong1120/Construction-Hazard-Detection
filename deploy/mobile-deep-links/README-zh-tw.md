🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 密碼重設的行動裝置深層連結

請將範例中的網域、位址與識別碼替換成自己部署環境的值。

目標是讓同一條正式 HTTPS reset link：

```text
https://app.example.org/reset_password?token={raw_reset_token}
```

在 iOS / Android 已安裝 app 時開啟 app，沒有安裝 app 或桌面瀏覽器時回到
Flutter Web 的 `index.html`。

## iOS 通用連結（Universal Links）

伺服器必須提供：

```text
https://app.example.org/.well-known/apple-app-site-association
```

注意：

- 不要副檔名 `.json`
- `Content-Type` 建議 `application/json`
- 不要 redirect
- `appIDs` 必須是 `APPLE_TEAM_ID.BUNDLE_ID`
- iOS app 也要開啟 Associated Domains：
  `applinks:app.example.org`

請依部署環境填寫 `.well-known/apple-app-site-association`，以下為範例格式。

```text
APPLE_TEAM_ID=YOURTEAMID
IOS_BUNDLE_ID=com.example.visionnaire
appID=YOURTEAMID.com.example.visionnaire
```

部署到：

```text
/.well-known/apple-app-site-association
```

## Android 應用程式連結（App Links）

伺服器必須提供：

```text
https://app.example.org/.well-known/assetlinks.json
```

注意：

- `Content-Type` 建議 `application/json`
- `package_name` 必須是 Android app 的正式 package
- `sha256_cert_fingerprints` 必須是正式簽章憑證 SHA-256 fingerprint
- Android app manifest 的 intent filter 需包含
  `android:autoVerify="true"`、host `app.example.org`、pathPrefix
  `/reset_password`

請依部署環境填寫 `.well-known/assetlinks.json`，以下為範例格式。

```text
ANDROID_PACKAGE_NAME=com.example.visionnaire
ANDROID_SIGNING_CERT_SHA256_FINGERPRINT=<release-certificate-sha256>
```

部署到：

```text
/.well-known/assetlinks.json
```

## 未開啟原生應用程式時的網頁

Nginx 需要確保：

```text
/reset_password?token=...
```

在 Web 上回 Flutter Web 的 `index.html`，不要 404。既有的 generic
`location / { try_files ... /index.html; }` 已涵蓋此路徑，不需要額外建立
`location = /reset_password`。可參考 `nginx.mobile-deep-links.conf`。

## 電子郵件中的密碼重設連結

後端目前會使用：

```text
APP_PUBLIC_URL/reset_password?token={raw_reset_token}
```

正式環境請設定：

```dotenv
APP_PUBLIC_URL=https://app.example.org
```

因此 email 會收到統一網址：

```text
https://app.example.org/reset_password?token={raw_reset_token}
```

手機 OS 會根據 Universal Links / App Links 驗證結果決定是否開啟 app。
沒有安裝 app、驗證尚未生效、或桌面瀏覽器開啟時，會 fallback 到 Flutter Web。
