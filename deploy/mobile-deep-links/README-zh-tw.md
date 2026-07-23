# Mobile deep links for password reset

目標是讓同一條正式 HTTPS reset link：

```text
https://changdar-server.mooo.com/reset_password?token={raw_reset_token}
```

在 iOS / Android 已安裝 app 時開啟 app，沒有安裝 app 或桌面瀏覽器時回到
Flutter Web 的 `index.html`。

## iOS Universal Links

伺服器必須提供：

```text
https://changdar-server.mooo.com/.well-known/apple-app-site-association
```

注意：

- 不要副檔名 `.json`
- `Content-Type` 建議 `application/json`
- 不要 redirect
- `appIDs` 必須是 `APPLE_TEAM_ID.BUNDLE_ID`
- iOS app 也要開啟 Associated Domains：
  `applinks:changdar-server.mooo.com`

正式檔已放在 `.well-known/apple-app-site-association`，內容使用：

```text
APPLE_TEAM_ID=5DU8R27949
IOS_BUNDLE_ID=com.changdar.visionnaire
appID=5DU8R27949.com.changdar.visionnaire
```

部署到：

```text
/.well-known/apple-app-site-association
```

## Android App Links

伺服器必須提供：

```text
https://changdar-server.mooo.com/.well-known/assetlinks.json
```

注意：

- `Content-Type` 建議 `application/json`
- `package_name` 必須是 Android app 的正式 package
- `sha256_cert_fingerprints` 必須是正式簽章憑證 SHA-256 fingerprint
- Android app manifest 的 intent filter 需包含
  `android:autoVerify="true"`、host `changdar-server.mooo.com`、pathPrefix
  `/reset_password`

正式檔已放在 `.well-known/assetlinks.json`，內容使用：

```text
ANDROID_PACKAGE_NAME=com.changdar.visionnaire
ANDROID_SIGNING_CERT_SHA256_FINGERPRINT=5E:79:A5:EE:8A:0E:28:91:CF:EF:B1:86:F6:E3:F1:2D:3A:D7:AD:49:24:0D:74:07:73:7B:56:98:B9:78:C8:85
```

部署到：

```text
/.well-known/assetlinks.json
```

## Web fallback

Nginx 需要確保：

```text
/reset_password?token=...
```

在 Web 上回 Flutter Web 的 `index.html`，不要 404。既有的 generic
`location / { try_files ... /index.html; }` 已涵蓋此路徑，不需要額外建立
`location = /reset_password`。可參考 `nginx.mobile-deep-links.conf`。

## Email reset link

後端目前會使用：

```text
APP_PUBLIC_URL/reset_password?token={raw_reset_token}
```

正式環境請設定：

```dotenv
APP_PUBLIC_URL=https://changdar-server.mooo.com
```

因此 email 會收到統一網址：

```text
https://changdar-server.mooo.com/reset_password?token={raw_reset_token}
```

手機 OS 會根據 Universal Links / App Links 驗證結果決定是否開啟 app。
沒有安裝 app、驗證尚未生效、或桌面瀏覽器開啟時，會 fallback 到 Flutter Web。
