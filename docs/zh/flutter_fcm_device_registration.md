# Flutter FCM 裝置註冊

## 目的

登入 token 與 Firebase Cloud Messaging（FCM）device token 是不同資料：登入 token
證明使用者身分；FCM token 則代表單一瀏覽器或手機安裝。後端無法在登入時自行產生
FCM token，因此 Flutter 必須在取得 FCM token 後註冊一次。

請將此流程集中在單一 `PushRegistrationCoordinator`，而非由每個頁面呼叫 API。
Coordinator 應在下列時機執行：

- 登入成功且已取得通知權限後；
- App 恢復登入狀態後；
- `FirebaseMessaging.instance.onTokenRefresh` 發出新 token 時。

同一 token 重複呼叫 `PUT` 是預期用法，後端會更新同一筆裝置註冊。

## 固定 JSON 契約

所有平台都使用完全相同的 body：

```json
{
  "device_token": "FCM_TOKEN",
  "device_lang": "zh-TW",
  "platform": "web"
}
```

- `device_token`：Firebase 取得的 FCM registration token。
- `device_lang`：完整 BCP 47 code，例如 `zh-TW`、`en-GB`。
- `platform`：只能是 `web`、`ios` 或 `android`。

三個欄位缺少、值不合法，或仍送出舊的 `user_id` 等額外欄位，後端皆回傳
`422 Unprocessable Content`。使用者由登入狀態決定，絕不放在 body。

## Flutter Web

Web 使用 BFF session cookie，**不使用也不持有 bearer token**。

1. `POST /bff/auth/login` 建立 BFF session。
2. `GET /bff/auth/csrf` 取得 CSRF token。
3. 以 session cookie 與 `X-CSRF-Token` 呼叫 `PUT /bff/fcm/devices`。

```text
PUT /bff/fcm/devices
X-CSRF-Token: <csrf-token>
Content-Type: application/json
```

BFF 會在伺服器端將 session 對應的 access token 轉送到 notification service。

## Flutter iOS 與 Android

原生 App 使用登入 API 回傳的 access token：

```text
PUT <notification-api-base>/devices
Authorization: Bearer <access-token>
Content-Type: application/json
```

App 不需也不得傳送 `user_id`。使用者由 JWT subject 決定。

## 平台值

```dart
import 'package:flutter/foundation.dart';

String fcmPlatform() {
  if (kIsWeb) {
    return 'web';
  }
  return switch (defaultTargetPlatform) {
    TargetPlatform.iOS => 'ios',
    TargetPlatform.android => 'android',
    _ => throw UnsupportedError('FCM is not configured for this platform.'),
  };
}
```

## Coordinator 的責任

Coordinator 取得 token、組合固定 JSON，並依平台選擇上列 transport；畫面、登入表單、
通知頁面不應各自實作註冊邏輯。登出前若要停止目前裝置接收該帳號通知，使用同一份
JSON 中的 `device_token` 呼叫 `DELETE /devices`；Web 對應路徑是
`DELETE /bff/fcm/devices`，並同樣需要 CSRF token。
