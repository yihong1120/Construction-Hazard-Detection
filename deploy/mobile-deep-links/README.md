🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Mobile deep links for password resets

Replace the example domains, addresses and identifiers with the values for your deployment.

The following production HTTPS link supports all password reset clients.

```text
https://app.example.org/reset_password?token={raw_reset_token}
```

On iOS and Android the link opens the installed app. Visitors without the app and desktop users reach the Flutter Web `index.html` page.

## iOS Universal Links

Serve the association file at this address.

```text
https://app.example.org/.well-known/apple-app-site-association
```

File and app requirements

- Use a filename without the `.json` extension.
- Use `application/json` as the recommended `Content-Type`.
- Serve the file directly, without a redirect.
- Set `appIDs` to `APPLE_TEAM_ID.BUNDLE_ID`.
- Enable Associated Domains in the iOS app with `applinks:app.example.org`.

For example, `.well-known/apple-app-site-association` uses values in this format.

```text
APPLE_TEAM_ID=YOURTEAMID
IOS_BUNDLE_ID=com.example.visionnaire
appID=YOURTEAMID.com.example.visionnaire
```

Deploy it to this location.

```text
/.well-known/apple-app-site-association
```

## Android App Links

Serve the association file at this address.

```text
https://app.example.org/.well-known/assetlinks.json
```

File and app requirements

- Use `application/json` as the recommended `Content-Type`.
- Set `package_name` to the production Android app package name.
- Set `sha256_cert_fingerprints` to the SHA-256 fingerprint of the production signing certificate.
- Include `android:autoVerify="true"`, host `app.example.org` and pathPrefix `/reset_password` in the Android app manifest's intent filter.

For example, `.well-known/assetlinks.json` uses values in this format.

```text
ANDROID_PACKAGE_NAME=com.example.visionnaire
ANDROID_SIGNING_CERT_SHA256_FINGERPRINT=<release-certificate-sha256>
```

Deploy it to this location.

```text
/.well-known/assetlinks.json
```

## Web fallback

Nginx must handle the following path.

```text
/reset_password?token=...
```

Return Flutter Web's `index.html` for this path. The existing generic `location / { try_files ... /index.html; }` handles it, so a separate `location = /reset_password` is unnecessary. See `nginx.mobile-deep-links.conf`.

## Password reset email

The backend uses the following setting.

```text
APP_PUBLIC_URL/reset_password?token={raw_reset_token}
```

Configure production as shown below.

```dotenv
APP_PUBLIC_URL=https://app.example.org
```

The email then contains the same link for all clients.

```text
https://app.example.org/reset_password?token={raw_reset_token}
```

The mobile OS opens the app according to its Universal Links or App Links verification result. Desktop browsers, devices without the app, and devices where verification has yet to take effect open Flutter Web.
