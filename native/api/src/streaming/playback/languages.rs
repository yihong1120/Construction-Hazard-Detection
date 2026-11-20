use axum::body::Bytes;

use super::*;

struct Locale {
    code: &'static str,
    notification: &'static str,
    english: &'static str,
    native: &'static str,
    aliases: &'static [&'static str],
}

const LOCALES: [Locale; 8] = [
    Locale {
        code: "en",
        notification: "en-GB",
        english: "English",
        native: "English",
        aliases: &["en-GB", "en-US"],
    },
    Locale {
        code: "zh-TW",
        notification: "zh-TW",
        english: "Traditional Chinese",
        native: "繁體中文",
        aliases: &["zh", "zh-Hant", "zh_TW", "zh-HK", "zh-mo"],
    },
    Locale {
        code: "zh-CN",
        notification: "zh-CN",
        english: "Simplified Chinese",
        native: "简体中文",
        aliases: &["zh-Hans", "zh_CN", "zh-SG"],
    },
    Locale {
        code: "ja",
        notification: "ja-JP",
        english: "Japanese",
        native: "日本語",
        aliases: &["jp", "ja-JP"],
    },
    Locale {
        code: "vi",
        notification: "vi-VN",
        english: "Vietnamese",
        native: "Tiếng Việt",
        aliases: &["vi-VN"],
    },
    Locale {
        code: "id",
        notification: "id-ID",
        english: "Indonesian",
        native: "Bahasa Indonesia",
        aliases: &["id-ID"],
    },
    Locale {
        code: "fr",
        notification: "fr-FR",
        english: "French",
        native: "Français",
        aliases: &["fr-FR", "fr-CA"],
    },
    Locale {
        code: "th",
        notification: "th-TH",
        english: "Thai",
        native: "ไทย",
        aliases: &["th-TH"],
    },
];

fn catalogue(
    allowed: &[&str],
    default: &str,
    public_base: &str,
    limit: u32,
    demand_ttl: u32,
) -> Value {
    let labels: Value =
        serde_json::from_str(include_str!("../../../../pipeline/assets/labels.json"))
            .expect("valid embedded pipeline translations");
    let notifications: Value =
        serde_json::from_str(include_str!("../../notifications/languages.json"))
            .expect("valid embedded notification translations");
    let mut aliases = serde_json::Map::new();
    for locale in &LOCALES {
        for code in std::iter::once(locale.code).chain(locale.aliases.iter().copied()) {
            aliases.insert(code.to_owned(), locale.code.into());
            aliases.insert(code.to_ascii_lowercase(), locale.code.into());
        }
    }
    let languages: Vec<_> = allowed
        .iter()
        .map(|code| {
            let locale = LOCALES
                .iter()
                .find(|locale| &locale.code == code)
                .expect("MediaAuth accepts only supported overlay languages");
            json!({
                "code": locale.code,
                "notification_code": locale.notification,
                "display_name": locale.english,
                "native_name": locale.native,
                "is_default": locale.code == default,
                "class_labels": &labels["CLASS_LABELS"][locale.code],
                "warning_labels": &labels["WARNING_LABELS"][locale.code],
                "notification_templates": &notifications[locale.notification],
            })
        })
        .collect();
    json!({
        "default_language": default,
        "allowed_language_codes": allowed,
        "aliases": aliases,
        "languages": languages,
        "stream_playback_endpoint": public_base,
        "max_active_languages_per_stream": limit,
        "demand_ttl_seconds": demand_ttl,
        "ready_ttl_seconds": 3,
    })
}

pub(super) async fn get(
    State(state): State<AppState>,
    headers: HeaderMap,
) -> Result<Response, Response> {
    state
        .auth
        .authorize(&headers, Grant::StreamRead)
        .await
        .map_err(ApiError::into_response)?;
    let playback = runtime(&state).map_err(Failure::into_response)?;
    let media = state.media_auth.playback_settings();
    let settings = &playback.settings;
    let payload = settings.language_response.get_or_init(|| {
        Bytes::from(
            catalogue(
                &media.languages,
                media.default_language,
                &settings.public_base,
                media.overlay_limit,
                settings.overlay_demand_ttl,
            )
            .to_string(),
        )
    });
    Ok((
        [
            (header::CONTENT_TYPE, "application/json"),
            (header::CACHE_CONTROL, "private, no-store"),
        ],
        payload.clone(),
    )
        .into_response())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn locale_catalogue_has_native_labels_and_notification_templates() {
        let codes: Vec<_> = LOCALES.iter().map(|entry| entry.code).collect();
        let catalogue = catalogue(&codes, "zh-TW", "/playback", 5, 90);
        assert_eq!(catalogue["aliases"]["zh_Hant"], Value::Null);
        assert_eq!(catalogue["aliases"]["zh-hant"], "zh-TW");
        assert_eq!(catalogue["aliases"]["zh_TW"], "zh-TW");
        assert_eq!(catalogue["languages"][1]["is_default"], true);
        assert_eq!(catalogue["stream_playback_endpoint"], "/playback");
        for language in catalogue["languages"].as_array().unwrap() {
            assert!(language["class_labels"]["person"].is_string());
            assert!(language["warning_labels"]["warning_no_hardhat"].is_string());
            assert!(language["notification_templates"]["warning_no_hardhat"].is_string());
        }
    }
}
