//! Shared Redis metadata names and stock overlay palette.

use base64::{Engine, engine::general_purpose::URL_SAFE};
use serde_json::{Map, Value};
use std::sync::OnceLock;

/// Encode a UTF-8 site or stream name using padded URL-safe Base64.
pub fn encode_stream_id(name: &str) -> String {
    URL_SAFE.encode(name)
}

pub fn generation_key(prefix: &str, site: &str) -> String {
    let site = encode_stream_id(site);
    format!("{prefix}stream_metadata_generation:{site}")
}

/// Build the Redis key from a site name and an already encoded stream ID.
/// The caller validates IDs originating in requests before calling this function.
pub fn metadata_key_from_stream_id(
    prefix: &str,
    site: &str,
    stream_id: &str,
    generation: i64,
) -> String {
    let site = encode_stream_id(site);
    if generation > 0 {
        format!("{prefix}stream_metadata:{site}:g{generation}|{stream_id}")
    } else {
        format!("{prefix}stream_metadata:{site}|{stream_id}")
    }
}

/// Return the base colour map shared by image and live video overlays.
pub fn overlay_palette() -> &'static Map<String, Value> {
    static PALETTE: OnceLock<Map<String, Value>> = OnceLock::new();
    PALETTE.get_or_init(|| {
        serde_json::from_str(include_str!("../assets/colors.json"))
            .expect("compiled overlay palette")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_preserve_legacy_prefix_padding_and_generation() {
        let stream_id = encode_stream_id("camera1");
        assert_eq!(stream_id, "Y2FtZXJhMQ==");
        assert_eq!(encode_stream_id("工地"), "5bel5Zyw");
        assert_eq!(
            generation_key("preview:", "demo"),
            "preview:stream_metadata_generation:ZGVtbw=="
        );
        assert_eq!(
            metadata_key_from_stream_id("preview:", "demo", &stream_id, 3),
            "preview:stream_metadata:ZGVtbw==:g3|Y2FtZXJhMQ=="
        );
        for generation in [0, -1] {
            assert_eq!(
                metadata_key_from_stream_id("", "工地", &stream_id, generation),
                "stream_metadata:5bel5Zyw|Y2FtZXJhMQ=="
            );
        }
    }

    #[test]
    fn default_palette_is_shared_and_preserves_colours() {
        let colors = overlay_palette();
        assert!(std::ptr::eq(colors, overlay_palette()));
        assert_eq!(colors.len(), 12);
        assert_eq!(colors["person"], "#FF9800");
        assert_eq!(colors["no_helmet"], "#F44336");
        assert_eq!(colors["forklift"], "#AB47BC");
    }
}
