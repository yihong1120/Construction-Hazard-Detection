pub(crate) fn canonical(language: &str) -> Option<&'static str> {
    match language {
        "en" => Some("en"),
        "zh-TW" => Some("zh-TW"),
        "zh-CN" => Some("zh-CN"),
        "ja" => Some("ja"),
        "vi" => Some("vi"),
        "id" => Some("id"),
        "fr" => Some("fr"),
        "th" => Some("th"),
        _ => {
            let normalized = language.trim().replace('_', "-").to_ascii_lowercase();
            match normalized.as_str() {
                "zh" | "zh-hant" | "zh-tw" | "zh-hk" | "zh-mo" => Some("zh-TW"),
                "zh-hans" | "zh-cn" | "zh-sg" => Some("zh-CN"),
                "jp" => Some("ja"),
                _ => match normalized.split('-').next().unwrap_or_default() {
                    "en" => Some("en"),
                    "ja" => Some("ja"),
                    "vi" => Some("vi"),
                    "id" => Some("id"),
                    "fr" => Some("fr"),
                    "th" => Some("th"),
                    _ => None,
                },
            }
        }
    }
}

pub(crate) enum Status {
    Warning,
    Pending,
    Offline,
}

impl Status {
    pub(crate) fn text(self, language: &str) -> &'static str {
        let (warning, pending, offline) = match canonical(language) {
            Some("zh-TW") => ("警告", "分析中", "影像來源無法使用，正在重新連線"),
            Some("zh-CN") => ("警告", "分析中", "视频源不可用，正在重新连接"),
            Some("ja") => ("警告", "解析中", "映像を受信できません。再接続中"),
            Some("vi") => (
                "CẢNH BÁO",
                "Đang phân tích",
                "Mất tín hiệu video, đang kết nối lại",
            ),
            Some("id") => (
                "PERINGATAN",
                "Sedang menganalisis",
                "Video terputus. Menyambung ulang",
            ),
            Some("fr") => (
                "ALERTE",
                "Analyse en cours",
                "Vidéo indisponible. Reconnexion en cours",
            ),
            Some("th") => ("คำเตือน", "กำลังวิเคราะห์", "ไม่พบสัญญาณภาพ กำลังเชื่อมต่อใหม่"),
            _ => (
                "WARNING",
                "Analysis pending",
                "Video source unavailable. Reconnecting",
            ),
        };
        match self {
            Self::Warning => warning,
            Self::Pending => pending,
            Self::Offline => offline,
        }
    }
}
