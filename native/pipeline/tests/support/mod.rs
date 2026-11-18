//! Refuse application databases in tests that persist fixture events.
pub fn check_postgres_if_configured() {
    if let Ok(url) = std::env::var("CHD_RUST_TEST_DATABASE_URL") {
        let parsed = reqwest::Url::parse(&url).expect("Invalid isolated PostgreSQL test URL");
        assert!(matches!(parsed.host_str(), Some("127.0.0.1" | "localhost")));
        assert_eq!(parsed.path(), "/chd_rust_pipeline_test");
    }
}
