use std::time::Duration;

use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

use crate::{
    auth::{Authorizer, DatabaseAccess, DatabaseAccessConfig, OidcSettings},
    streaming::{media_auth::MediaAuth, metadata::Metadata},
};

use super::*;

const PUBLIC: &str = "https://fixture.example/hazard/api";
const HOST: &str = "fixture.example";
const CLIENT: &str = "native-fixture";
const REDIRECT: &str = "visionnaire://callback";
const CHALLENGE: &str = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";

struct Client {
    http: reqwest::Client,
    api: String,
    idp: String,
    key: Hmac<Sha256>,
}

impl Client {
    async fn send(&self, route: &str, payload: Value, token: Option<&str>, expected: u16) -> Value {
        let mut request = self
            .http
            .post(format!("{}/auth/native-social/{route}", self.api))
            .header("Host", HOST)
            .header("X-Real-IP", "192.0.2.7")
            .json(&payload);
        if let Some(token) = token {
            request = request.bearer_auth(token);
        }
        let response = request.send().await.unwrap();
        assert_eq!(response.status().as_u16(), expected, "{route}");
        response.json().await.unwrap()
    }

    async fn begin(&self, provider: &str) -> Value {
        self.send("exchanges", json!({
            "client_id": CLIENT, "redirect_uri": REDIRECT, "code_challenge": CHALLENGE,
            "code_challenge_method":"S256", "state":"state from native SDK", "provider":provider,
        }), None, 200).await
    }

    async fn signed(
        &self,
        provider: &str,
        subject: &str,
        nonce: &str,
        mail: &str,
        session: &str,
        age: i64,
    ) -> String {
        let response: Value = self
            .http
            .get(format!("{}/fixture/token", self.idp))
            .query(&[
                ("provider", provider),
                ("subject", subject),
                ("nonce", nonce),
                ("email", mail),
                ("sid", session),
                ("age", &age.to_string()),
            ])
            .send()
            .await
            .unwrap()
            .error_for_status()
            .unwrap()
            .json()
            .await
            .unwrap();
        s(&response, "token").to_owned()
    }

    async fn redeemed(
        &self,
        id: &str,
        host: Option<&str>,
        drift: i64,
        valid_signature: bool,
        challenge: &str,
        status: u16,
    ) -> Value {
        let payload = json!({
            "transaction_id":id, "client_id":CLIENT, "redirect_uri":REDIRECT, "code_challenge":challenge,
        }).to_string();
        let timestamp = (chrono::Utc::now().timestamp() + drift).to_string();
        let mut mac = self.key.clone();
        mac.update(timestamp.as_bytes());
        mac.update(b".");
        mac.update(payload.as_bytes());
        let signature = URL_SAFE_NO_PAD.encode(mac.finalize().into_bytes());
        let mut request = self
            .http
            .post(format!("{}/auth/native-social/keycloak/redeem", self.api))
            .header("X-Visionnaire-Timestamp", timestamp)
            .header(
                "X-Visionnaire-Signature",
                if valid_signature {
                    &signature
                } else {
                    "invalid"
                },
            )
            .body(payload);
        if let Some(host) = host {
            request = request.header("Host", host);
        }
        let response = request.send().await.unwrap();
        assert_eq!(response.status().as_u16(), status);
        response.json().await.unwrap()
    }

    async fn complete(
        &self,
        proof: &Value,
        token: Option<&str>,
        code: Option<&str>,
        expected: u16,
    ) -> Value {
        self.send("exchanges/complete", json!({
            "transaction_id":s(proof,"transaction_id"),"id_token":token,"authorization_code":code,
        }), None, expected).await
    }
}

fn s<'a>(response: &'a Value, key: &str) -> &'a str {
    response[key].as_str().unwrap()
}

#[tokio::test]
#[ignore = "requires the isolated PostgreSQL, Redis and provider fixture"]
async fn native_social_exchange_and_account_link() {
    let mock = std::env::var("CHD_SOCIAL_FIXTURE_HTTP").unwrap();
    let apple_key = std::env::var("CHD_SOCIAL_FIXTURE_APPLE_KEY").unwrap();
    let scope: DeploymentScope = serde_json::from_value(json!({
        "tenant_id":"1b17d9e1-477f-4de8-ab7c-8ad00feeb1cd",
        "deployment_id":"3e6b9a8e-5673-49ee-9f74-a9faff836a61",
    }))
    .unwrap();
    let db = crate::connect_database().await.unwrap();
    let redis = redis::Client::open(std::env::var("RUST_API_REDIS_URL").unwrap()).unwrap();
    let manager = ConnectionManager::new(redis.clone()).await.unwrap();
    let issuer = format!("{mock}/realms/fixture");
    let pg_access = DatabaseAccess::new(
        DatabaseAccessConfig {
            public_api_base_url: PUBLIC.into(),
            loopback_deployment_access: false,
        },
        scope,
        db.clone(),
    )
    .await
    .unwrap();
    let auth = Authorizer::configured(
        OidcSettings {
            issuer: issuer.clone(),
            audiences: vec!["fixture-api".into()],
            jwks_url: format!("{mock}/keycloak/jwks"),
        },
        vec![],
        manager.clone(),
        Some(pg_access),
    )
    .unwrap();
    let state = AppState {
        db: db.clone(),
        auth,
        media_auth: MediaAuth::new(redis.clone()).await.unwrap(),
        metadata: Metadata::new(redis, manager.clone()),
        evidence_api: false,
        playback: None,
        bff: None,
        ice: None,
        shutdown: CancellationToken::new(),
        websocket_lifetime: Duration::from_secs(20),
        #[cfg(feature = "tensorrt")]
        inference: None,
    };

    let key = Hmac::<Sha256>::new_from_slice(b"test-only-broker-secret-valid-length").unwrap();
    let providers = provider::Providers::with_endpoints(serde_json::from_value(json!({
        "google_client_ids":["google-fixture"],"apple_client_ids":["apple-web","apple-mobile"],
        "apple":{"team_id":"TESTTEAM", "key_id":"TESTKEY", "private_key_path":apple_key,
            "service_id":"apple-web","bundle_id":"apple-mobile","redirect_uri":"https://fixture.example/callback"},
    })).unwrap(),&format!("{mock}/google/jwks"),&format!("{mock}/apple/jwks"),&format!("{mock}/apple/token")).unwrap();
    let router = router(Social(Arc::new(Settings {
        scope,
        allowed_clients: HashMap::from([(CLIENT.into(), vec![REDIRECT.into()])]),
        authorization_endpoint: Url::parse(&format!("{issuer}/protocol/openid-connect/auth"))
            .unwrap(),
        issuer,
        exchange_ttl: 90,
        link_ttl: 120,
        max_auth_age: 300,
        trust_proxy: true,
        redis: manager.clone(),
        broker_key: key.clone(),
        keycloak: Keycloak::from_env().unwrap().unwrap(),
        providers,
    })))
    .with_state(state);
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let api = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        axum::serve(
            listener,
            router.into_make_service_with_connect_info::<SocketAddr>(),
        )
        .await
        .unwrap()
    });
    let c = Client {
        api,
        idp: mock,
        key,
        http: reqwest::Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap(),
    };

    let google = c.begin("google").await;
    let invalid = c
        .signed("google", "google-foreign", "wrong", "", "", 0)
        .await;
    c.complete(&google, Some(&invalid), None, 401).await;
    let again = c.complete(&google, Some(&invalid), None, 401).await;
    assert_eq!(again["detail"], "native_social_exchange_expired");

    let google = c.begin("google").await;
    let wrong_issuer = c
        .signed(
            "bad-google",
            "google-existing",
            s(&google, "nonce"),
            "",
            "",
            0,
        )
        .await;
    assert_eq!(
        c.complete(&google, Some(&wrong_issuer), None, 401).await["detail"],
        "Invalid provider token"
    );
    let google = c.begin("google").await;
    let token = c
        .signed("google", "google-existing", s(&google, "nonce"), "", "", 0)
        .await;
    let result = c.complete(&google, Some(&token), None, 200).await;
    let authorization = Url::parse(s(&result, "authorization_url")).unwrap();
    let params: HashMap<_, _> = authorization.query_pairs().into_owned().collect();
    assert_eq!(
        params["native_social_exchange"],
        s(&google, "transaction_id")
    );
    assert_eq!(params["redirect_uri"], REDIRECT);
    assert_eq!(params["code_challenge"], CHALLENGE);
    assert_eq!(params["state"], "state from native SDK");
    assert_eq!(result["expires_in"], 90);
    let transaction = s(&google, "transaction_id");
    c.redeemed(transaction, Some(HOST), 0, true, CHALLENGE, 403)
        .await;
    c.redeemed(transaction, None, 90, true, CHALLENGE, 401)
        .await;
    c.redeemed(transaction, None, 0, false, CHALLENGE, 401)
        .await;
    let redeemed = c.redeemed(transaction, None, 0, true, CHALLENGE, 200).await;
    assert_eq!(
        redeemed,
        json!({"provider":"google","provider_subject_b64":URL_SAFE_NO_PAD.encode("google-existing")})
    );
    c.redeemed(transaction, None, 0, true, CHALLENGE, 401).await;

    let apple = c.begin("apple").await;
    let code = format!("apple-code:{}", s(&apple, "nonce"));
    let response = c.complete(&apple, None, Some(&code), 200).await;
    assert!(response["authorization_url"].is_string());
    c.redeemed(
        s(&apple, "transaction_id"),
        None,
        0,
        true,
        "foreign-challenge",
        401,
    )
    .await;
    c.redeemed(s(&apple, "transaction_id"), None, 0, true, CHALLENGE, 401)
        .await;
    let apple = c.begin("apple").await;
    let apple_token = c
        .signed("apple", "apple-existing", s(&apple, "nonce"), "", "", 0)
        .await;
    c.complete(&apple, Some(&apple_token), None, 401).await;
    let apple = c.begin("apple").await;
    let apple_nonce = s(&apple, "nonce");
    let apple_token = c
        .signed("apple", "apple-existing", apple_nonce, "", "", 0)
        .await;
    c.complete(
        &apple,
        Some(&apple_token),
        Some(&format!("wrong-subject:{apple_nonce}")),
        401,
    )
    .await;

    let recent = c
        .signed("keycloak", "kc-user-one", "", "", "current-session", 1)
        .await;
    let other = c
        .signed("keycloak", "kc-user-two", "", "", "other-session", 1)
        .await;
    let old = c
        .signed("keycloak", "kc-user-one", "", "", "current-session", 350)
        .await;
    let rotated = c
        .signed("keycloak", "kc-user-one", "", "", "rotated-session", 1)
        .await;
    let stale = c
        .send("links", json!({"provider":"google"}), Some(&old), 401)
        .await;
    assert_eq!(stale["detail"], "keycloak_reauthentication_required");
    let link = c
        .send("links", json!({"provider":"google"}), Some(&recent), 200)
        .await;
    let proof = c
        .signed("google", "google-one", s(&link, "nonce"), "", "", 0)
        .await;
    let body = json!({"transaction_id":s(&link,"transaction_id"),"id_token":proof});
    c.send("links/complete", body.clone(), Some(&rotated), 401)
        .await;
    c.send("links/complete", body, Some(&recent), 401).await;

    let google = c.begin("google").await;
    let proof = c
        .signed(
            "google",
            "google-one",
            s(&google, "nonce"),
            "member@example.test",
            "",
            0,
        )
        .await;
    let link = c.complete(&google, Some(&proof), None, 409).await;
    assert_eq!(link["detail"]["code"], "account_link_required");
    let confirm = json!({"transaction_id":s(&link["detail"],"link_transaction_id")});
    c.send(
        "email-link-confirmations/complete",
        confirm,
        Some(&other),
        401,
    )
    .await;

    let google = c.begin("google").await;
    let proof = c
        .signed(
            "google",
            "google-one",
            s(&google, "nonce"),
            "member@example.test",
            "",
            0,
        )
        .await;
    let link = c.complete(&google, Some(&proof), None, 409).await;
    let confirm = json!({"transaction_id":s(&link["detail"],"link_transaction_id")});
    assert_eq!(
        c.send(
            "email-link-confirmations/complete",
            confirm.clone(),
            Some(&recent),
            200
        )
        .await,
        json!({"provider":"google","status":"linked"})
    );
    c.send(
        "email-link-confirmations/complete",
        confirm,
        Some(&recent),
        401,
    )
    .await;
    let mirrors: Vec<(String, String, Option<String>, bool)> = sqlx::query_as(
        "SELECT provider, provider_user_id, email, email_verified
           FROM public.user_identities WHERE provider <> 'keycloak'",
    )
    .fetch_all(&db)
    .await
    .unwrap();
    assert_eq!(
        mirrors,
        vec![("google".into(), "google-one".into(), None, false)]
    );

    for (actor, status) in [(&recent, 200), (&other, 409)] {
        let link = c
            .send("links", json!({"provider":"google"}), Some(actor), 200)
            .await;
        let proof = c
            .signed("google", "google-one", s(&link, "nonce"), "", "", 0)
            .await;
        let linked = c
            .send(
                "links/complete",
                json!({
                    "transaction_id":s(&link,"transaction_id"),"id_token":proof,
                }),
                Some(actor),
                status,
            )
            .await;
        if status == 200 {
            assert_eq!(linked["status"], "already_linked");
        } else {
            assert_eq!(linked["detail"], "provider_identity_already_linked");
        }
    }

    let remaining: u64 = redis::cmd("GET")
        .arg(format!("native-social:begin-rate:{}", digest("192.0.2.7")))
        .query_async(&mut manager.clone())
        .await
        .unwrap();
    for _ in remaining..20 {
        c.begin("google").await;
    }
    let limited = c
        .http
        .post(format!("{}/auth/native-social/exchanges", c.api))
        .header("Host", HOST)
        .header("X-Real-IP", "192.0.2.7")
        .json(&json!({
            "client_id":CLIENT,"redirect_uri":REDIRECT,"code_challenge":CHALLENGE,
            "code_challenge_method":"S256","state":"still local","provider":"google",
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(limited.status(), 429);
    assert!(
        limited.headers()["Retry-After"]
            .to_str()
            .unwrap()
            .parse::<u64>()
            .unwrap()
            <= 300
    );
    server.abort();
}

#[test]
fn reads_python_transaction_wire_format() {
    let form = r#"{"kind":"login-begin","provider":"apple","nonce":"nonce","client_id":"native-fixture","redirect_uri":"visionnaire://callback","code_challenge":"challenge","state":"state"}"#;
    let transaction: Transaction = serde_json::from_str(form).unwrap();
    assert_eq!(
        serde_json::to_value(transaction).unwrap(),
        serde_json::from_str::<Value>(form).unwrap()
    );
}
