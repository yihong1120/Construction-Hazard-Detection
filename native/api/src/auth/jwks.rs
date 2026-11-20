use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Result, ensure};
use jsonwebtoken::{Algorithm, DecodingKey, Validation, decode, decode_header};
use serde::{Deserialize, de::DeserializeOwned};
use tokio::sync::Mutex;

const TTL: Duration = Duration::from_secs(300);
const REFRESH_INTERVAL: Duration = Duration::from_secs(10);
const MAX_BYTES: usize = 256 * 1024;

pub(crate) enum Error {
    Invalid,
    Unavailable,
}

#[derive(Deserialize)]
struct KeySet {
    keys: Vec<Jwk>,
}

#[derive(Deserialize)]
struct Jwk {
    kty: String,
    kid: Option<String>,
    alg: Option<String>,
    #[serde(rename = "use")]
    purpose: Option<String>,
    key_ops: Option<Vec<String>>,
    n: Option<String>,
    e: Option<String>,
}

type Keys = HashMap<String, Arc<DecodingKey>>;

#[derive(Default)]
struct Cache {
    keys: Keys,
    loaded: Option<Instant>,
    attempted: Option<Instant>,
}

pub(crate) struct Verifier {
    client: reqwest::Client,
    url: String,
    validation: Validation,
    cache: Mutex<Cache>,
}

impl Verifier {
    pub(crate) fn new(url: String, issuers: &[&str], audiences: &[String]) -> Result<Self> {
        ensure!(
            !issuers.is_empty() && !audiences.is_empty(),
            "Missing issuer or audience"
        );
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .connect_timeout(Duration::from_secs(3))
            .timeout(Duration::from_secs(5))
            .build()?;
        let mut validation = Validation::new(Algorithm::RS256);
        validation.set_issuer(issuers);
        validation.set_audience(audiences);
        validation.set_required_spec_claims(&["iss", "aud", "sub", "exp"]);
        validation.validate_nbf = true;
        validation.leeway = 0;
        Ok(Self {
            client,
            url,
            validation,
            cache: Mutex::new(Cache::default()),
        })
    }

    pub(crate) async fn verify<T: DeserializeOwned>(&self, token: &str) -> Result<T, Error> {
        if token.len() > 16 * 1024 || token.bytes().any(|byte| byte.is_ascii_whitespace()) {
            return Err(Error::Invalid);
        }
        let header = decode_header(token).map_err(|_| Error::Invalid)?;
        if header.alg != Algorithm::RS256 {
            return Err(Error::Invalid);
        }
        let kid = header
            .kid
            .filter(|id| !id.is_empty() && id.len() <= 256)
            .ok_or(Error::Invalid)?;
        let key = {
            let mut cache = self.cache.lock().await;
            self.refresh(&mut cache, Some(&kid)).await?;
            cache.keys.get(&kid).cloned().ok_or(Error::Invalid)?
        };
        decode::<T>(token, &key, &self.validation)
            .map(|token| token.claims)
            .map_err(|_| Error::Invalid)
    }

    pub(crate) async fn ready(&self) -> Result<(), Error> {
        self.refresh(&mut *self.cache.lock().await, None).await
    }

    async fn refresh(&self, cache: &mut Cache, kid: Option<&str>) -> Result<(), Error> {
        let fresh = cache.loaded.is_some_and(|instant| instant.elapsed() < TTL);
        if fresh && kid.is_none_or(|kid| cache.keys.contains_key(kid)) {
            return Ok(());
        }
        if cache
            .attempted
            .is_some_and(|instant| instant.elapsed() < REFRESH_INTERVAL)
        {
            return if fresh {
                Ok(())
            } else {
                Err(Error::Unavailable)
            };
        }
        cache.attempted = Some(Instant::now());
        cache.keys = self.fetch().await.map_err(|_| Error::Unavailable)?;
        cache.loaded = Some(Instant::now());
        Ok(())
    }

    async fn fetch(&self) -> Result<Keys> {
        let mut response = self
            .client
            .get(&self.url)
            .send()
            .await?
            .error_for_status()?;
        let mut payload = Vec::new();
        while let Some(chunk) = response.chunk().await? {
            ensure!(
                payload.len() + chunk.len() <= MAX_BYTES,
                "Identity key set is too large"
            );
            payload.extend_from_slice(&chunk);
        }
        let set: KeySet = serde_json::from_slice(&payload)?;
        let mut keys = HashMap::new();
        for jwk in set.keys {
            if jwk.kty != "RSA"
                || jwk.purpose.is_some_and(|usage| usage != "sig")
                || jwk.alg.is_some_and(|alg| alg != "RS256")
                || jwk
                    .key_ops
                    .is_some_and(|ops| !ops.iter().any(|op| op == "verify"))
            {
                continue;
            }
            let (Some(kid), Some(n), Some(e)) = (jwk.kid, jwk.n, jwk.e) else {
                continue;
            };
            let key = Arc::new(DecodingKey::from_rsa_components(&n, &e)?);
            ensure!(
                keys.insert(kid, key).is_none(),
                "Duplicate identity signing key"
            );
        }
        ensure!(!keys.is_empty(), "Missing RS256 identity keys");
        Ok(keys)
    }
}
