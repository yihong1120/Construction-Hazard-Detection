use std::{collections::BTreeMap, io, path::PathBuf};

use axum::{Json, http::StatusCode, response::IntoResponse};
use serde::Deserialize;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sqlx::PgConnection;

use crate::{ApiError, tenancy::Account};

pub(super) enum ResourceError {
    Api(ApiError),
    Code(StatusCode, &'static str),
    Detail(StatusCode, Value),
}

impl From<ApiError> for ResourceError {
    fn from(error: ApiError) -> Self {
        Self::Api(error)
    }
}

impl IntoResponse for ResourceError {
    fn into_response(self) -> axum::response::Response {
        match self {
            Self::Api(error) => error.into_response(),
            Self::Code(status, code) => {
                (status, Json(json!({"detail": {"code": code}}))).into_response()
            }
            Self::Detail(status, detail) => {
                (status, Json(json!({"detail": detail}))).into_response()
            }
        }
    }
}

const UNAVAILABLE: ResourceError = ResourceError::Code(StatusCode::CONFLICT, "MODEL_UNAVAILABLE");
const REGISTRY_ERROR: ResourceError = ResourceError::Code(
    StatusCode::SERVICE_UNAVAILABLE,
    "MODEL_REGISTRY_UNAVAILABLE",
);

#[derive(Deserialize)]
struct Definition {
    display_name: String,
    version: String,
    sha256: String,
    #[serde(default = "enabled")]
    enabled: bool,
    capabilities: Vec<String>,
    #[serde(default)]
    tenant_ids: Vec<String>,
    #[serde(default)]
    site_ids: Vec<i32>,
    #[serde(default = "default_roles")]
    roles: Vec<String>,
}

fn enabled() -> bool {
    true
}

fn default_roles() -> Vec<String> {
    ["admin", "user", "guest", "super_admin"]
        .map(str::to_owned)
        .into()
}

fn is_sha256(digest: &str) -> bool {
    digest.len() == 64
        && digest
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

impl Definition {
    fn allowed(&self, tenant: &str, role: &str, site_id: i32) -> bool {
        self.enabled
            && self.capabilities.iter().any(|role| role == "stream")
            && self.tenant_ids.iter().any(|id| id == tenant)
            && self.roles.iter().any(|allowed| allowed == role)
            && self.site_ids.contains(&site_id)
    }
}

struct Model {
    id: String,
    default: bool,
    raw: Value,
    definition: Definition,
}

pub(super) struct Catalog(Vec<Model>);

impl Catalog {
    pub(super) async fn load(db: &mut PgConnection) -> Result<Self, ResourceError> {
        let rows: Vec<(String, bool, Value)> = sqlx::query_as(
            "SELECT model_key, is_default, definition::jsonb FROM detection_model_catalog
             ORDER BY display_order, model_key",
        )
        .fetch_all(db)
        .await
        .map_err(|error| {
            tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "model catalog query failed");
            REGISTRY_ERROR
        })?;
        let mut models = Vec::with_capacity(rows.len());
        for (id, default, raw) in rows {
            let definition: Definition =
                serde_json::from_value(raw.clone()).map_err(|_| REGISTRY_ERROR)?;
            if id.trim().is_empty()
                || definition.display_name.trim().is_empty()
                || definition.version.trim().is_empty()
                || !is_sha256(&definition.sha256)
            {
                return Err(REGISTRY_ERROR);
            }
            models.push(Model {
                id,
                default,
                raw,
                definition,
            });
        }
        Ok(Self(models))
    }

    pub(super) fn options(&self, tenant: &str, account: &Account, site_id: i32) -> Value {
        let mut options = Vec::new();
        let mut default_id = None;
        for model in &self.0 {
            if model.definition.allowed(tenant, &account.role, site_id) {
                options.push(json!({"id": model.id, "label": model.definition.display_name}));
                if model.default {
                    default_id = Some(&model.id);
                }
            }
        }
        let source = self
            .0
            .iter()
            .map(|model| (&model.id, model.default, &model.raw))
            .collect::<Vec<_>>();
        resource_options(
            tenant,
            account,
            json!(site_id),
            options,
            json!(default_id),
            json!(source),
        )
    }

    pub(super) async fn validate_models<'a>(
        &self,
        tenant: &str,
        account: &Account,
        site_id: i32,
        model_ids: impl IntoIterator<Item = &'a str>,
    ) -> Result<(), ResourceError> {
        let mut selected = BTreeMap::new();
        for id in model_ids {
            let model = self
                .0
                .iter()
                .find(|model| model.id == id && model.definition.enabled)
                .ok_or(UNAVAILABLE)?;
            if !model.definition.allowed(tenant, &account.role, site_id) {
                return Err(ApiError::new(
                    StatusCode::FORBIDDEN,
                    "Model is not allowed for this resource",
                )
                .into());
            }
            selected.insert(
                model.id.clone(),
                (
                    model.definition.version.clone(),
                    model.definition.sha256.clone(),
                ),
            );
        }
        if selected.is_empty() {
            return Ok(());
        }
        let path = std::env::var_os("RUST_API_STREAM_ENGINE_MANIFEST")
            .map(PathBuf::from)
            .ok_or(ResourceError::Code(
                StatusCode::SERVICE_UNAVAILABLE,
                "MODEL_ENGINES_UNAVAILABLE",
            ))?;
        tokio::task::spawn_blocking(move || verify_engines(path, selected))
            .await
            .map_err(|_| UNAVAILABLE)?
    }
}

#[derive(Deserialize)]
struct EngineManifest {
    format: u32,
    models: BTreeMap<String, EnginePin>,
}

#[derive(Deserialize)]
struct EnginePin {
    artifact: PathBuf,
    sha256: String,
    source_sha256: String,
    version: String,
}

fn verify_engines(
    path: PathBuf,
    selected: BTreeMap<String, (String, String)>,
) -> Result<(), ResourceError> {
    let file = std::fs::File::open(path).map_err(|_| UNAVAILABLE)?;
    let manifest: EngineManifest = serde_json::from_reader(file).map_err(|_| UNAVAILABLE)?;
    if manifest.format != 1 {
        return Err(UNAVAILABLE);
    }
    for (key, (version, digest)) in selected {
        let engine = manifest.models.get(&key).ok_or(UNAVAILABLE)?;
        if engine.version != version
            || engine.source_sha256 != digest
            || !is_sha256(&engine.sha256)
            || !engine.artifact.is_absolute()
            || engine
                .artifact
                .extension()
                .is_none_or(|extension| extension != "engine")
        {
            return Err(UNAVAILABLE);
        }
        let mut file = std::fs::File::open(&engine.artifact).map_err(|_| UNAVAILABLE)?;
        if !file.metadata().map_err(|_| UNAVAILABLE)?.is_file() {
            return Err(UNAVAILABLE);
        }
        let mut hash = Sha256::new();
        io::copy(&mut file, &mut hash).map_err(|_| UNAVAILABLE)?;
        if format!("{:x}", hash.finalize()) != engine.sha256 {
            return Err(UNAVAILABLE);
        }
    }
    Ok(())
}

pub(super) fn resource_options(
    tenant: &str,
    account: &Account,
    target: Value,
    options: Vec<Value>,
    default_id: Value,
    source: Value,
) -> Value {
    let mut result =
        json!({"schema_version": 1, "options": options, "default_id": default_id, "policy": {}});
    let input = json!({
        "user": account.id, "tenant": tenant, "role": account.role,
        "target": target, "source": source, "result": result,
    });
    result["revision"] = json!(format!(
        "{:x}",
        Sha256::digest(input.to_string().as_bytes())
    ));
    result
}
