use std::{
    collections::BTreeMap,
    net::{IpAddr, Ipv4Addr, SocketAddr},
    path::PathBuf,
};

use anyhow::{Context, Result, ensure};
use axum::{
    Extension, Json, Router,
    extract::{
        ConnectInfo, Path, Query, Request, State,
        rejection::{PathRejection, QueryRejection},
    },
    http::{HeaderMap, StatusCode},
    routing::{get, post, put},
};
use chrono::{DateTime, Utc};
use ipnet::IpNet;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use sqlx::{PgConnection, PgPool, types::Json as DbJson};
use uuid::Uuid;

use crate::{
    ApiError, AppState,
    auth::Grant,
    tenancy::{DeploymentScope, database_unavailable},
};

use super::{admin, groups::require_super, models::ResourceError};

pub mod controller;

const MAX_RULES: usize = 256;
const CLIENT_IP: ResourceError =
    ResourceError::Code(StatusCode::BAD_REQUEST, "client_ip_unavailable");
const CATALOG: ResourceError = ResourceError::Code(
    StatusCode::SERVICE_UNAVAILABLE,
    "website_access_schema_not_ready",
);
const MISSING: ResourceError = ResourceError::Code(StatusCode::NOT_FOUND, "website_not_found");
const RULE_MISSING: ResourceError =
    ResourceError::Code(StatusCode::NOT_FOUND, "website_access_rule_not_found");
const CONFLICT: ResourceError =
    ResourceError::Code(StatusCode::CONFLICT, "website_access_version_conflict");
const DUPLICATE: ResourceError =
    ResourceError::Code(StatusCode::CONFLICT, "website_access_duplicate_network");
const LOCKOUT: ResourceError =
    ResourceError::Code(StatusCode::CONFLICT, "website_access_would_lock_you_out");
const LIMIT: ResourceError = ResourceError::Code(StatusCode::CONFLICT, "website_access_rule_limit");
const INVALID_RULE: ResourceError = ResourceError::Code(
    StatusCode::UNPROCESSABLE_ENTITY,
    "invalid_website_access_rule",
);
const INVALID_ARGUMENT: ResourceError = ResourceError::Code(
    StatusCode::UNPROCESSABLE_ENTITY,
    "invalid_website_access_argument",
);

pub(super) fn routes() -> Router<AppState> {
    Router::new()
        .route("/admin/website-access", get(list))
        .route("/admin/website-access/context", get(context))
        .route("/admin/website-access/gateway", get(gateway))
        .route("/admin/website-access/audit", get(audit))
        .route("/admin/website-access/{site_key}", put(update_policy))
        .route("/admin/website-access/{site_key}/rules", post(create_rule))
        .route(
            "/admin/website-access/{site_key}/rules/{rule_id}",
            put(update_rule).delete(delete_rule),
        )
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(rename_all = "lowercase")]
enum SiteKey {
    Ai,
    Main,
    Wiki,
}

impl SiteKey {
    const ALL: [Self; 3] = [Self::Ai, Self::Main, Self::Wiki];

    fn as_str(self) -> &'static str {
        match self {
            Self::Ai => "ai",
            Self::Main => "main",
            Self::Wiki => "wiki",
        }
    }

    fn parse(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|key| key.as_str() == value)
    }
}

type Hosts = BTreeMap<SiteKey, String>;

struct Settings {
    hosts: Hosts,
    status_file: PathBuf,
}

impl Settings {
    fn from_env() -> Result<Self> {
        let hosts = match std::env::var("WEBSITE_ACCESS_HOSTS_JSON") {
            Ok(value) => serde_json::from_str(&value).context("Invalid website hosts")?,
            Err(std::env::VarError::NotPresent) => BTreeMap::from([
                (SiteKey::Ai, "ai.changdar-server.mooo.com".to_owned()),
                (SiteKey::Main, "changdar-server.mooo.com".to_owned()),
                (SiteKey::Wiki, "wiki.changdar-server.mooo.com".to_owned()),
            ]),
            Err(_) => anyhow::bail!("Invalid website hosts"),
        };
        validate_hosts(&hosts)?;
        let status_file = std::env::var_os("WEBSITE_ACCESS_STATUS_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| "/run/website-access/status.json".into());
        Ok(Self { hosts, status_file })
    }
}

fn validate_hosts(hosts: &Hosts) -> Result<()> {
    ensure!(
        hosts.len() == SiteKey::ALL.len(),
        "Expected main, ai and wiki website hosts"
    );
    let names = hosts.values().collect::<std::collections::HashSet<_>>();
    ensure!(
        names.len() == SiteKey::ALL.len()
            && hosts.values().all(|host| {
                host.len() <= 253
                    && host.split('.').all(|part| {
                        (1..=63).contains(&part.len())
                            && !part.starts_with('-')
                            && !part.ends_with('-')
                            && part
                                .bytes()
                                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
                    })
            }),
        "Invalid website hostnames"
    );
    Ok(())
}

fn settings() -> Result<Settings, ResourceError> {
    Settings::from_env().map_err(|_| {
        ResourceError::Code(
            StatusCode::SERVICE_UNAVAILABLE,
            "website_access_configuration_unavailable",
        )
    })
}

fn client_ip(headers: &HeaderMap, connection: Option<SocketAddr>) -> Option<IpAddr> {
    let address = if std::env::var("RUST_API_WEBSITE_ACCESS_TRUST_PROXY")
        .is_ok_and(|truth| matches!(truth.as_str(), "true" | "1"))
    {
        if !connection?.ip().is_loopback() {
            return None;
        }
        // The loopback Nginx listener sets this field to $remote_addr. Do not
        // accept client-selected chains, duplicate fields, or missing values.
        let mut addresses = headers.get_all("x-real-ip").iter();
        let address = addresses.next()?.to_str().ok()?.parse::<IpAddr>().ok()?;
        if addresses.next().is_some() {
            return None;
        }
        address
    } else {
        connection?.ip()
    };
    Some(address.to_canonical())
}

pub(crate) fn website_client_ip(request: &Request) -> Option<IpAddr> {
    let connection = request
        .extensions()
        .get::<ConnectInfo<SocketAddr>>()
        .map(|value| value.0);
    client_ip(request.headers(), connection)
}

type Peer = Option<Extension<ConnectInfo<SocketAddr>>>;

fn peer_address(peer: Peer, headers: &HeaderMap) -> Result<IpAddr, ResourceError> {
    client_ip(headers, peer.map(|Extension(value)| value.0)).ok_or(CLIENT_IP)
}

#[derive(Deserialize, Serialize)]
struct Receipt {
    checked_at: String,
    versions: BTreeMap<SiteKey, i32>,
    hosts: Hosts,
}

fn receipt_status(settings: Settings) -> Value {
    let result = || -> Result<Value> {
        let receipt: Receipt = serde_json::from_reader(std::fs::File::open(settings.status_file)?)?;
        let checked_at = DateTime::parse_from_rfc3339(&receipt.checked_at)?;
        ensure!(
            receipt.versions.len() == SiteKey::ALL.len()
                && receipt.versions.values().all(|&version| version >= 0),
            "Invalid gateway receipt"
        );
        let age = Utc::now().signed_duration_since(checked_at);
        let online = receipt.hosts == settings.hosts
            && age >= chrono::Duration::zero()
            && age < chrono::Duration::seconds(30);
        Ok(json!({"online": online,"checked_at": receipt.checked_at,"versions": receipt.versions}))
    };
    result().unwrap_or_else(|_| offline_receipt())
}

fn offline_receipt() -> Value {
    json!({"online": false, "checked_at": null, "versions": {}})
}

async fn require_super_reader(
    state: &AppState,
    scope: DeploymentScope,
    headers: &HeaderMap,
) -> Result<(), ApiError> {
    if admin(state, scope, headers, Grant::AccountRead)
        .await?
        .is_some()
    {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Super administrator required.",
        ));
    }
    Ok(())
}

async fn context(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    peer: Peer,
) -> Result<Json<Value>, ResourceError> {
    require_super_reader(&state, scope, &headers).await?;
    Ok(Json(
        json!({"client_ip": peer_address(peer, &headers)?, "can_manage_website_access": true}),
    ))
}

async fn gateway(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Value>, ResourceError> {
    require_super_reader(&state, scope, &headers).await?;
    let Ok(config) = settings() else {
        return Ok(Json(offline_receipt()));
    };
    let status = tokio::task::spawn_blocking(move || receipt_status(config))
        .await
        .map_err(|_| {
            ApiError::new(
                StatusCode::SERVICE_UNAVAILABLE,
                "Gateway status unavailable.",
            )
        })?;
    Ok(Json(status))
}

#[derive(Deserialize, Serialize)]
struct Rule {
    id: Uuid,
    network: String,
    label: String,
    enabled: bool,
    created_at: DateTime<Utc>,
}

struct Policy {
    site: SiteKey,
    restricted: bool,
    version: i32,
    updated_at: DateTime<Utc>,
    rules: Vec<Rule>,
}

impl Policy {
    fn snapshot(&self) -> Value {
        let rules: Vec<_> = self.rules.iter().map(|rule| {
            json!({"id":rule.id,"network":rule.network,"label":rule.label,"enabled":rule.enabled})
        }).collect();
        json!({"site_key":self.site,"restricted":self.restricted,"version":self.version,"rules":rules})
    }

    fn projection(&self, hosts: &Hosts) -> Value {
        json!({"site_key":self.site,"hostname":hosts[&self.site],"restricted":self.restricted,
            "version":self.version,"updated_at":self.updated_at,"rules":self.rules})
    }

    fn allows(&self, address: IpAddr) -> bool {
        !self.restricted
            || self.rules.iter().any(|rule| {
                rule.enabled
                    && canonical_network(&rule.network, true)
                        .is_some_and(|net| net.contains(&address))
            })
    }
}

fn network_prefix(mask: &str, is_v4: bool) -> Option<u8> {
    if !mask.is_empty() && mask.bytes().all(|byte| byte.is_ascii_digit()) {
        return mask.parse().ok();
    }
    if !is_v4 {
        return None;
    }
    let mask = u32::from(mask.parse::<Ipv4Addr>().ok()?);
    [mask, !mask].into_iter().find_map(|value| {
        let prefix = value.leading_ones();
        (value == u32::MAX.checked_shl(32 - prefix).unwrap_or(0)).then_some(prefix as u8)
    })
}

fn canonical_network(value: &str, strict: bool) -> Option<IpNet> {
    let (address, prefix) = if let Some((host, mask)) = value.split_once('/') {
        let address = host.parse::<IpAddr>().ok()?;
        (address, network_prefix(mask, address.is_ipv4())?)
    } else {
        let address = value.parse::<IpAddr>().ok()?;
        (address, if address.is_ipv4() { 32 } else { 128 })
    };
    let network = IpNet::new(address, prefix).ok()?.trunc();
    let mapped = "::ffff:0:0/96"
        .parse::<IpNet>()
        .expect("valid IPv4-mapped prefix");
    if prefix == 0
        || (strict && network.network() != address)
        || (address.is_ipv6()
            && (mapped.contains(&network.network()) || network.contains(&mapped.network())))
    {
        return None;
    }
    Some(network)
}

type PolicyRow = (String, bool, i32, DateTime<Utc>, DbJson<Vec<Rule>>);

async fn load_catalog(db: &PgPool) -> Result<Vec<Policy>, ResourceError> {
    let rows: Vec<PolicyRow> = sqlx::query_as(
        "SELECT p.site_key,p.restricted,p.version,p.updated_at,COALESCE(
           (SELECT jsonb_agg(to_jsonb(r)-'site_key' ORDER BY r.network) FROM website_access_rules r WHERE r.site_key=p.site_key),
           '[]'::jsonb) FROM website_access_policies p ORDER BY p.site_key",
    ).fetch_all(db).await.map_err(|error| {
        tracing::warn!(code = ?error.as_database_error().and_then(|error| error.code()), "website catalog unavailable");
        CATALOG
    })?;
    if rows.len() != SiteKey::ALL.len() {
        return Err(CATALOG);
    }
    rows.into_iter()
        .map(|(key, restricted, version, updated_at, DbJson(rules))| {
            Ok(Policy {
                site: SiteKey::parse(&key).ok_or(CATALOG)?,
                restricted,
                version,
                updated_at,
                rules,
            })
        })
        .collect()
}

async fn list(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
) -> Result<Json<Vec<Value>>, ResourceError> {
    require_super_reader(&state, scope, &headers).await?;
    let hosts = settings()?.hosts;
    let policies = load_catalog(&state.db).await?;
    Ok(Json(
        policies
            .iter()
            .map(|policy| policy.projection(&hosts))
            .collect(),
    ))
}

#[derive(Deserialize)]
struct AuditQuery {
    site_key: Option<SiteKey>,
    #[serde(default = "default_audit_limit")]
    limit: i64,
}

fn default_audit_limit() -> i64 {
    50
}

async fn audit(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    query: Result<Query<AuditQuery>, QueryRejection>,
) -> Result<Json<Vec<Value>>, ResourceError> {
    require_super_reader(&state, scope, &headers).await?;
    let Query(query) = query.map_err(|_| INVALID_ARGUMENT)?;
    if !(1..=200).contains(&query.limit) {
        return Err(ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "Invalid audit limit.").into());
    }
    let rows = sqlx::query_scalar(
        "SELECT jsonb_build_object('id',a.id,'site_key',a.site_key,'actor_user_id',a.actor_user_id,
           'actor_ip',a.actor_ip,'action',a.action,'before',a.before,'after',a.after,'created_at',a.created_at)
         FROM website_access_audit_logs a WHERE $1::text IS NULL OR a.site_key=$1
         ORDER BY a.created_at DESC,a.id DESC LIMIT $2",
    ).bind(query.site_key.map(SiteKey::as_str)).bind(query.limit)
        .fetch_all(&state.db).await.map_err(database_unavailable)?;
    Ok(Json(rows))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyInput {
    restricted: bool,
    expected_version: i32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RuleInput {
    network: String,
    label: String,
    #[serde(default = "enabled")]
    enabled: bool,
    expected_version: i32,
}

fn enabled() -> bool {
    true
}

impl RuleInput {
    fn normalize(&mut self) -> Result<(), ResourceError> {
        if !(1..=64).contains(&self.network.chars().count())
            || !(1..=120).contains(&self.label.chars().count())
            || self.label.contains('\0')
        {
            return Err(INVALID_RULE);
        }
        self.network = canonical_network(self.network.trim(), false)
            .ok_or(INVALID_RULE)?
            .to_string();
        self.label = self.label.trim().to_owned();
        if self.label.is_empty() {
            return Err(INVALID_RULE);
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct VersionQuery {
    expected_version: i32,
}

enum Change {
    Policy(PolicyInput),
    Create(RuleInput),
    Update(Uuid, RuleInput),
    Delete(Uuid, VersionQuery),
}

impl Change {
    fn normalize(&mut self) -> Result<(), ResourceError> {
        if let Self::Create(rule) | Self::Update(_, rule) = self {
            rule.normalize()?;
        }
        Ok(())
    }

    fn expected_version(&self) -> i32 {
        match self {
            Self::Policy(body) => body.expected_version,
            Self::Create(body) | Self::Update(_, body) => body.expected_version,
            Self::Delete(_, query) => query.expected_version,
        }
    }

    fn action(&self) -> &'static str {
        match self {
            Self::Policy(_) => "policy_updated",
            Self::Create(_) => "rule_created",
            Self::Update(..) => "rule_updated",
            Self::Delete(..) => "rule_deleted",
        }
    }
}

async fn lock_policy(
    db: &mut PgConnection,
    site: SiteKey,
    expected: i32,
) -> Result<Policy, ResourceError> {
    let (restricted, version, updated_at) = sqlx::query_as(
        "SELECT restricted,version,updated_at FROM website_access_policies WHERE site_key=$1 FOR UPDATE",
    ).bind(site.as_str()).fetch_optional(&mut *db).await.map_err(database_unavailable)?.ok_or(MISSING)?;
    if version != expected {
        return Err(CONFLICT);
    }
    let DbJson(rules) = sqlx::query_scalar(
        "SELECT COALESCE(jsonb_agg(to_jsonb(r)-'site_key' ORDER BY r.network),'[]'::jsonb)
         FROM website_access_rules r WHERE r.site_key=$1",
    )
    .bind(site.as_str())
    .fetch_one(db)
    .await
    .map_err(database_unavailable)?;
    Ok(Policy {
        site,
        restricted,
        version,
        updated_at,
        rules,
    })
}

async fn write_rules(
    db: &mut PgConnection,
    policy: &mut Policy,
    change: Change,
) -> Result<(), ResourceError> {
    let site = policy.site.as_str();
    match change {
        Change::Policy(body) => policy.restricted = body.restricted,
        Change::Create(rule) => {
            if policy.rules.iter().any(|r| r.network == rule.network) {
                return Err(DUPLICATE);
            }
            if policy.rules.len() >= MAX_RULES {
                return Err(LIMIT);
            }
            let id = Uuid::new_v4();
            let created_at = sqlx::query_scalar(
                "INSERT INTO website_access_rules(id,site_key,network,label,enabled)
                 VALUES($1,$2,$3,$4,$5) RETURNING created_at",
            )
            .bind(id)
            .bind(site)
            .bind(&rule.network)
            .bind(&rule.label)
            .bind(rule.enabled)
            .fetch_one(db)
            .await
            .map_err(database_unavailable)?;
            policy.rules.push(Rule {
                id,
                network: rule.network,
                label: rule.label,
                enabled: rule.enabled,
                created_at,
            });
        }
        Change::Update(id, rule) => {
            let target = policy
                .rules
                .iter()
                .position(|r| r.id == id)
                .ok_or(RULE_MISSING)?;
            if policy
                .rules
                .iter()
                .any(|r| r.id != id && r.network == rule.network)
            {
                return Err(DUPLICATE);
            }
            sqlx::query("UPDATE website_access_rules SET network=$3,label=$4,enabled=$5 WHERE site_key=$1 AND id=$2")
                .bind(site).bind(id).bind(&rule.network).bind(&rule.label).bind(rule.enabled)
                .execute(db).await.map_err(database_unavailable)?;
            let target = &mut policy.rules[target];
            target.network = rule.network;
            target.label = rule.label;
            target.enabled = rule.enabled;
        }
        Change::Delete(id, _) => {
            let target = policy
                .rules
                .iter()
                .position(|r| r.id == id)
                .ok_or(RULE_MISSING)?;
            sqlx::query("DELETE FROM website_access_rules WHERE site_key=$1 AND id=$2")
                .bind(site)
                .bind(id)
                .execute(db)
                .await
                .map_err(database_unavailable)?;
            policy.rules.remove(target);
        }
    }
    policy
        .rules
        .sort_unstable_by(|left, right| left.network.cmp(&right.network));
    Ok(())
}

async fn change_policy(
    state: &AppState,
    scope: DeploymentScope,
    headers: &HeaderMap,
    peer: Peer,
    site: SiteKey,
    mut change: Change,
) -> Result<Json<Value>, ResourceError> {
    let actor = state.auth.authorize(headers, Grant::AccountManage).await?;
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let account_id = require_super(&mut tx, scope, &actor.subject).await?;
    let address = peer_address(peer, headers)?;
    let hosts = settings()?.hosts;
    let expected = change.expected_version();
    if expected < 0 {
        return Err(
            ApiError::new(StatusCode::UNPROCESSABLE_ENTITY, "Invalid policy version.").into(),
        );
    }
    change.normalize()?;
    let mut policy = lock_policy(&mut tx, site, expected).await?;
    let action = change.action();
    let before = policy.snapshot();
    write_rules(&mut tx, &mut policy, change).await?;
    if !policy.allows(address) {
        return Err(LOCKOUT);
    }
    (policy.version, policy.updated_at) = sqlx::query_as(
        "UPDATE website_access_policies SET restricted=$2,version=version+1,updated_at=now()
         WHERE site_key=$1 RETURNING version,updated_at",
    )
    .bind(site.as_str())
    .bind(policy.restricted)
    .fetch_one(&mut *tx)
    .await
    .map_err(database_unavailable)?;
    sqlx::query(
        "INSERT INTO website_access_audit_logs(id,site_key,actor_user_id,actor_ip,action,before,after)
         VALUES($1,$2,$3,$4,$5,$6,$7)",
    ).bind(Uuid::new_v4()).bind(site.as_str()).bind(account_id).bind(address.to_string())
        .bind(action).bind(before).bind(policy.snapshot()).execute(&mut *tx).await.map_err(database_unavailable)?;
    tx.commit().await.map_err(database_unavailable)?;
    Ok(Json(policy.projection(&hosts)))
}

async fn update_policy(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    peer: Peer,
    path: Result<Path<SiteKey>, PathRejection>,
    Json(body): Json<PolicyInput>,
) -> Result<Json<Value>, ResourceError> {
    let Path(site) = path.map_err(|_| INVALID_ARGUMENT)?;
    change_policy(&state, scope, &headers, peer, site, Change::Policy(body)).await
}

async fn create_rule(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    peer: Peer,
    path: Result<Path<SiteKey>, PathRejection>,
    Json(body): Json<RuleInput>,
) -> Result<(StatusCode, Json<Value>), ResourceError> {
    let Path(site) = path.map_err(|_| INVALID_ARGUMENT)?;
    let response = change_policy(&state, scope, &headers, peer, site, Change::Create(body)).await?;
    Ok((StatusCode::CREATED, response))
}

async fn update_rule(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    peer: Peer,
    path: Result<Path<(SiteKey, Uuid)>, PathRejection>,
    Json(body): Json<RuleInput>,
) -> Result<Json<Value>, ResourceError> {
    let Path((site, rule)) = path.map_err(|_| INVALID_ARGUMENT)?;
    change_policy(
        &state,
        scope,
        &headers,
        peer,
        site,
        Change::Update(rule, body),
    )
    .await
}

async fn delete_rule(
    State(state): State<AppState>,
    Extension(scope): Extension<DeploymentScope>,
    headers: HeaderMap,
    peer: Peer,
    path: Result<Path<(SiteKey, Uuid)>, PathRejection>,
    query: Result<Query<VersionQuery>, QueryRejection>,
) -> Result<Json<Value>, ResourceError> {
    let Path((site, rule)) = path.map_err(|_| INVALID_ARGUMENT)?;
    let Query(query) = query.map_err(|_| INVALID_ARGUMENT)?;
    change_policy(
        &state,
        scope,
        &headers,
        peer,
        site,
        Change::Delete(rule, query),
    )
    .await
}
