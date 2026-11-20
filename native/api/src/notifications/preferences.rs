use std::collections::BTreeMap;

use axum::{
    Extension, Json,
    extract::State,
    http::{HeaderMap, StatusCode},
};
use serde::Deserialize;
use serde_json::Value;
use sqlx::{Executor, Postgres};

use crate::{
    ApiError, AppState,
    accounts::is_super_admin,
    auth::Actor,
    tenancy::{Account, database_unavailable},
};

use super::{Settings, account};

async fn visible_sites<'e>(
    db: impl Executor<'e, Database = Postgres>,
    actor: &Actor,
    user: &Account,
    lock: bool,
) -> Result<Vec<(i32, String, Value)>, ApiError> {
    let super_admin = is_super_admin(user);
    if user.group_id.is_none() && !super_admin {
        return Err(ApiError::new(StatusCode::FORBIDDEN, "User without group."));
    }
    // Explicit service grants retain their configured site boundary. A database
    // user may manage preferences for group sites before receiving stream access.
    let dynamic = actor.user_id.is_some();
    let ids: Vec<_> = if dynamic {
        Vec::new()
    } else {
        actor.sites().into_iter().map(|(id, _)| id).collect()
    };
    let group_lock = if lock { "FOR KEY SHARE OF sg" } else { "" };
    let mut sql = format!(
        "
        SELECT s.id, s.name, jsonb_build_object(
            'site_id', s.id, 'site_name', s.name, 'group_name', g.name,
            'is_enabled', COALESCE(p.is_enabled, us.user_id IS NOT NULL OR $6)
        )
        FROM sites s
        JOIN LATERAL (
            SELECT sg.group_id FROM site_groups sg
            WHERE sg.site_id = s.id AND (
                ($3 AND EXISTS (
                    SELECT 1 FROM users member
                    WHERE member.group_id = sg.group_id
                    AND member.tenant_id = (SELECT tenant_id FROM users WHERE id = $2)
                ))
                OR (NOT $3 AND sg.group_id = $1)
            )
            ORDER BY sg.group_id LIMIT 1 {group_lock}
        ) scope ON TRUE
        JOIN group_info g ON g.id = scope.group_id
        LEFT JOIN site_notification_preferences p ON p.site_id = s.id AND p.user_id = $2
        LEFT JOIN user_sites us ON us.site_id = s.id AND us.user_id = $2
        WHERE ($4 OR s.id::bigint = ANY($5))
        ORDER BY s.id"
    );
    if lock {
        // Receipt revision triggers also write sites. Take that lock in ID order.
        sql.push_str(" FOR UPDATE OF s");
    }
    let mut sites: Vec<(i32, String, Value)> = sqlx::query_as(&sql)
        .bind(user.group_id)
        .bind(user.id)
        .bind(super_admin)
        .bind(dynamic)
        .bind(ids)
        .bind(user.role == "super_admin")
        .fetch_all(db)
        .await
        .map_err(database_unavailable)?;
    if !dynamic {
        sites.retain(|(id, name, _)| actor.allows_site(i64::from(*id), name));
    }
    Ok(sites)
}

pub(super) async fn list(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
) -> Result<Json<Vec<Value>>, ApiError> {
    let (actor, user) = account(&state, &settings, &headers).await?;
    Ok(Json(
        visible_sites(&state.db, &actor, &user, false)
            .await?
            .into_iter()
            .map(|(_, _, view)| view)
            .collect(),
    ))
}

#[derive(Deserialize)]
pub(super) struct Preferences {
    preferences: Vec<Preference>,
}

#[derive(Deserialize)]
struct Preference {
    site_id: i32,
    is_enabled: bool,
}

pub(super) async fn update(
    State(state): State<AppState>,
    Extension(settings): Extension<Settings>,
    headers: HeaderMap,
    Json(body): Json<Preferences>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let (actor, user) = account(&state, &settings, &headers).await?;
    if body.preferences.is_empty() || body.preferences.len() > 128 {
        return Err(ApiError::new(
            StatusCode::UNPROCESSABLE_ENTITY,
            "Invalid notification preferences",
        ));
    }
    let mut changes = BTreeMap::new();
    for preference in body.preferences {
        if changes
            .insert(preference.site_id, preference.is_enabled)
            .is_some()
        {
            return Err(ApiError::new(
                StatusCode::UNPROCESSABLE_ENTITY,
                "Duplicate notification preference",
            ));
        }
    }
    let mut tx = state.db.begin().await.map_err(database_unavailable)?;
    let sites = visible_sites(&mut *tx, &actor, &user, true).await?;
    if changes
        .keys()
        .any(|requested| !sites.iter().any(|(id, _, _)| id == requested))
    {
        return Err(ApiError::new(
            StatusCode::FORBIDDEN,
            "Cannot subscribe to sites outside your scope.",
        ));
    }
    let (ids, enabled): (Vec<i32>, Vec<bool>) =
        changes.iter().map(|(&id, &value)| (id, value)).unzip();
    sqlx::query(
        "INSERT INTO site_notification_preferences (user_id, site_id, is_enabled, updated_at)
         SELECT $1, site_id, enabled, now() FROM unnest($2::int[], $3::bool[]) AS value(site_id, enabled)
         ON CONFLICT (user_id, site_id) DO UPDATE SET is_enabled = EXCLUDED.is_enabled, updated_at = now()
         WHERE site_notification_preferences.is_enabled IS DISTINCT FROM EXCLUDED.is_enabled"
    ).bind(user.id).bind(ids).bind(enabled).execute(&mut *tx).await.map_err(database_unavailable)?;
    tx.commit().await.map_err(database_unavailable)?;
    Ok(Json(
        sites
            .into_iter()
            .map(|(id, _, mut view)| {
                if let Some(enabled) = changes.get(&id) {
                    view["is_enabled"] = Value::Bool(*enabled);
                }
                view
            })
            .collect(),
    ))
}
