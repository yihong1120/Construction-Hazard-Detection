use crate::{
    config::{Model, Redis, Stream},
    hazard::{HazardResult, Polygons, Track, Warnings},
};
use anyhow::Result;
use construction_stream_contract::{encode_stream_id, generation_key, metadata_key_from_stream_id};
use redis::aio::ConnectionManager;
use serde::Serialize;
use serde_json::{Value, json};
use std::{collections::BTreeMap, time::Duration};

#[derive(Serialize)]
struct Payload<'a> {
    stream_id: &'a str,
    site_id: i64,
    sequence: u64,
    generation: u64,
    owner: &'a str,
    timestamp: f64,
    source_pts: Option<f64>,
    frame_width: u32,
    frame_height: u32,
    labels: &'a BTreeMap<u32, BTreeMap<String, String>>,
    track_data: &'a [Track],
    warnings: &'a Warnings,
    cone_polys: &'a Polygons,
    pole_polys: &'a Polygons,
    model_id: &'a str,
    model_version: &'a str,
    catalog_revision: &'a Option<String>,
    class_metadata: &'a [Value],
}

pub struct Sink {
    connection: ConnectionManager,
    generation_key: String,
    generation: i64,
    owner: String,
    token: String,
    latest: String,
    warning: String,
    last_warning: i64,
    last_status: i64,
    prefix: String,
}
impl Sink {
    pub async fn new(config: &Redis, spec: &Stream) -> Result<Self> {
        let url = std::env::var(&config.url_env)?;
        let client = redis::Client::open(url)?;
        let mut connection =
            tokio::time::timeout(Duration::from_secs(5), ConnectionManager::new(client)).await??;
        let generation_key = generation_key(&config.prefix, &spec.site);
        let generation: Option<i64> = redis::cmd("GET")
            .arg(&generation_key)
            .query_async(&mut connection)
            .await?;
        let generation = generation.unwrap_or(0);
        let warning = metadata_key_from_stream_id(
            &config.prefix,
            &spec.site,
            &encode_stream_id(&spec.name),
            generation,
        );
        Ok(Self {
            connection,
            generation_key,
            generation,
            owner: format!("{}rust_owner:{}:{}", config.prefix, spec.site_id, spec.id),
            token: uuid::Uuid::new_v4().to_string(),
            latest: format!("{}rust_latest:{}:{}", config.prefix, spec.site_id, spec.id),
            warning,
            last_warning: 0,
            last_status: 0,
            prefix: config.prefix.clone(),
        })
    }
    pub async fn publish(
        &mut self,
        spec: &Stream,
        sequence: u64,
        generation: u64,
        timestamp: i64,
        tracks: &[Track],
        hazard: &HazardResult,
        working: bool,
        model: &Model,
    ) -> Result<bool> {
        self.publish_source(
            spec, sequence, generation, timestamp, tracks, hazard, working, model, None,
        )
        .await
    }
    pub async fn publish_source(
        &mut self,
        spec: &Stream,
        sequence: u64,
        generation: u64,
        timestamp: i64,
        tracks: &[Track],
        hazard: &HazardResult,
        working: bool,
        model: &Model,
        source_pts: Option<f64>,
    ) -> Result<bool> {
        let warn =
            working && !hazard.warnings.is_empty() && timestamp / 1000 - self.last_warning >= 30;
        let payload = serde_json::to_string(&Payload {
            stream_id: &spec.id,
            site_id: spec.site_id,
            sequence,
            generation,
            owner: &self.token,
            timestamp: timestamp as f64 / 1000.,
            source_pts,
            frame_width: spec.width,
            frame_height: spec.height,
            labels: &model.labels,
            track_data: tracks,
            warnings: &hazard.warnings,
            cone_polys: &hazard.cone_polys,
            pole_polys: &hazard.pole_polys,
            model_id: &model.key,
            model_version: &model.version,
            catalog_revision: &model.catalog_revision,
            class_metadata: &model.class_metadata,
        })?;
        let status = if model.catalog_revision.is_some()
            && timestamp / 1000 - self.last_status >= 10
        {
            serde_json::to_string(&json!({"model_id":model.key,"model_version":model.version,
                "catalog_revision":model.catalog_revision,"model_sha256":model.source_sha256.as_ref().unwrap_or(&model.sha256)}))?
        } else {
            String::new()
        };
        let script=redis::Script::new("local g=redis.call('GET',KEYS[1]) or '0'; if g~=ARGV[1] then return 0 end;
          local o=redis.call('GET',KEYS[2]); if o and o~=ARGV[2] then return 0 end;
          redis.call('SET',KEYS[2],ARGV[2],'EX',30);redis.call('SET',KEYS[3],ARGV[3],'EX',30);
          if ARGV[6]=='1' then redis.call('XADD',KEYS[4],'MAXLEN','~',32,'*','has_warning',ARGV[7],'overlay',ARGV[3]);redis.call('EXPIRE',KEYS[4],30) elseif ARGV[4]=='1' then redis.call('XADD',KEYS[4],'MAXLEN','~',10,'*','has_warning','1') end;
          if ARGV[5]~='' then local s=cjson.decode(ARGV[5]);s.publication=redis.call('GET',KEYS[6]) or cjson.null;redis.call('SET',KEYS[5],cjson.encode(s),'EX',45) end;return 1");
        let accepted: i32 = tokio::time::timeout(
            Duration::from_secs(3),
            script
                .key(&self.generation_key)
                .key(&self.owner)
                .key(&self.latest)
                .key(&self.warning)
                .key(format!("{}stream-model-status:{}", self.prefix, spec.id))
                .key(format!(
                    "{}stream-model-challenge:{}",
                    self.prefix, model.key
                ))
                .arg(self.generation)
                .arg(&self.token)
                .arg(payload)
                .arg(if warn { 1 } else { 0 })
                .arg(&status)
                .arg(if spec.output.client_overlay { 1 } else { 0 })
                .arg(if working && !hazard.warnings.is_empty() {
                    1
                } else {
                    0
                })
                .invoke_async(&mut self.connection),
        )
        .await??;
        if accepted == 1 && warn {
            self.last_warning = timestamp / 1000;
        }
        if accepted == 1 && !status.is_empty() {
            self.last_status = timestamp / 1000;
        }
        Ok(accepted == 1)
    }
    pub async fn close(mut self) -> Result<()> {
        let _:i32=tokio::time::timeout(Duration::from_secs(3),redis::Script::new("if redis.call('GET',KEYS[1])==ARGV[1] then redis.call('DEL',KEYS[1],KEYS[2],KEYS[3]);return 1 end;return 0")
            .key(&self.owner).key(&self.latest).key(&self.warning).arg(&self.token).invoke_async(&mut self.connection)).await??;
        Ok(())
    }
}

/// Latest visual state only. Alarm evidence continues through the durable outbox.
#[derive(Clone)]
pub struct Snapshot {
    pub sequence: u64,
    pub generation: u64,
    pub timestamp: i64,
    pub tracks: std::sync::Arc<Vec<Track>>,
    pub hazard: std::sync::Arc<HazardResult>,
    pub working: bool,
    pub source_pts: Option<f64>,
}
#[derive(Default)]
pub struct Stats {
    pub offered: std::sync::atomic::AtomicU64,
    pub sent: std::sync::atomic::AtomicU64,
    pub superseded: std::sync::atomic::AtomicU64,
    pub errors: std::sync::atomic::AtomicU64,
}
pub struct Actor {
    tx: tokio::sync::watch::Sender<Option<(u64, Snapshot)>>,
    pub stats: std::sync::Arc<Stats>,
    task: tokio::task::JoinHandle<()>,
}
impl Actor {
    pub fn new(config: Redis, spec: Stream, model: Model) -> Self {
        use std::sync::{Arc, atomic::Ordering};
        let (tx, mut rx) = tokio::sync::watch::channel::<Option<(u64, Snapshot)>>(None);
        let stats = Arc::new(Stats::default());
        let counters = stats.clone();
        let task = tokio::spawn(async move {
            let mut sink = None;
            let mut previous = 0;
            while rx.changed().await.is_ok() {
                let Some((revision, snapshot)) = rx.borrow_and_update().clone() else {
                    continue;
                };
                counters
                    .superseded
                    .fetch_add(revision - previous - 1, Ordering::Relaxed);
                previous = revision;
                let operation = async {
                    if sink.is_none() {
                        sink = Some(Sink::new(&config, &spec).await?);
                    }
                    sink.as_mut()
                        .unwrap()
                        .publish_source(
                            &spec,
                            snapshot.sequence,
                            snapshot.generation,
                            snapshot.timestamp,
                            &snapshot.tracks,
                            &snapshot.hazard,
                            snapshot.working,
                            &model,
                            snapshot.source_pts,
                        )
                        .await
                };
                match tokio::time::timeout(Duration::from_secs(5), operation).await {
                    Ok(Ok(true)) => {
                        counters.sent.fetch_add(1, Ordering::Relaxed);
                    }
                    _ => {
                        counters.errors.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
            if let Some(sink) = sink {
                let _ = sink.close().await;
            }
        });
        Self { tx, stats, task }
    }
    pub fn offer(&self, snapshot: Snapshot) {
        use std::sync::atomic::Ordering;
        let revision = self.stats.offered.fetch_add(1, Ordering::Relaxed) + 1;
        self.tx.send_replace(Some((revision, snapshot)));
    }
    pub async fn finish(self) {
        drop(self.tx);
        let _ = self.task.await;
    }
}
