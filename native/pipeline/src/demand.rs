//! Existing playback leases control outputs, never inference or event storage.
use std::collections::HashMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;
use tokio::sync::Notify;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

pub struct Gate {
    state: AtomicU64,
    pub changed: Arc<Notify>,
    filter: Option<Box<dyn Fn() -> bool + Send + Sync>>,
}
impl Gate {
    pub fn new(active: bool) -> Arc<Self> {
        Self::observed(active, Arc::default())
    }
    pub fn observed(active: bool, changed: Arc<Notify>) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU64::new(u64::from(active)),
            changed,
            filter: None,
        })
    }
    pub fn filtered(
        parent: Arc<Self>,
        filter: impl Fn() -> bool + Send + Sync + 'static,
    ) -> Arc<Self> {
        let changed = parent.changed.clone();
        Arc::new(Self {
            state: AtomicU64::new(1),
            changed,
            filter: Some(Box::new(move || parent.active() && filter())),
        })
    }
    pub fn active(&self) -> bool {
        self.version() % 2 == 1 && self.filter.as_ref().is_none_or(|f| f())
    }
    pub fn version(&self) -> u64 {
        self.state.load(Ordering::Acquire)
    }
    pub fn set(&self, active: bool) {
        let updated = self
            .state
            .fetch_update(Ordering::AcqRel, Ordering::Acquire, |old| {
                ((old % 2 == 1) != active).then_some(old + 1)
            });
        if updated.is_ok() {
            self.changed.notify_waiters();
        }
    }
}

struct Ready {
    key: String,
    flag: Arc<AtomicBool>,
    previously_ready: AtomicBool,
}

struct Camera {
    owner: String,
    gates: Vec<(String, Arc<Gate>)>,
    ready: Vec<Ready>,
}

#[derive(Default)]
struct Registry {
    cameras: Mutex<HashMap<Uuid, Arc<Camera>>>,
    changed: Notify,
}

#[derive(Clone)]
pub struct Demand {
    prefix: Arc<str>,
    registry: Arc<Registry>,
}

pub struct Registration {
    registry: Arc<Registry>,
    id: Uuid,
}

impl Drop for Registration {
    fn drop(&mut self) {
        let mut cameras = self.registry.cameras.lock().unwrap();
        if let Some(camera) = cameras.remove(&self.id) {
            for (_, gate) in &camera.gates {
                gate.set(false);
            }
        }
    }
}

impl Demand {
    pub fn spawn(
        config: crate::config::Redis,
        stop: CancellationToken,
    ) -> (Self, tokio::task::JoinHandle<()>) {
        let demand = Self {
            prefix: config.prefix.into(),
            registry: Arc::default(),
        };
        let task = tokio::spawn(demand.clone().poll(config.url_env, stop));
        (demand, task)
    }

    pub fn register(
        &self,
        gates: Vec<(String, Arc<Gate>)>,
        ready: Vec<(String, Arc<AtomicBool>)>,
    ) -> Registration {
        let id = Uuid::new_v4();
        let qualify = |key| format!("{}{key}", self.prefix);
        let camera = Camera {
            owner: id.to_string(),
            gates: gates
                .into_iter()
                .map(|(key, gate)| (qualify(key), gate))
                .collect(),
            ready: ready
                .into_iter()
                .map(|(key, flag)| Ready {
                    key: qualify(key),
                    flag,
                    previously_ready: AtomicBool::new(false),
                })
                .collect(),
        };
        self.registry
            .cameras
            .lock()
            .unwrap()
            .insert(id, Arc::new(camera));
        self.registry.changed.notify_one();
        Registration {
            registry: self.registry.clone(),
            id,
        }
    }

    fn close_gates(&self) {
        for camera in self.registry.cameras.lock().unwrap().values() {
            for (_, gate) in &camera.gates {
                gate.set(false);
            }
        }
    }

    async fn poll(self, url_env: String, stop: CancellationToken) {
        let mut connection = None;
        let script = redis::Script::new(
            "for i,key in ipairs(KEYS) do
                local owner=ARGV[2*i-1]
                local old=redis.pcall('GET',key)
                if ARGV[2*i]=='1' then
                    if not old or old==owner then redis.call('SET',key,owner,'EX',3) end
                elseif old==owner then
                    redis.call('DEL',key)
                end
            end
            return 1",
        );
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            if self.registry.cameras.lock().unwrap().is_empty() {
                // Release the Redis connection while there are no on-demand cameras.
                connection = None;
                tokio::select! {
                    _ = stop.cancelled() => break,
                    _ = self.registry.changed.notified() => tick.reset_immediately(),
                }
            }
            tokio::select! {
                _ = stop.cancelled() => break,
                _ = tick.tick() => (),
            }
            let cameras: Vec<_> = self
                .registry
                .cameras
                .lock()
                .unwrap()
                .iter()
                .map(|(&id, camera)| (id, camera.clone()))
                .collect();
            if cameras.is_empty() {
                continue;
            }
            let operation = async {
                if connection.is_none() {
                    let client = redis::Client::open(std::env::var(&url_env)?)?;
                    connection = Some(client.get_multiplexed_async_connection().await?);
                }
                let connection = connection.as_mut().expect("Redis connection");
                let mut query = redis::cmd("MGET");
                for (_, camera) in &cameras {
                    for (key, _) in &camera.gates {
                        query.arg(key);
                    }
                }
                let values: Vec<redis::Value> = query.query_async(connection).await?;

                let mut invocation = script.prepare_invoke();
                let mut changed = false;
                {
                    // Apply a reply only to owners still registered. The same lock
                    // fences gate closure while a camera is removed or reloaded.
                    let registered = self.registry.cameras.lock().unwrap();
                    let mut values = values.into_iter();
                    for (id, camera) in &cameras {
                        let active = values.by_ref().take(camera.gates.len());
                        if !registered.contains_key(id) {
                            active.for_each(drop);
                            continue;
                        }
                        for ((_, gate), value) in camera.gates.iter().zip(active) {
                            gate.set(!matches!(value, redis::Value::Nil));
                        }
                        for ready in &camera.ready {
                            let active = ready.flag.load(Ordering::Acquire);
                            let previous = ready.previously_ready.swap(active, Ordering::Relaxed);
                            if active || previous {
                                invocation
                                    .key(&ready.key)
                                    .arg(&camera.owner)
                                    .arg(u8::from(active));
                                changed = true;
                            }
                        }
                    }
                }
                if changed {
                    invocation.invoke_async::<i32>(connection).await?;
                }
                anyhow::Ok(())
            };
            let result = tokio::select! {
                _ = stop.cancelled() => break,
                result = tokio::time::timeout(Duration::from_secs(1), operation) => result,
            };
            if !matches!(result, Ok(Ok(()))) {
                connection = None;
                self.close_gates();
            }
        }
        self.close_gates();
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn demand_edges_advance_generation_once() {
        let g = Gate::new(false);
        assert!(!g.active());
        g.set(true);
        let n = g.version();
        g.set(true);
        assert_eq!(g.version(), n);
        g.set(false);
        assert!(!g.active());
        assert_eq!(g.version(), n + 1);
    }
}
