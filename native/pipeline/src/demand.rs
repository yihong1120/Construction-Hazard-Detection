//! Existing playback leases control outputs, never inference or event storage.
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU64, Ordering},
};
use std::time::Duration;
use tokio_util::sync::CancellationToken;
pub struct Gate {
    state: AtomicU64,
    pub changed: Arc<tokio::sync::Notify>,
    filter: Option<Box<dyn Fn() -> bool + Send + Sync>>,
}
impl Gate {
    pub fn new(active: bool) -> Arc<Self> {
        Arc::new(Self {
            state: AtomicU64::new(u64::from(active)),
            changed: Arc::default(),
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
pub fn spawn(
    config: crate::config::Redis,
    keys: Vec<(String, Arc<Gate>)>,
    ready: Vec<(String, Arc<AtomicBool>)>,
    stop: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut connection = None;
        let token = uuid::Uuid::new_v4().to_string();
        let demand_keys: Vec<_> = keys
            .iter()
            .map(|(key, _)| format!("{}{key}", config.prefix))
            .collect();
        let ready_keys: Vec<_> = ready
            .iter()
            .map(|(key, _)| format!("{}{key}", config.prefix))
            .collect();
        let script = redis::Script::new(
            "for i,key in ipairs(KEYS) do
                local old=redis.call('GET',key)
                if ARGV[i+1]=='1' then
                    if not old or old==ARGV[1] then redis.call('SET',key,ARGV[1],'EX',3) end
                elseif old==ARGV[1] then
                    redis.call('DEL',key)
                end
            end
            return 1",
        );
        let mut previously_ready = vec![false; ready.len()];
        let mut tick = tokio::time::interval(Duration::from_millis(250));
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {_=stop.cancelled()=>break,_=tick.tick()=>()}
            let operation = async {
                if connection.is_none() {
                    let client = redis::Client::open(std::env::var(&config.url_env)?)?;
                    connection = Some(client.get_multiplexed_async_connection().await?);
                }
                let connection = connection.as_mut().expect("Redis connection");
                let values: Vec<Option<String>> = redis::cmd("MGET")
                    .arg(&demand_keys)
                    .query_async(connection)
                    .await?;
                for ((_, gate), value) in keys.iter().zip(values) {
                    gate.set(value.is_some());
                }

                let mut invocation = script.prepare_invoke();
                invocation.arg(&token);
                let mut changed = false;
                for ((key, (_, flag)), previous) in
                    ready_keys.iter().zip(&ready).zip(&mut previously_ready)
                {
                    let active = flag.load(Ordering::Acquire);
                    if active || *previous {
                        invocation.key(key).arg(u8::from(active));
                        changed = true;
                    }
                    // If the command fails, any old marker still expires after three seconds.
                    *previous = active;
                }
                if changed {
                    invocation.invoke_async::<i32>(connection).await?;
                }
                anyhow::Ok(())
            };
            if !matches!(
                tokio::time::timeout(Duration::from_secs(1), operation).await,
                Ok(Ok(()))
            ) {
                connection = None;
                for (_, gate) in &keys {
                    gate.set(false);
                }
            }
        }
        for (_, gate) in keys {
            gate.set(false);
        }
    })
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
