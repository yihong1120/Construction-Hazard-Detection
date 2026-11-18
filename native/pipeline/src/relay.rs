//! Bounded encoded packet fanout off the existing decoder input connection.
use crate::demand::Gate;
use construction_media::{EncodedPacket, Remuxer};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;
#[derive(Default)]
pub struct Stats {
    pub packets: AtomicU64,
    pub dropped: AtomicU64,
    pub errors: AtomicU64,
    pub fallback: AtomicU64,
}
#[derive(Clone)]
pub struct Sender {
    tx: mpsc::Sender<(u64, u64, EncodedPacket)>,
    pub gate: Arc<Gate>,
    pub mode: Arc<AtomicU8>,
    epoch: Arc<AtomicU64>,
    waiting: Arc<AtomicBool>,
    pub stats: Arc<Stats>,
    width: u32,
    height: u32,
    fps: u32,
}
impl Sender {
    pub fn invalidate(&self) {
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.waiting.store(true, Ordering::Release);
        self.mode.store(0, Ordering::Release);
        self.gate.changed.notify_waiters();
    }
    pub fn gap(&self) {
        self.stats.dropped.fetch_add(1, Ordering::Relaxed);
        self.epoch.fetch_add(1, Ordering::AcqRel);
        self.waiting.store(true, Ordering::Release);
    }
    pub fn offer(&self, packet: EncodedPacket) {
        if !self.gate.active() {
            return;
        }
        if packet.width != self.width as i32
            || packet.height != self.height as i32
            || !packet.fps.is_finite()
            || (packet.fps - self.fps as f64).abs() > 0.1
        {
            if self.mode.swap(2, Ordering::AcqRel) != 2 {
                self.stats.fallback.fetch_add(1, Ordering::Relaxed);
                self.gate.changed.notify_waiters();
            }
            return;
        }
        if self.mode.swap(1, Ordering::AcqRel) != 1 {
            self.gate.changed.notify_waiters();
        }
        if self.waiting.load(Ordering::Acquire) && !packet.key {
            return;
        }
        self.waiting.store(false, Ordering::Release);
        let message = (
            self.gate.version(),
            self.epoch.load(Ordering::Acquire),
            packet,
        );
        if self.tx.try_send(message).is_err() {
            self.stats.dropped.fetch_add(1, Ordering::Relaxed);
            self.epoch.fetch_add(1, Ordering::AcqRel);
            self.waiting.store(true, Ordering::Release);
        }
    }
}
pub struct Relay {
    pub sender: Sender,
    pub task: tokio::task::JoinHandle<anyhow::Result<()>>,
}
pub fn spawn(
    target: String,
    width: u32,
    height: u32,
    fps: u32,
    gate: Arc<Gate>,
    stop: CancellationToken,
) -> Relay {
    let (tx, mut rx) = mpsc::channel::<(u64, u64, EncodedPacket)>(8);
    let sender = Sender {
        tx,
        gate,
        mode: Arc::new(AtomicU8::new(0)),
        epoch: Arc::new(AtomicU64::new(0)),
        waiting: Arc::new(AtomicBool::new(true)),
        stats: Arc::default(),
        width,
        height,
        fps,
    };
    let owner = sender.clone();
    let task = tokio::spawn(async move {
        loop {
            let first = tokio::select! {_=stop.cancelled()=>return Ok(()),p=rx.recv()=>match p {Some(p)=>p,None=>return Ok(())}};
            if !owner.gate.active()
                || owner.mode.load(Ordering::Acquire) == 2
                || first.0 != owner.gate.version()
                || first.1 != owner.epoch.load(Ordering::Acquire)
                || !first.2.key
            {
                continue;
            }
            let (owner, stop, target) = (owner.clone(), stop.clone(), target.clone());
            let handle = tokio::runtime::Handle::current();
            rx=crate::executor::dedicated("packet-remux".into(), move || {
        let mut mux: Option<Remuxer> = None;
        let mut previous = None;
        let mut initial=Some(first);
        loop {
            let changed=owner.gate.changed.notified();tokio::pin!(changed);changed.as_mut().enable();
            if stop.is_cancelled() { break; }
            if !owner.gate.active() || owner.mode.load(Ordering::Acquire)==2 {
                owner.waiting.store(true,Ordering::Release);
                while rx.try_recv().is_ok() {}
                break;
            }
            let message=if let Some(p)=initial.take() { Some(p) } else {
                handle.block_on(async {tokio::select! {_=stop.cancelled()=>None,_=changed.as_mut()=>None,p=rx.recv()=>p}})
            };
            let Some((generation,epoch,packet))=message else {
                if rx.is_closed() || stop.is_cancelled() { break; }
                continue;
            };
            if generation != owner.gate.version() || epoch != owner.epoch.load(Ordering::Acquire) {
                continue;
            }
            if previous != Some((generation, epoch)) {
                mux = None;
                previous = Some((generation, epoch));
            }
            if mux.is_none() {
                if !packet.key {
                    continue;
                }
                match Remuxer::open(&packet, &target) {
                    Ok(value) => mux = Some(value),
                    Err(_) => {
                        owner.stats.errors.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }
                }
            }
            if mux.as_mut().unwrap().write(&packet).is_err() {
                mux = None;
                owner.stats.errors.fetch_add(1, Ordering::Relaxed);
                owner.waiting.store(true, Ordering::Release);
                owner.epoch.fetch_add(1, Ordering::AcqRel);
            } else {
                owner.stats.packets.fetch_add(1, Ordering::Relaxed);
            }
        }
        anyhow::Ok(rx)
            }).await??;
        }
    });
    Relay { sender, task }
}
