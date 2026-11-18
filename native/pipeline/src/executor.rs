//! Bounded CPU work and dedicated media owners; long-lived codec calls never
//! occupy Tokio's I/O/blocking pool.
use anyhow::{Context, Result, ensure};
use std::{
    collections::VecDeque,
    sync::{Arc, Condvar, Mutex, MutexGuard},
};
use tokio::sync::{Notify, oneshot};

type Job = Box<dyn FnOnce() + Send + 'static>;

struct State {
    jobs: VecDeque<Job>,
    workers: usize,
    running: usize,
    outstanding: usize,
    waiting: usize,
    closed: bool,
}

struct Queue {
    state: Mutex<State>,
    has_work: Condvar,
    // Keep analysis and evidence requests ahead of nonblocking preview admission.
    admission: tokio::sync::Mutex<()>,
    slot_available: Notify,
}

impl Queue {
    fn lock(&self) -> MutexGuard<'_, State> {
        self.state.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn reserve(self: &Arc<Self>, state: &mut State) -> Option<CpuPermit> {
        if state.outstanding >= state.workers * 2 {
            return None;
        }
        state.outstanding += 1;
        Some(CpuPermit(self.clone()))
    }

    fn next_job(&self) -> Option<Job> {
        let mut state = self.lock();
        loop {
            if (state.closed || state.running < state.workers)
                && let Some(job) = state.jobs.pop_front()
            {
                state.running += 1;
                return Some(job);
            }
            if state.closed {
                return None;
            }
            state = self.has_work.wait(state).unwrap_or_else(|e| e.into_inner());
        }
    }

    fn run(&self) {
        while let Some(job) = self.next_job() {
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(job));
            let has_work = {
                let mut state = self.lock();
                state.running -= 1;
                !state.jobs.is_empty()
            };
            if has_work {
                self.has_work.notify_one();
            }
        }
    }
}

pub struct CpuPermit(Arc<Queue>);

impl Drop for CpuPermit {
    fn drop(&mut self) {
        self.0.lock().outstanding -= 1;
        self.0.slot_available.notify_one();
    }
}

struct AdmissionWaiter<'a>(&'a Queue);

impl Drop for AdmissionWaiter<'_> {
    fn drop(&mut self) {
        self.0.lock().waiting -= 1;
    }
}

struct Inner {
    queue: Arc<Queue>,
    threads: Vec<std::thread::JoinHandle<()>>,
}
impl Drop for Inner {
    fn drop(&mut self) {
        self.queue.lock().closed = true;
        self.queue.has_work.notify_all();
        let current = std::thread::current().id();
        for thread in self.threads.drain(..) {
            if thread.thread().id() != current {
                let _ = thread.join();
            }
        }
    }
}
#[derive(Clone)]
pub struct Cpu(Arc<Inner>);
impl Cpu {
    pub fn new(workers: usize) -> Result<Self> {
        let workers = workers.clamp(1, 64);
        Self::elastic(workers, workers)
    }

    /// Allocate a fixed set of threads with a separately adjustable work budget.
    pub fn elastic(initial: usize, max: usize) -> Result<Self> {
        ensure!(
            (1..=64).contains(&max) && (1..=max).contains(&initial),
            "CPU worker limits must satisfy 1 <= initial <= max <= 64"
        );
        let mut inner = Inner {
            queue: Arc::new(Queue {
                state: Mutex::new(State {
                    jobs: VecDeque::with_capacity(max * 2),
                    workers: initial,
                    running: 0,
                    outstanding: 0,
                    waiting: 0,
                    closed: false,
                }),
                has_work: Condvar::new(),
                admission: tokio::sync::Mutex::new(()),
                slot_available: Notify::new(),
            }),
            threads: Vec::with_capacity(max),
        };
        for index in 0..max {
            let queue = inner.queue.clone();
            inner.threads.push(
                std::thread::Builder::new()
                    .name(format!("cpu-{index}"))
                    .spawn(move || queue.run())?,
            );
        }
        Ok(Self(Arc::new(inner)))
    }

    /// Lowering the limit allows running jobs to finish and retains queued jobs.
    pub fn set_workers(&self, workers: usize) -> Result<()> {
        ensure!(
            (1..=self.0.threads.len()).contains(&workers),
            "CPU worker limit must be between 1 and {}",
            self.0.threads.len()
        );
        let grew = {
            let mut state = self.0.queue.lock();
            let grew = workers > state.workers;
            state.workers = workers;
            grew
        };
        if grew {
            self.0.queue.has_work.notify_all();
            self.0.queue.slot_available.notify_one();
        }
        Ok(())
    }

    pub fn workers(&self) -> usize {
        self.0.queue.lock().workers
    }

    pub fn try_permit(&self) -> Option<CpuPermit> {
        let queue = &self.0.queue;
        let mut state = queue.lock();
        if state.waiting > 0 {
            return None;
        }
        queue.reserve(&mut state)
    }

    pub async fn permit(&self) -> Result<CpuPermit> {
        let queue = &self.0.queue;
        let _waiting = {
            let mut state = queue.lock();
            if state.waiting == 0
                && let Some(permit) = queue.reserve(&mut state)
            {
                return Ok(permit);
            }
            state.waiting += 1;
            AdmissionWaiter(queue)
        };
        let _admission = queue.admission.lock().await;
        loop {
            if let Some(permit) = queue.reserve(&mut queue.lock()) {
                return Ok(permit);
            }
            // There is one waiter on this notification. A release between reserve
            // and await leaves a notification for that waiter to consume.
            queue.slot_available.notified().await;
        }
    }
    pub async fn run<R: Send + 'static>(
        &self,
        job: impl FnOnce() -> R + Send + 'static,
    ) -> Result<R> {
        let permit = self.permit().await?;
        self.run_permitted(permit, job).await
    }
    pub async fn run_permitted<R: Send + 'static>(
        &self,
        permit: CpuPermit,
        job: impl FnOnce() -> R + Send + 'static,
    ) -> Result<R> {
        ensure!(
            Arc::ptr_eq(&permit.0, &self.0.queue),
            "CPU permit belongs to another executor"
        );
        let (tx, rx) = oneshot::channel();
        self.0.queue.lock().jobs.push_back(Box::new(move || {
            let _permit = permit;
            let value = job();
            let _ = tx.send(value);
        }));
        self.0.queue.has_work.notify_one();
        rx.await.context("CPU job panicked")
    }
}
pub fn dedicated<R: Send + 'static>(
    name: String,
    run: impl FnOnce() -> Result<R> + Send + 'static,
) -> tokio::task::JoinHandle<Result<R>> {
    let (tx, rx) = oneshot::channel();
    let thread = std::thread::Builder::new().name(name).spawn(move || {
        let result = run();
        let _ = tx.send(result);
    });
    tokio::spawn(async move {
        let thread = thread?;
        let result = rx.await;
        thread
            .join()
            .map_err(|_| anyhow::anyhow!("Media owner panicked"))?;
        result.context("Media owner disappeared")?
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{
            atomic::{AtomicUsize, Ordering},
            mpsc,
        },
        time::Duration,
    };
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cpu_budget_is_bounded_and_recovers_after_a_panicked_job() {
        let cpu = Cpu::new(2).unwrap();
        let active = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut tasks = tokio::task::JoinSet::new();
        for _ in 0..24 {
            let (cpu, active, peak) = (cpu.clone(), active.clone(), peak.clone());
            tasks.spawn(async move {
                cpu.run(move || {
                    let count = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(count, Ordering::SeqCst);
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    active.fetch_sub(1, Ordering::SeqCst);
                })
                .await
                .unwrap()
            });
        }
        while let Some(result) = tasks.join_next().await {
            result.unwrap();
        }
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert!(cpu.run(|| panic!("fixture panic")).await.is_err());
        assert_eq!(cpu.run(|| 42).await.unwrap(), 42);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn elastic_limits_preserve_queued_and_running_jobs() {
        let cpu = Cpu::elastic(3, 4).unwrap();
        {
            let _admission = cpu.0.queue.admission.lock().await;
            assert!(cpu.try_permit().is_some());
        }
        let (started, mut starts) = tokio::sync::mpsc::unbounded_channel();
        let mut releases = Vec::new();
        let mut tasks = tokio::task::JoinSet::new();
        for index in 0..6 {
            let permit = cpu.try_permit().unwrap();
            let (release, wait) = mpsc::channel();
            releases.push(release);
            let (cpu, started) = (cpu.clone(), started.clone());
            tasks.spawn(async move {
                cpu.run_permitted(permit, move || {
                    started.send(index).unwrap();
                    let _ = wait.recv();
                    index
                })
                .await
                .unwrap()
            });
        }
        assert!(cpu.try_permit().is_none());
        let timeout = Duration::from_secs(3);
        let mut running = Vec::new();
        for _ in 0..3 {
            running.push(
                tokio::time::timeout(timeout, starts.recv())
                    .await
                    .unwrap()
                    .unwrap(),
            );
        }

        cpu.set_workers(1).unwrap();
        assert_eq!(cpu.workers(), 1);
        for &index in &running[..2] {
            releases[index].send(()).unwrap();
        }
        let interval = Duration::from_millis(30);
        assert!(tokio::time::timeout(interval, starts.recv()).await.is_err());
        assert!(cpu.try_permit().is_none());
        releases[running[2]].send(()).unwrap();
        let next = tokio::time::timeout(timeout, starts.recv())
            .await
            .unwrap()
            .unwrap();

        let waiting = cpu.permit();
        tokio::pin!(waiting);
        assert!(tokio::time::timeout(interval, &mut waiting).await.is_err());
        cpu.set_workers(4).unwrap();
        assert_eq!(cpu.workers(), 4);
        assert!(cpu.set_workers(5).is_err());
        assert!(cpu.try_permit().is_none());
        let permit = tokio::time::timeout(timeout, &mut waiting)
            .await
            .unwrap()
            .unwrap();
        for _ in 0..2 {
            let index = tokio::time::timeout(timeout, starts.recv())
                .await
                .unwrap()
                .unwrap();
            releases[index].send(()).unwrap();
        }
        assert_eq!(cpu.run_permitted(permit, || 42).await.unwrap(), 42);
        releases[next].send(()).unwrap();

        let mut completed = Vec::new();
        while let Some(result) = tokio::time::timeout(timeout, tasks.join_next())
            .await
            .unwrap()
        {
            completed.push(result.unwrap());
        }
        completed.sort_unstable();
        assert_eq!(completed, (0..6).collect::<Vec<_>>());
    }
}
