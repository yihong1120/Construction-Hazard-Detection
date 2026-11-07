# Evidence outbox operations

Each camera has one SQLite outbox under `VIOLATION_OUTBOX_DIRECTORY`. Pending
events and failed archives have separate capacity accounting. Archived failures
do not occupy pending-event slots. Existing failed rows move into the archive
on first open; existing pending rows, upload IDs and leases are preserved.

Defaults retain up to 1,000 failed events for seven days, with a 64 MiB archive
payload budget. Configure nested fields with
`VIOLATION_OUTBOX_ARCHIVE__MAX_EVENTS`, `VIOLATION_OUTBOX_ARCHIVE__MAX_BYTES` and
`VIOLATION_OUTBOX_ARCHIVE__SECONDS`. Expired and oldest excess failures are removed
during writes; explicit purge also works while a queue is idle. The combined
pending/archive payload remains within the per-outbox `VIOLATION_OUTBOX_MAX_BYTES`
budget; new pending evidence can evict old archives to stay within this budget.
SQLite page/index/WAL overhead is additional to this logical payload limit.
Archive eviction removes the archived evidence, so retain operational exports
separately when longer retention is required.

Use the same environment settings as the delivery process:

```bash
python -m scripts.violation_outbox /private/outbox.sqlite3 list
python -m scripts.violation_outbox /private/outbox.sqlite3 show --event-id EVENT_ID
python -m scripts.violation_outbox /private/outbox.sqlite3 requeue --event-id EVENT_ID
python -m scripts.violation_outbox /private/outbox.sqlite3 purge --older-than-days 7
```

`show` returns metadata and image size without exposing image bytes. Requeue is
atomic and respects pending capacity; an event that already uploaded evidence
continues notification delivery using its saved upload ID.

Worker pool capacities (`YOLO_WORKER_CAMERAS_PER_ENGINE`,
`YOLO_WORKER_QUEUE_SIZE`, `YOLO_WORKER_RESULT_QUEUE_SIZE` and per-model limits)
must be positive. Invalid values stop startup/reconfiguration before replacing
the active topology. Process shutdown uses bounded concurrency with a shared
grace period and continues cleanup through task cancellation.

## Directory scheduling and budget

All current writers coordinate through a private directory lock and
`.outbox-index.db`. The dispatcher reads indexed due queues instead of opening
every camera database each poll. Legacy files are discovered at startup and at
`VIOLATION_OUTBOX_DISCOVERY_SECONDS` intervals (default 60 seconds). Successful
queue transactions update their pending/archive byte counts and next due time.
A durable dirty marker forces rediscovery after an interrupted accounting update.

`VIOLATION_OUTBOX_DIRECTORY_MAX_BYTES` defaults to 2 GiB of combined pending and
archived payload across camera queues. `VIOLATION_OUTBOX_MIN_FREE_BYTES` reserves
128 MiB of filesystem space before accepting evidence. The logical directory
budget excludes SQLite/index/WAL allocation overhead. Capacity exhaustion rejects
new submissions and preserves existing pending events.

Scheduler capacity and cached queue objects are bounded by
`VIOLATION_OUTBOX_SCHEDULER_CAPACITY` (128) and `VIOLATION_OUTBOX_CACHE_SIZE` (64).
Disabled/deleted cameras remain eligible while their pending events are due.
Idle maintenance rotates across queues, expires archives, and incrementally
vacuums newly created databases. Empty databases are removed after
`VIOLATION_OUTBOX_IDLE_SECONDS` (300), only while holding the directory lock and
an exclusive writer lease. Active producers hold a shared OS file lease until
stream shutdown; process death releases it automatically. A surviving queue
object can recreate an already reclaimed empty database safely.

Upgrade/restart all producer processes sharing the directory before enabling
this directory protocol: older releases do not participate in its budget or OS
writer leases. Retain archives using the existing policy; pending data is never
removed by idle reclamation. SQLite index/lock files remain directory metadata.

Queue counts and payload bytes now use SQLite triggers, so each mutation updates
its statistics in the same queue transaction. Directory budget updates use an
incremental total in the index transaction. Normal puts/claims/retries no longer
re-scan payload rows or sum all camera queues. The dirty marker names the one
queue requiring reconciliation after interruption. Initial index upgrade and
legacy unnamed dirty markers require a full reconciliation before reservations.

Periodic discovery walks the directory outside the global lock and reconciles
one queue per lock acquisition. Idle maintenance likewise releases the lock
between queues. `VIOLATION_OUTBOX_MAINTENANCE_BATCH_SIZE` defaults to 16 entries,
`MAINTENANCE_BUDGET_SECONDS` to 0.1 seconds and `ARCHIVE_CLEANUP_BATCH_SIZE` to 128
expired archived events per camera (all under the `VIOLATION_OUTBOX_` prefix). Vacuum
work remains limited to 128 pages per visit. These are cooperative time budgets:
one SQLite operation can exceed the time budget before control returns. A scan
can span multiple polling cycles; the discovery interval begins after it ends.

Unreadable databases are quarantined for accounting: healthy queued events can
continue delivery, while new reservations fail closed until unknown payload
sizes are repaired or the offending file is removed. Statistics and index schema
upgrades occur lazily under the directory lock; no manual SQLite migration is
needed. Upgrade/restart all processes sharing this directory together, including
older dispatchers and CLI tools, because older index writers do not maintain the
new incremental budget.

Global failed-event retention also has a directory budget:
`VIOLATION_OUTBOX_DIRECTORY_ARCHIVE_MAX_BYTES` defaults to one quarter of
`VIOLATION_OUTBOX_DIRECTORY_MAX_BYTES` and may be set to zero. Under archive or
total-budget pressure, the oldest archived events across all cameras are
removed first, including unexpired archives. Pending or leased events are never
evicted to admit a new event. Archive retention is therefore a maximum, not a
guarantee when the global budget is exhausted. Cleanup is bounded by the existing
maintenance batch size and time budget; a busy writer may need to retry admission
when one maintenance pass cannot reclaim enough space.

The directory index upgrades to schema version 3 and local queue counters to
version 1 automatically. Stop and upgrade all producers, dispatchers and CLI
processes sharing a directory together. Do not mix older writers with upgraded
counters. A durable marker tracks every queue changed during global eviction;
restart repairs accounting if a process stops between a queue commit and its
index commit. SQLite may retain freed pages for reuse; the existing free-disk
reserve and maintenance/vacuum policy still apply.

Admission now checks duplicate IDs, per-camera pending capacity, irreducible
pending bytes across the directory, quarantine state, and free-disk reserve
before global archive eviction. These rejected requests preserve other cameras'
archives. Once a feasible admission starts reclamation, global cleanup remains
bounded; a later I/O failure or insufficient cleanup budget can still require a
retry after eligible archives have been reclaimed. Pending events are retained.
