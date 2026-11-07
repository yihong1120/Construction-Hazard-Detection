# Evidence identity migration

Run these scripts through the deployment database workflow before starting the
new application. Application startup never applies migrations.

The manifest also includes `20261101_01_model_publications.sql`, which adds the
durable publication journal required by the model publication CLI. It does not
change existing catalog entries or start a model publication.

1. Apply `20261030_01_evidence_identity.sql`. It adds nullable IDs, constraints
   and indexes while leaving the legacy name foreign key in place. Schedule
   index creation for a maintenance window on large tables; ordinary index
   creation blocks writers until completion. Lock acquisition is bounded.
2. If legacy writers remain, apply
   `20261031_01_legacy_evidence_site_dual_write.sql` after step 1. Its trigger
   fills a missing `site_id` using the exact current site name and rejects an
   unknown name. Explicit IDs and historical snapshots are preserved. The
   function binds the actual table schemas, including installations where
   evidence and notifications use different schemas.
3. Deploy the application that writes `site_id` alongside the historical name
   and accepts evidence event IDs. Both old rows without IDs and new rows are
   readable. Names are a display snapshot; permissions use the current site's
   ID. Media filenames and public media URLs remain compatible.
4. Repeat `20261030_02_backfill_evidence_sites.sql` until `remaining` is zero.
   Investigate unmatched site names; never guess the owner of orphaned evidence.
5. Verify uploads, queries and site cleanup, then apply
   `20261030_03_contract_evidence_sites.sql`. This permits later site renames
   without changing evidence snapshots. Retire old writers before allowing
   site renames; if the rolling trigger is installed, schema validation can
   finish while they continue to use current site names. IDs stay nullable for the rolling
   transition; enforce NOT NULL in a subsequent migration after legacy writers
   are permanently retired.

Before step 5, rollback means redeploying the previous application; the added
nullable columns and indexes can remain. After step 5, restore the legacy name
foreign key only if every snapshot name still resolves to its original site.
Do not reverse renamed snapshots automatically.

The stream worker retains unsent events in `VIOLATION_OUTBOX_DIRECTORY`
(default `outputs/event-outbox`). Docker Compose mounts a persistent named volume at that path, initialised
for the application user with private permissions. For native workers, retain
that directory on persistent local storage. Capacity defaults to 100
events / 256 MiB per camera. A full outbox reports an error and leaves the
notification clock unchanged. Pending events resume after a worker restart;
FCM retries reuse the uploaded evidence ID. An upload retry uses the same UUID,
so an uncertain HTTP response does not create another evidence row.

Evidence storage deduplicates by UUID and notification-centre records by
recipient and violation ID. Push delivery is at least once: a provider timeout
after accepting a push can still produce a repeated device notification.

## Migration runner

Use the same protected environment/configuration file as the target service.
The database name is required and checked against the connected server. Planning
is the default; `--check` also exits unsuccessfully when migrations remain.

```bash
python -m deploy.migrations.run --expected-database DATABASE --config /private/service.json
python -m deploy.migrations.run --expected-database DATABASE --config /private/service.json --apply --backup /private/new-backup-directory --backup-verifier-config /private/isolated-verifier.json
python -m deploy.migrations.run --expected-database DATABASE --config /private/service.json --check
```

A config may contain an `environment` object or the environment fields directly.
Without `--config`, `DATABASE_URL` comes from the process environment; repository
`.env` files are never loaded implicitly. `--schema` may be repeated to select an
explicit search path. PostgreSQL client tools must support the server version
when requesting a backup. One private archive contains the complete database.
An exported PostgreSQL snapshot keeps the archive and table-count receipt on the
same baseline. The migration lock covers planning, backup, verification and DDL.

`--backup-verifier-config` contains a PostgreSQL `DATABASE_URL` for an isolated
verification server. Its role needs permission to create databases. Verification
creates a random `chd_restore_...` database, restores the archive, checks every
application table's row count and index validity, and removes that newly created
database in a `finally` block. Existing databases are never restored or dropped.
A failed verification keeps the archive for diagnosis and stops before migration
DDL. A successful backup includes a private SHA256 receipt with
`restore_verified: true`.

`manifest.json` defines the rollout order, including the legacy-writer bridge
before backfill. Completed stages are checked against their original SHA256 and
postconditions. A database advisory lock prevents overlapping runs. Each backfill
batch commits independently; interruption or a batch limit can be resumed by
rerunning the command. Unknown site names, zero progress, checksum changes, or
schema drift stop the run rather than guessing or marking a stage complete.

## Notification consistency

`20261101_02_notification_delivery.sql` adds `sites.recipient_revision`,
subscription/identity/scope triggers and per-registration delivery receipts.
Apply the complete manifest before deploying the notification services. Fresh
initialization includes these objects. Application startup does not run DDL.

Recipient lookups compare the database version with the cached version; Redis
leases expire after five minutes. Preference, account-status, role, group and
site-assignment changes advance the version within their database transaction.
Cache rebuilds cannot publish an older version over a concurrent rebuild.
Only active users with an explicit opt-in and current site access are recipients.

Confirmed FCM successes are retained for seven days by durable violation ID,
user, token hash and registration revision. An event retry skips those successes;
a new device registration can receive the event again. Events without a durable
violation ID keep the existing short semantic deduplication window and do not
share persistent receipts. Delivery remains at least once when a provider accepts
a request but its response is lost before a success receipt can be committed.

Notification receipt expiry now runs in the notification application's lifespan,
not in notification dispatch. Each sweep uses indexed expiry, `SKIP LOCKED`,
bounded batches and a time budget. Configure `NOTIFICATION_RECEIPTS_INTERVAL_SECONDS`
(default 60), `BATCH_SIZE` (500), `MAX_BATCHES` (4) and `BUDGET_SECONDS` (2) under
that same prefix. Existing receipt tables/indexes are reused; no new PostgreSQL
migration is required for this maintenance change.

`20261102_01_model_versions.sql` adds immutable model-version evidence metadata.
Apply the full manifest before deploying the updated violation service. The
migration backfills the current catalog and trusted definitions in publication
journals; it is repeatable and rejects conflicting hashes/classes for the same
model key and version. Future catalog inserts, updates, rollback and deletion
retain metadata in the same transaction. A version's hash and class definitions
cannot be overwritten; publish a new version to change either.

Evidence uploads check the current model's tenant, role, site, enabled state and
stream capability, then resolve the submitted version's historical classes.
They do not download or load weights. History contains no permission grants.
Versions absent from both the catalog and publication journals cannot be
reconstructed automatically and return `MODEL_VERSION_UNKNOWN`; restore only
trusted metadata for those versions. Previously archived HTTP 409 events remain
available for explicit requeue with the outbox CLI after the migration.

### Account and evidence recovery (20261102_02)

Apply the manifest with the migration runner before starting the updated
management and evidence APIs. `20261102_02_account_evidence_lifecycle.sql` adds
`users.deletion_pending`, `violations.event_fingerprint`, and the durable
`account_operations` / `evidence_write_operations` journals. Existing users stay
active according to their existing status; existing evidence keeps its image
and receives a fingerprint only after a matching retry has been verified.
Fresh installations receive the same schema from `scripts/init.postgres.sql`.

A delete first commits local suspension and intent. Remote deletion and local
cleanup are idempotent; the management API retries unfinished work after
restart. HTTP 503 `account_deletion_pending` means the account remains suspended
and deletion is queued. Retrying the same authorized user-ID deletion is safe.
Keep the API running until pending work completes; operation records survive
local user deletion so a repeated request can still be authorized by its saved
scope. Do not manually reactivate a pending deletion.

New Keycloak users start disabled. A creation records an operation ID without
credentials, records the returned subject before password reset, and activates
the account only after local adoption is committed. A lost remote creation
response is `uncertain`: automatic recovery never guesses the subject from a
username or deletes an existing account. Inspect with:

```sh
python -m scripts.account_operations list
python -m scripts.account_operations retry OPERATION_ID
```

For an uncertain creation, an operator must verify Keycloak creation logs and
the `visionnaire_operation_id` user attribute before explicitly supplying the
subject with `retry OPERATION_ID --subject VERIFIED_SUBJECT`. This can delete an
unadopted remote account. If the subject journal write failed, application logs
also record the operation ID and returned subject. Journals and logs contain no
password. Failed activation preserves the adopted identity and retries enabling
it rather than deleting a successfully committed local account.

Evidence uploads stage and fsync image bytes before promotion, then commit the
record and remove the write intent in one database transaction. Cancellation
before commit cleans the owned files; a lost commit response preserves the
image until recovery can prove whether a record exists. The evidence API checks
intents older than one minute every 30 seconds and skips active writers using
PostgreSQL advisory locks. All instances must share the same evidence storage
root. Reused event IDs require matching normalized metadata and image bytes;
a different body returns HTTP 409 `event_id_conflict`. Missing legacy image
bytes cannot be verified and also return conflict rather than silently adopting
a new body.

### Account state revisions and evidence retries (20261103_01)

Apply `20261103_01_account_state_evidence_recovery.sql` with the existing runner
before starting the updated management and evidence APIs. It adds optimistic
`users.state_revision`, the synchronized revision in account operations, and
attempt/error/next-attempt fields for evidence recovery. Existing evidence
intents remain pending and eligible for recovery. Existing migration files and
checksums are unchanged; fresh initialization includes this migration's schema.

Deploy management workers together: older workers do not understand the new
status/profile/rename operation kinds. Account changes lock and refresh the
current user before validation. They commit local data once and then reconcile
Keycloak from the latest committed state under the same user-row coordination.
An HTTP 503 `account_reconciliation_required` can mean local data was committed
but remote synchronization is pending; retry or inspect `scripts.account_operations`.
No password is stored in these operations. A lost local commit response is
resolved from the committed row, so rollback recovery cannot replay a stale
remote snapshot over a newer account state. Keep the existing operation tables
when rolling back application code until pending operations have been drained.

Evidence recovery uses `EVIDENCE_RECOVERY_` settings: `INTERVAL_SECONDS` (30),
`MIN_AGE_SECONDS` (60), `BATCH_SIZE` (128), `BUDGET_SECONDS` (2), `RETRY_SECONDS`
(5), and `MAX_RETRY_SECONDS` (300). Locked writers are deferred without counting
as failures; failed jobs receive exponential backoff. Selection by next attempt
prevents permanently failing old files from starving later work. Cancellation
waits for in-progress file synchronization or commits before releasing ownership.

Account and evidence operations share independent journal and ownership pools,
recreated from the source pool to preserve connection arguments, custom creators
and connection listeners. They are reused for the source engine's lifecycle and
closed during application shutdown; they never consume request-pool connections
while the request holds its transaction. Budget database connection limits for
these two additional bounded pools per process.

### OpenWebUI progress and operation retention (20261103_02)

Apply `20261103_02_sync_progress_retention.sql` before deploying the new
management workers. It adds `sync_parts` and `completed_at`, a partial due index,
a completed-job index and `account_deletion_receipts`. Previously completed
operations receive the upgrade timestamp so they retain a full retention window;
completed deletions are backfilled into receipts without extending existing
receipt expiry when the SQL is repeated. Fresh initialization includes the same
schema. Existing migration checksums are unchanged.

OpenWebUI synchronization now uses the durable account workflow. Local changes
commit once; state, profile and application roles are synchronized in separate
idempotent steps, each recording progress. A local commit whose response was
lost is resolved from committed data. Retries skip completed steps, and a new
account revision resets progress so old values cannot overwrite current values.
Only the managed OpenWebUI role subset is changed. No-op role updates perform
reads only. Passwords are never stored in operation records and cannot be
replayed automatically; an uncertain password reset still requires an explicit
retry by the operator.

`ACCOUNT_OPERATIONS_COMPLETED_DAYS` defaults to 30 and
`ACCOUNT_OPERATIONS_CLEANUP_BATCH_SIZE` to 128. Completed operations are purged
in bounded batches; pending and uncertain work is retained.
`ACCOUNT_OPERATIONS_DELETION_RECEIPT_DAYS` defaults to 365 for new deletions.
Receipts preserve the original authorization scope for repeat user-ID deletions
after their detailed operation is purged; requests after receipt expiry return
404. The migration grants existing completed deletions a 365-day receipt window.
Inspect pending progress with `python -m scripts.account_operations list` and
resume with `retry OPERATION_ID`.

Upgrade all management, BFF and notification instances together. Older BFF
writers do not enforce refresh revisions, and older notification handlers still
perform unconditional cache deletion. Existing BFF sessions without a revision
are treated as revision zero and move forward on their first successful refresh;
no token plaintext is written to PostgreSQL. Old FCM caches without registration
versions are invalidated for reconstruction instead of being blindly deleted.


Account maintenance no longer couples retention to synchronization. The
`AccountLifecycle` coordinator retains operation ownership and commits;
`AccountSynchronizer` owns typed synchronization plans and validated per-part
progress; `AccountRetention` owns completion receipts and bounded pruning.
Invalid persisted progress is rejected before remote effects.

Under `ACCOUNT_OPERATIONS_`, `SYNC_INTERVAL_SECONDS` defaults to 5,
`SYNC_BATCH_SIZE` to 16, `SYNC_CONCURRENCY` to 2 and `SYNC_BUDGET_SECONDS` to 30.
Independent `CLEANUP_INTERVAL_SECONDS` defaults to 60 and
`CLEANUP_BUDGET_SECONDS` to 2. A cleanup timeout cannot skip pending operations;
synchronization isolates per-operation failures and reschedules timed-out work.
Shutdown cancels both maintenance tasks and drains transaction cleanup before
releasing pools. These scheduling changes require no additional SQL migration.
