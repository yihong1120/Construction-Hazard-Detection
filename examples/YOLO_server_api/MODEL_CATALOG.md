# Database model keys with Hugging Face weights

Clients still submit only `model_key` for streams and `model` for images.
`detection_model_catalog` is the only catalog: one row per key, with definition
JSON containing display name, enabled state, capabilities, tenant/site/role
grants, exact Hugging Face repository/file/commit, SHA-256 and classes. This
avoids separate model-version/class tables while retaining reproducibility.
Model weights are stored on Hugging Face, not in PostgreSQL.

## Publishing

Apply `scripts/migrate_model_catalog.sql` before deploying. With the inference
Python environment, run from the repository:

```bash
python -m scripts.sync_huggingface_models --model-key yolo26m
```

The default repository is `yihong1120/Construction-Hazard-Detection`. Standard
keys map to `models/yolo26/pt/yolo26m.pt` (or the corresponding YOLO family).
Custom keys use `--repo` and `--filename`; they are not restricted to YOLO names.

For a new key, explicitly grant access:

```bash
python -m scripts.sync_huggingface_models --model-key site-detector \
  --repo owner/models --filename weights.pt \
  --tenant-id TENANT_UUID --site-id 20
```

Use `--revision COMMIT_OR_TAG` to choose a publication; `main` is resolved to an
immutable commit before any database write. `--default KEY` selects a default
from the keys in this command. Existing keys preserve grants and enabled state;
explicit tenant/site arguments replace the respective grant list. The command
checks file hashes and class uniqueness before committing any changed rows.
Only run it against repositories you trust to supply PyTorch model files.

`--import-registry FILE` is solely a one-time migration input for old grants.
No API reads it and `MODEL_REGISTRY_PATH` is unused. The existing local JSON
files may be retained until old server processes have been replaced, then
removed. The initial five keys have already been imported in this deployment.

## Runtime

Catalog reads occur in worker threads. Missing database/table or malformed
metadata returns 503; an intentionally empty/disabled catalog is an empty 200.
Options and mutations both check current tenant/site/role grants. Model files
are downloaded only when needed, from the published commit via Hugging Face's
cache, then SHA-256 verified. Configure `HF_TOKEN` for private repositories and
`HF_HOME` for a persistent cache shared by service processes. Each container
must be able to read the database and write its Hugging Face cache.

WebSocket detection also resolves keys through the catalog using the verified
account scope; it no longer uses the static legacy model list.

`GET /models` exposes that commit as `version`. `/detect` executes that version
or rejects a stale version with 409, preserving the existing result array.
`/model_file_update` rejects in-place replacements: publish to Hugging Face and
sync the key instead. `/get_new_model` serves the same authorized artifact with
`X-Model-Id`, `X-Model-Version`, `X-Model-SHA256` and an ETag. Downloads use checksum
conditional requests; local filesystem timestamps do not decide model identity.

Stream workers load their key from the same catalog and attach the actual
loaded commit to their results. The violation sender forwards that metadata,
and the violation backend stores the associated class snapshot. Historical
records without metadata are not filled with today's classes.

`python -m scripts.sync_huggingface_models` is the publication entry point. It
serializes publications with a database advisory lock, verifies the selected
artifacts and the previous artifacts needed for restoration, then commits and
notifies the stream supervisor. Catalog revisions are part of worker topology;
only affected shards and camera clients restart when their IPC layout is stable.
Camera dispatch and result queues use the immutable database `stream_id`.
The commit compares the catalog against its preflight snapshot, preserving edits
made while artifacts were being prepared.

Publication waits for each affected active stream to report its actual loaded
catalog revision. A fresh Redis challenge is required, so an older cached
acknowledgement cannot satisfy verification. `--verify-timeout` defaults to 180
seconds. A failed or cancelled publication restores the previous catalog and
verifies it with another challenge. Concurrent catalog edits are preserved and
reported as a restoration conflict. A verification or restoration failure exits
unsuccessfully. `--catalog-only` permits initial publication only when no affected
stream is active.

Before deploying this publication entry point, apply the migration manifest,
including `20261101_01_model_publications.sql`. Fresh database initialization
includes this table. Publication prints its ID and persists the immutable desired
catalog, previous definitions, affected cameras and phase in PostgreSQL. Catalog
changes and the `applied` phase commit together; restoration records its catalog
commit separately from worker verification.

If the CLI is killed or the host restarts, another publication stops and reports
the pending ID. Resume verification without downloading or resolving a mutable
Hugging Face revision again:

```bash
python -m scripts.sync_huggingface_models --recover PUBLICATION_ID
python -m scripts.sync_huggingface_models --recover PUBLICATION_ID --restore
```

Recovery creates a fresh worker challenge and uses the recorded verification
timeout. A `prepared` operation is safely abandoned by restoring and verifying its
previous state. An interrupted restoration continues restoration. Completed
operations can be recovered repeatedly. Failed verification keeps the operation
pending so it can be retried; concurrent catalog edits require reconciliation and
are never overwritten. Retain the referenced previous artifacts until recovery
finishes. Recovery runs explicitly when the operator invokes this entry point.

For INT8, prepare the new engine manifest before publication. Both the selected
engine and the previous version must be available and checksum-valid. Engine
builds use immutable versioned filenames and retain two previous manifest entries
for restoration. Missing previous artifacts fail before the catalog changes.
Historical violation snapshots remain readable throughout publication.

Deploy/restart management, detection, violations and stream workers together.
This change supports catalog-published weights; the old worker model-directory
and filename-suffix settings no longer override the catalog artifact.

## Class colors

Classes optionally carry opaque sRGB `color` (`#RRGGBB`). Invalid colors are
replaced by server-owned colors by stable code without invalidating metadata. Publishing from Hugging Face
preserves configured colors by stable class code and initializes missing colors.
For legacy catalogs, run `python -m scripts.publish_class_colors --apply`
before enabling immutable model-version history if colors need initialization.
After history is installed, class metadata changes require a new published
version; the one-time color utility cannot overwrite an archived version.
Stream supervisors reload published metadata on their next configuration check.
Record uploads persist class snapshots;
list/detail `class_metadata` and feedback options read those original snapshots.

## API runtime ownership

The image API uses one version-keyed lease cache. It has no filename watcher,
legacy preload cache or direct raw-model getter. LRU hits update recency; active
leases and loads count toward `MAX_LOADED_MODELS`. The shared runtime bounds
both HTTP and WebSocket jobs, including artifact resolution/copy/verification.
Catalog authorization is separate from artifact I/O so cancelled authorization
requests cannot abandon unaccounted native model work. WebSocket selections keep
metadata only and reacquire a lease per frame. See DETECTION_MODEL_API.md for
capacity settings and cancellation behavior.

## Evidence version history

The live catalog controls new inference and current access rights. Immutable
`detection_model_versions` rows retain each model key/version's SHA-256 and class
metadata for delayed evidence. Database triggers retain both sides of catalog
changes, including rollback and deletion; reusing a version with different
weights or classes is rejected. Permission and display-order changes remain
possible. Model deletion/disablement or revoked current access never becomes an
access grant through history.

Run the migration manifest, including `20261102_01_model_versions.sql`, before
upgrading the violation service. It backfills trusted current and journaled
metadata. Evidence validation is separate from loading inference artifacts.
Unknown historical versions fail explicitly instead of accepting caller-supplied
class metadata.

Publication verification follows currently active streams for the selected
models on each poll. A second membership read confirms acknowledgements before
completion; new members must acknowledge the same revision and publication
challenge, while removed members stop blocking verification. The journal records
`verified_streams` as of this final check. This is a verification point, not a
freeze on camera configuration: later activations follow the normal supervisor
configuration lifecycle. An initially idle model obtains a Redis challenge if
its first camera becomes active during verification.
