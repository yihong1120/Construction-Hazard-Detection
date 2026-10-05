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

After syncing a different commit, restart stream workers to load it; running
workers deliberately retain their loaded weights. If an old worker submits an
unpublished version, the backend rejects it rather than recording the current
version incorrectly. Existing historical snapshots remain readable.

Deploy/restart management, detection, violations and stream workers together.
This change supports catalog-published weights; the old worker model-directory
and filename-suffix settings no longer override the catalog artifact.

## Class colors

Classes optionally carry opaque sRGB `color` (`#RRGGBB`). Invalid colors are
replaced by server-owned colors by stable code without invalidating metadata. Publishing from Hugging Face
preserves configured colors by stable class code and initializes missing colors.
Run `python -m scripts.publish_class_colors --apply` once for existing catalogs;
this updates catalog JSON only, never historical violations. Restart inference
workers to reload published metadata. Record uploads persist class snapshots;
list/detail `class_metadata` and feedback options read those original snapshots.
