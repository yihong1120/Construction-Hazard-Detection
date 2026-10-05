# Resource-specific options APIs

The management `/client-options` endpoint and all scopes have been removed.
There are no compatibility gateways. Options belong to their resource APIs.

| Service | Method and relative path | Notes |
| --- | --- | --- |
| Management | `GET /sites/{site_id}/stream-models` | Current site/model permissions |
| Management | `GET /users/assignable-roles?target_user_id=123&locale=zh-TW` | Omit target_user_id when creating a user |
| Management | `GET /deployment-enrollment-codes/expiry-options` | Current deployment administrator policy |
| Violations | `GET /violations/{id}/feedback-options?locale=zh-TW` | Original record snapshot categories and colors |
| Violations | `GET /violations/{id}/review-options?locale=zh-TW` | Current role and record state |
| Chat | `GET /chat/upload-policy` | Direct chat backend policy; no management proxy |

The first five endpoints return `schema_version: 1`, `revision`,
`options: [{id, label}]`, `default_id`, and `policy`. The chat endpoint returns
`{revision, policy}`; frontend chat policy parsing must use that response shape.
Native clients send bearer access tokens. Browser clients use the respective
BFF service paths: `/bff/db_management`, `/bff/violations`, and `/bff/chat`.
All successful policy responses are private/no-store. Missing/invalid policy
configuration returns 503; successful empty selections return 200.

Management owns role/invitation/site authorization in its services. Detection
owns `YOLO_server_api/model_registry.py`, also imported by stream configuration
and violation upload. Violations own `record_policy.py`. Chat owns attachment
policy and enforcement in the separate `openclaw-orchestrator` repository.

## Rollout prerequisites

1. Apply `scripts/migrate_client_options.sql` before restarting any service
   using the shared ORM. Existing records intentionally retain NULL metadata:
   their original classes cannot safely be inferred from today's model.
2. Apply `scripts/migrate_model_catalog.sql`. The single
   `detection_model_catalog` table is authoritative for model keys, publication
   state, Hugging Face commit/file, class metadata and grants. APIs never read
   a JSON registry or silently fall back to a static model list.
3. Publish keys with `python -m scripts.sync_huggingface_models --model-key
   yolo26m`. The command verifies Hugging Face's SHA-256 and reads classes from
   those exact weights, then atomically publishes the metadata. Existing grants
   are preserved. New keys need explicit `--tenant-id` and, for streaming,
   `--site-id`. See `examples/YOLO_server_api/MODEL_CATALOG.md`.
4. Optionally set `INVITATION_POLICY_PATH` to a JSON map keyed by tenant UUID,
   as in `config/invitation_policy.example.json`. The same policy validates
   invitation creation; every offered value and default must fit the ceiling.
   Without this optional override, the server owns a policy of
   30/60/120/1440 minutes with a 1440-minute maximum. With an override, a
   missing tenant is an outage, not a fallback.
5. Deploy the companion `openclaw-orchestrator` changes, apply its
   `migrations/20260928_chat_tenant.sql`, and set `CHAT_AUTH_ME_URL` to
   management's `/me` endpoint. Clients request chat's `/chat/upload-policy`
   directly. `CHAT_UPLOAD_POLICY_URL` is no longer used and can be removed
   from management configuration. See chat's `docs/chat-upload-policy.md`.
6. Restart affected services, run iOS/Web end-to-end checks, then release the
   dependent frontend. Applying migrations and restarting services are
   separate rollout steps; changing source files alone does neither.

## Enforcement and compatibility

- Role options and create/update requests share the same allow-list. A group
  admin cannot assign admin or manage peer administrators. Cross-tenant
  target mutation and the singleton super-admin role are denied.
- Stream create/update/replacement reload the registry and recheck current
  site membership, tenant, role, stream capability, enabled state and artifact
  hash. `target_id` itself grants nothing. Explicit site membership is required
  for the new options and write checks, including platform operators.
- Detection `/models` reads the database, returning `version` and `classes`.
  `/detect` consumes optional multipart `model_version`, rejects stale/disabled
  versions with 409 `MODEL_UNAVAILABLE`, and loads a verified immutable copy.
  Missing local weights are fetched from the published Hugging Face commit.
  Discovery does not download weights. Database failure returns 503.
- Violation upload accepts optional `model_id` and `model_version` together,
  validates them against the registry and site, and saves the server-owned
  class snapshot in the same transaction as the evidence record. Update
  producers to send the actual version that performed inference; do not
  derive it from the camera's current settings. Both model fields are required.
- Feedback options and submissions use the original record class snapshot.
  Records missing snapshots retain empty options; never infer classes from the latest model.
  Feedback requests do not need model ID/version or color; unknown snapshot
  codes and invalid coordinates are rejected.
- Reviews lock/reload records and apply current transitions: pending to
  resolved/dismissed; resolved/dismissed back to pending. Non-reviewers get
  no review actions and cannot forge a review submission.
- The companion chat service enforces content-inspected MIME, per-file size,
  total new plus retained attachments, and ownership on upload/create/edit.
  It rechecks stored drafts under current policy. Exact MIME limits precede
  wildcard limits and remain bounded by the global ceiling. Its rooms retain
  tenant identity to prevent access after an account changes tenants. Both
  repositories must be deployed and tested together. Chat performs its own
  attachment writes and policy checks.
- GET revisions hash the current visible response plus actor/target and policy
  source. Clients need not send revisions back. GET never grants permanent
  authority; mutations validate again. Responses use private/no-store caching.

Locale is a hint. Role/review labels support Chinese and English; registry
labels and historical class snapshots are the fallback for other locales.
Model and class IDs never change with locale.


## Current database publication

Five YOLO26 keys have been imported from the existing local registry and
verified against Hugging Face. Stream grants are unchanged: site 20 has
n/m/l/x, site 1 has m/x, site 43 has m. The s variant has no stream site grants.
The old JSON files are migration/rollback inputs only; `MODEL_REGISTRY_PATH`
is no longer read. Restart management, detection, violations and stream workers
after deployment. Workers keep the loaded commit pinned until restarted.
