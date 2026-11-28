🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Settings and their owners

The settings layer reads environment variables into `EnvironmentSettings` models. Each component reads the fields it needs. A service can start without credentials for unused notification providers or imports of machine learning packages.

| Component | Settings | When they are read |
| --- | --- | --- |
| Database and startup | `examples/auth/settings_fields.py`, `connections.py` | Engine creation or application startup |
| Redis and outbound APIs | `connections.py` | Client creation |
| BFF proxy and cookies | `examples/bff/settings.py` | Process initialisation |
| Session expiry and encryption | `sessions.py` | Process initialisation, or first use of the cipher |
| Worker authentication | `worker_auth.py` | The token provider reads credentials on first use and reuses them for renewal |
| Capture and inference | `capture.py`, `inference.py` | Stream or detector creation |
| Rust and Python GPU workers | `backend` and `native_library_dirs` in `inference.py` | Worker creation, with backend selection after model verification |
| GPU quotas across processes | `gpu_resources.py`, `inference.py` | Participants share a directory and physical GPU identifier. NVENC sessions use this quota by default. Inference and memory limits are opt-in |
| Rust post-processing | `postprocess.py` | Camera process startup. Restart it after a change |
| OpenCV and Rust capture | Backend, decoder and tool paths in `capture.py` | Camera startup. Select the Rust FFmpeg backend explicitly |
| Media, text rendering and NVENC | `media.py` | Gateway or renderer creation. The NVENC limit is fixed for the reservation |
| Notification providers | `notifications.py` | Provider creation |
| MCP | `examples/mcp_server/settings.py` | Transport startup, tool creation or first use |
| Management policies and diagnostics | `policy_settings.py`, `debug_settings.py` | Request handling or middleware creation |

Components own their settings. Pure functions take the data they operate on, and constructors continue to accept explicit values.

In Python, `main.py` uses `STREAM_CAPTURE_INFERENCE_HISTORY_BY_CAMERA` to set how many frames each camera can retain for inference. Keys are PostgreSQL `stream_configs.id` values. The depth is 1 to 4 and includes the latest frame. For example, `STREAM_CAPTURE_INFERENCE_HISTORY_BY_CAMERA='{"64":4}'` in `.env` lets camera 64 process a burst of frames in order. The 100 ms window starts when the capture loop hands a frame to the Python inference buffer. Inference takes the latest frame if older frames have expired. Reconnecting clears the buffer. The default `{}` retains just the latest frame per camera. Live previews always use the latest frame.

GPU quotas use Rust file-lock leases. The Python fallback uses the same protocol when the Rust wheel is unavailable. All participants must use the same `GPU_RESOURCE_DIRECTORY`, `GPU_RESOURCE_DEVICE_ID` and limits. Camera processes and replicas in Compose share the `gpu-resource-locks` volume. Other Compose projects and host processes must mount the same directory and run with the same UID. Keep lock files and their directory in place while jobs run. Participants reject quota changes that conflict with active processes.

`MEDIA_NVENC_MAX_SESSIONS` limits sessions within that shared scope. FFmpeg inherits the lease descriptor, so a surviving encoder retains its slot if the controller exits. `GPU_RESOURCE_INFERENCE_SLOTS` and `GPU_RESOURCE_MEMORY_LIMIT_MB` default to 0. Enabling either requires `YOLO_WORKER_BACKEND=rust`. Memory limits also require `YOLO_WORKER_GPU_MEMORY_RESERVATION_MB` for each model. Reservations round up to 64 MiB and the total limit rounds down. Allow headroom for external CUDA processes, NVDEC and GPU glyph caches. API inference and the Python backend run outside this accounting, so the quotas cannot guarantee sufficient memory.

`YOLO_WORKER_NATIVE_ASYNC=true` enables pinned memory, CUDA streams and events, and two buffers. `YOLO_WORKER_NATIVE_MICRO_BATCH_SIZE=0` uses the engine capacity. Smaller values split the worker batch. The worker holds at most two outstanding results and drains them before a shape change. A request consisting of a single undivided batch has no second batch to overlap. See the [Rust GPU worker measurements](../../native/inference/README.md).

## Precedence and validation

Explicit constructor arguments take precedence over environment variables, which take precedence over defaults. `from_overrides()` handles field aliases and `AliasChoices`. Only `None` indicates an omitted argument. Empty strings and zero go through validation, so an explicitly empty Redis password can override the environment.

Existing environment names and valid defaults remain compatible. Settings validate timeouts, ports, capacities and booleans at load time. Misspellings, non-finite numbers, out-of-range values and unsupported MCP transports raise errors. Source, publishing and preview frame rates follow their existing selection rules.

Service URLs require HTTP(S), a host and a valid port, and reject embedded credentials. Worker token endpoints require HTTPS outside loopback. A `__Host-` cookie requires the secure flag. The API and worker use the same overlay readiness TTL with a minimum of five seconds.

Provider and transport credentials use `SecretStr` to mask secrets in model representations and ordinary serialisation. Passwords preserve whitespace. BFF encryption keys and the MCP static bearer retain their established trimming rules. Redis URLs encode passwords and support IPv6.

## Configuration changes and tests

A running component retains its settings when `os.environ` changes. Recreate the component or restart the service after changing its deployment environment. The token provider renews with its original credentials. The session cipher keeps the key from its first use. Rotating a BFF key requires a plan to revoke existing sessions or migrate their keys. Tests should set the environment before constructing a component.

Site and tenant policy files and gateway state files retain their existing read behaviour. Overlay language policies use a bounded cache keyed by content. Callers hold an immutable policy for their own lifetime. The settings layer still assembles legacy social client ID fields and reads the language cache.

`tests/src/configuration_contract_test.py` covers precedence, validation before resource creation, settings snapshots, secrets and shared TTLs. It also checks that business modules do not add direct `os.getenv()` calls.
