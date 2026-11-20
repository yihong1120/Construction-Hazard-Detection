🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# Live metadata API benchmark

Compare per-viewer Python serialisation, shared Python serialisation and a Rust Axum prototype. Each service receives the same synthetic camera metadata from an isolated Redis process and delivers it over SSE or WebSocket on loopback. Both Python variants use the frame reader and fanout from `streaming_web`.

The measured path starts when the producer prepares a Redis publication batch and ends when the load client receives the update. The Rust prototype serves fixture routes for this experiment. In production, authentication, playback leases, BFF proxying and video delivery add their own costs.

## Run

Use Linux with Python 3.15, Rust 1.93.1 and `redis-server` installed. Run from the repository root.

```bash
uv sync --frozen --no-install-project
uv pip install 'wsproto==1.3.2'
cargo build --release --locked \
  --manifest-path experiments/metadata_api_benchmark/rust/Cargo.toml
```

Run the benchmark with 32 synthetic cameras and both protocols. Each case is repeated three times, with the service order rotated between repetitions.

```bash
uv run --no-sync python -m experiments.metadata_api_benchmark.benchmark \
  --rust-binary experiments/metadata_api_benchmark/rust/target/release/metadata-api-benchmark \
  --cameras 32 --viewers-per-camera 1 4 --events-per-second 15 --duration 6 \
  --protocol sse ws --repeats 3 --output outputs/metadata_api_benchmark/steady.json
```

The viewer count is the number of connections per camera. The JSON report records service CPU use relative to one core, sampled process memory, p95 and p99 delivery latency, and the fraction of expected updates delivered. Under load the service can skip superseded frames, so compare delivery rate alongside latency.

Set `--slow-client-delay-ms 125` to pace the last client for each camera. The report separates paced and regular clients. TCP and client buffers can absorb short bursts even when the client application reads slowly.
