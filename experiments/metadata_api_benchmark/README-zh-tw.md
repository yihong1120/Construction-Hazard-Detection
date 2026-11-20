🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 即時中繼資料 API 效能比較

比較 Python 逐連線序列化、Python 共用序列化與 Rust Axum 原型。每個服務都從隔離的 Redis 程序取得相同的模擬攝影機中繼資料，並透過本機 SSE 或 WebSocket 連線發布。兩種 Python 版本都使用 `streaming_web` 的影格讀取與推送程式。

量測從發布端開始準備 Redis 批次，到壓測用戶端收到更新為止。Rust 原型提供壓測專用的端點。正式環境還有授權、播放租約、BFF 代理和影片傳送各自的開銷。

## 執行方式

請使用已安裝 Python 3.15、Rust 1.93.1 及 `redis-server` 的 Linux 環境，並從儲存庫根目錄執行。

```bash
uv sync --frozen --no-install-project
uv pip install 'wsproto==1.3.2'
cargo build --release --locked \
  --manifest-path experiments/metadata_api_benchmark/rust/Cargo.toml
```

以下測試使用 32 路模擬攝影機與兩種協定。每個情境重複三次，並輪替服務的測試順序。

```bash
uv run --no-sync python -m experiments.metadata_api_benchmark.benchmark \
  --rust-binary experiments/metadata_api_benchmark/rust/target/release/metadata-api-benchmark \
  --cameras 32 --viewers-per-camera 1 4 --events-per-second 15 --duration 6 \
  --protocol sse ws --repeats 3 --output outputs/metadata_api_benchmark/steady.json
```

觀看數代表每一路攝影機的連線數。JSON 報告記錄以單核心為基準的服務 CPU 使用率、程序記憶體取樣、p95 與 p99 傳送延遲，以及更新送達率。負載過高時，佇列可跳過已被取代的舊更新，因此解讀延遲時要一起看送達率。

加上 `--slow-client-delay-ms 125` 可延緩每路攝影機最後一條連線的應用程式讀取速度。報告會分開統計慢速與一般用戶端。短暫突發流量仍可能由 TCP 或用戶端緩衝區吸收。
