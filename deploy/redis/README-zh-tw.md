🇬🇧 [English (UK)](./README.md) | 🇹🇼 [繁體中文（台灣）](./README-zh-tw.md)

# 使用 systemd 部署 Redis

執行下列指令安裝 Redis 服務並設定重新啟動的間隔。

```bash
sudo install -D -m 644 deploy/redis/redis-server.service.d/override.conf \
  /etc/systemd/system/redis-server.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl enable redis-server.service
sudo systemctl restart redis-server.service
sudo systemctl status redis-server.service --no-pager
```

套件提供的服務單元會在開機時啟動 Redis。覆寫設定將舊主機專用的監聽位址改為 `127.0.0.1`，服務失敗後等候五秒再重試，以免觸發 systemd 的短時間重啟次數上限。
