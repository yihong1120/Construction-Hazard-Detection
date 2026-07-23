# Redis systemd deployment

Install the managed Redis service and its restart delay:

```bash
sudo install -D -m 644 deploy/redis/redis-server.service.d/override.conf \
  /etc/systemd/system/redis-server.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl enable redis-server.service
sudo systemctl restart redis-server.service
sudo systemctl status redis-server.service --no-pager
```

The packaged unit starts Redis at boot. The drop-in overrides the obsolete
host-specific bind address with `127.0.0.1`, then retries failures after five
seconds instead of exhausting systemd's rapid-restart limit.
