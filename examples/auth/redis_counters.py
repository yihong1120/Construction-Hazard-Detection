"""Atomic fixed-window counters with expiry repair for legacy keys."""
from __future__ import annotations

FIXED_WINDOW_SCRIPT = """
local count = redis.call('INCR', KEYS[1])
local ttl = redis.call('TTL', KEYS[1])
if ttl < 0 then
    redis.call('EXPIRE', KEYS[1], ARGV[1])
    ttl = tonumber(ARGV[1])
end
return {count, ttl}
"""


async def increment_window(redis, key, seconds):
    if type(seconds) is not int or seconds <= 0:
        raise ValueError(
            'A counter window must be a positive number of seconds.',
        )
    count, ttl = await redis.eval(FIXED_WINDOW_SCRIPT, 1, key, seconds)
    return int(count), int(ttl)
