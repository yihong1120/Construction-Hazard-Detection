-- The aggregate describes exact streaming revisions. Check every member before
-- renewal so an expired/replaced camera cannot partially renew the wall.
local group_key = KEYS[1]
local action, raw, owner = ARGV[1], ARGV[2], ARGV[3]
local current = redis.call('GET', group_key)
if action == 'register' then
    if current then return -2 end
elseif action == 'abort' then
    if current and current ~= raw then return -2 end
elseif current ~= raw then
    return -1
end

local removing = action == 'release' or action == 'abort'
local lease = cjson.decode(raw)
if lease.owner ~= owner then return -1 end
local now = tonumber(redis.call('TIME')[1])
local session_ttl = tonumber(ARGV[7])
local expires = math.min(tonumber(ARGV[4]), now + session_ttl, now + 900)
if not removing and
    (expires <= now or redis.call('EXISTS', ARGV[5], ARGV[6]) > 0) then
    return -1
end

local function path_ok(path)
    return type(path) == 'string' and #path <= 1024 and
        string.match(path, '^hazard_[%w_-]+$') ~= nil
end

local function index_ok(key)
    local kind = redis.call('TYPE', key).ok
    return kind == 'none' or kind == 'zset'
end

local entries = {}
for _, member in ipairs(lease.members) do
    local session_key = 'stream_playback_session:' .. member.id
    local session_raw = redis.call('GET', session_key)
    local session = session_raw and cjson.decode(session_raw)
    local matches = session and session.revision == member.revision and
        session.username == owner and session.label == lease.site and
        session.stream_name == member.camera
    if not matches then
        if not removing then return -1 end
    else
        local base = session.base_media_path
        local selected = base
        if not path_ok(base) then return -2 end
        local type_key = 'clean:' .. base
        local producer = ARGV[8] .. 'media_clean_demand:' .. base
        if session.profile == 'overlay' then
            selected = session.overlay_media_path
            if not path_ok(selected) then return -2 end
            local root, language = string.match(selected, '^(.*)_annotated_([%w_-]+)$')
            if root ~= base or #language > 16 then return -2 end
            type_key = 'overlay:' .. base .. ':' .. language
            producer = ARGV[8] .. 'media_overlay_demand:' .. base .. ':' .. language
        elseif session.profile ~= 'clean' then
            return -2
        end
        local media = 'stream_playback_media_session:' .. selected
        local fallback = 'stream_playback_media_session:' .. base
        local demand = 'stream_playback_demand_session:' .. type_key
        if not index_ok(media) or not index_ok(fallback) or not index_ok(demand) then
            return -2
        end
        local capability_key = member.capability_key
        local cap_raw = redis.call('GET', capability_key)
        local cap = cap_raw and cjson.decode(cap_raw)
        if cap and (cap.parent ~= session_key or cap.playback_revision ~= member.revision) then
            return -2
        end
        if not removing then
            if not cap or tonumber(cap.expires_at) <= now or
                redis.call('EXISTS', cap.access_revocation_key, cap.oidc_revocation_key) > 0 then
                return -1
            end
            if action == 'register' then expires = math.min(expires, tonumber(cap.expires_at)) end
        end
        table.insert(entries, {
            session_key = session_key, capability_key = capability_key,
            media = media, fallback = fallback, demand = demand, producer = producer,
            id = member.id, capability = cap
        })
    end
end

if removing then
    for _, item in ipairs(entries) do
        redis.call('DEL', item.session_key, item.capability_key)
        redis.call('ZREM', item.media, item.id)
        redis.call('ZREM', item.fallback, item.id)
        redis.call('ZREM', item.demand, item.id)
        redis.call('ZREMRANGEBYSCORE', item.demand, '-inf', now)
        if redis.call('ZCARD', item.demand) == 0 then redis.call('DEL', item.producer) end
    end
    redis.call('DEL', group_key)
    return 0
end

if action == 'renew' then
    for _, item in ipairs(entries) do
        local cap = item.capability
        cap.expires_at = expires
        cap.access_revocation_key = ARGV[5]
        cap.oidc_revocation_key = ARGV[6]
        redis.call('SET', item.capability_key, cjson.encode(cap), 'EXAT', expires)
        redis.call('EXPIRE', item.session_key, session_ttl)
        for _, key in ipairs({item.media, item.demand}) do
            redis.call('ZADD', key, now + session_ttl, item.id)
            redis.call('EXPIRE', key, session_ttl)
        end
    end
end
redis.call('SET', group_key, raw, 'EXAT', expires)
return expires - now
