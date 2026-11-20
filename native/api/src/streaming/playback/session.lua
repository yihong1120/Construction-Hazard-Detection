local action = ARGV[1]
local owner = ARGV[2]
local options = cjson.decode(ARGV[3])
local entries = cjson.decode(ARGV[4])
local legacy_owner = ARGV[5]
local time = redis.call('TIME')
local now = tonumber(time[1]) + tonumber(time[2]) / 1000000
local expires = now + options.session_ttl
local previous = {}
local languages = {}
local ready = {}
local function path_ok(path)
    return type(path) == 'string' and #path <= 1024 and
        string.match(path, '^hazard_[%w_-]+$') ~= nil
end
local function index_ok(key)
    local kind = redis.call('TYPE', key).ok
    return kind == 'none' or kind == 'zset'
end
local function media_index(path)
    return 'stream_playback_media_session:' .. path
end
local function producer(label_path, language_key)
    if language_key then
        return options.demand_prefix .. 'media_overlay_demand:' .. label_path .. ':' .. language_key
    end
    return options.demand_prefix .. 'media_clean_demand:' .. label_path
end
local function demand_index(label_path, language_key)
    if language_key then
        return 'stream_playback_demand_session:overlay:' .. label_path .. ':' .. language_key
    end
    return 'stream_playback_demand_session:clean:' .. label_path
end
local function get_previous(entry)
    local raw = redis.call('GET', entry.key)
    if not raw then
        if entry.replace then return 1 end
        return 0
    end
    if not entry.replace then return 4 end
    local valid, old = pcall(cjson.decode, raw)
    if not valid or type(old) ~= 'table' then return 5 end
    -- Python used the verified database username and a 24-digit hex
    -- revision. Native sessions are owned by the signed OIDC subject.
    local rev = old.revision
    local legacy = rev == nil or (type(rev) == 'string' and #rev == 24 and rev:match('^[0-9a-f]+$'))
    if legacy then
        if legacy_owner == '' or old.username ~= legacy_owner then return 2 end
    elseif old.username ~= owner then return 2 end
    if not path_ok(old.base_media_path) then return 5 end
    local language_key
    if old.profile == 'overlay' then
        if not path_ok(old.overlay_media_path) then return 5 end
        local base, suffix = string.match(old.overlay_media_path, '^(.*)_annotated_([%w_-]+)$')
        if base ~= old.base_media_path or #suffix > 16 then return 5 end
        language_key = suffix
    elseif old.profile ~= 'clean' then
        return 5
    end
    local path = language_key and old.overlay_media_path or old.base_media_path
    local media = media_index(path)
    local demand = demand_index(old.base_media_path, language_key)
    if not index_ok(media) or not index_ok(demand) then return 5 end
    previous[entry.key] = {
        profile = old.profile,
        media = media,
        fallback = media_index(old.base_media_path),
        demand = demand,
        producer = producer(old.base_media_path, language_key)
    }
    if not index_ok(previous[entry.key].fallback) then return 5 end
    return 0
end
for i, entry in ipairs(entries) do
    local status = get_previous(entry)
    if status ~= 0 then return {status} end
    if action == 'create' then
        if not index_ok(media_index(entry.selected_path)) or
            not index_ok(demand_index(entry.base_path, entry.language_key)) then
            return {5}
        end
        if entry.language_key then
            local camera = entry.detail_path
            if not languages[camera] then
                local current = {active = {}, requested = {}}
                for _, key in ipairs(options.language_keys) do
                    if redis.call('EXISTS', producer(camera, key), producer(camera .. '_preview', key)) > 0 then
                        current.active[key] = true
                    end
                end
                languages[camera] = current
            end
            languages[camera].requested[entry.language_key] = true
            ready[i] = redis.call('EXISTS', options.demand_prefix .. 'media_overlay_ready:' .. entry.selected_path)
        else
            ready[i] = 0
        end
        if redis.call('EXISTS', entry.capability_key) > 0 then return {4} end
        if tonumber(entry.capability_expires) <= tonumber(time[1]) or
            redis.call('EXISTS', entry.revocation_key, entry.oidc_revocation_key) > 0 then
            return {6}
        end
    end
end
for _, camera in pairs(languages) do
    local new = false
    for key in pairs(camera.requested) do
        if not camera.active[key] then new = true end
        camera.active[key] = true
    end
    local count = 0
    for _ in pairs(camera.active) do count = count + 1 end
    if new and count > options.overlay_limit then return {3} end
end
local function remove_previous(entry)
    local old = previous[entry.key]
    if old then
        redis.call('ZREM', old.media, entry.id)
        if old.fallback ~= old.media then redis.call('ZREM', old.fallback, entry.id) end
        redis.call('ZREM', old.demand, entry.id)
    end
end
if action == 'release' then
    local entry = entries[1]
    local old = previous[entry.key]
    remove_previous(entry)
    redis.call('DEL', entry.key)
    redis.call('ZREMRANGEBYSCORE', old.demand, '-inf', now)
    if redis.call('ZCARD', old.demand) == 0 then redis.call('DEL', old.producer) end
    if old.profile == 'overlay' then return {0, 1} else return {0, 0} end
end
for _, entry in ipairs(entries) do remove_previous(entry) end
for _, entry in ipairs(entries) do
    redis.call('SET', entry.key, entry.raw, 'EX', options.session_ttl)
    redis.call('SET', entry.capability_key, entry.capability, 'EX', entry.capability_expires - tonumber(time[1]))
    local demand_lease = entry.language_key and options.overlay_demand_ttl or options.clean_demand_ttl
    local producer_key = producer(entry.base_path, entry.language_key)
    -- A playback request must not shorten an already authorised media lease.
    local producer_ttl = redis.call('TTL', producer_key)
    redis.call('SET', producer_key, '1', 'EX', math.max(demand_lease, producer_ttl))
    for _, index in ipairs({media_index(entry.selected_path), demand_index(entry.base_path, entry.language_key)}) do
        redis.call('ZADD', index, expires, entry.id)
        redis.call('EXPIRE', index, options.session_ttl)
    end
end
table.insert(ready, 1, 0)
return ready
