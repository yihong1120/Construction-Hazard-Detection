local public, token, parent, generation, parent_index = KEYS[1], KEYS[2], KEYS[3], KEYS[4], KEYS[5]
local raw, renew = ARGV[1], ARGV[2] == '1'
if redis.call('GET', public) ~= token or redis.call('GET', token) ~= raw then return -1 end
local index_type = redis.call('TYPE', parent_index).ok
if index_type ~= 'none' and index_type ~= 'set' then return -3 end

if not renew then
    redis.call('DEL', public, token)
    redis.call('SREM', parent_index, token)
    return 0
end

local now = tonumber(redis.call('TIME')[1])
local cap = cjson.decode(raw)
local expiry = math.min(now + tonumber(ARGV[4]), tonumber(ARGV[3]))
if cap.user_active == false or (tonumber(cap.expires_at) or 0) <= now or expiry <= now
    or tonumber(cap.parent_generation or 0) ~= tonumber(redis.call('GET', generation) or '0')
    or redis.call('EXISTS', ARGV[5], ARGV[6]) > 0
    or (ARGV[7] == '1' and redis.call('EXISTS', parent) == 0) then return -1 end

local demands = cjson.decode(ARGV[8])
for _, demand in ipairs(demands) do
    if demand.language_position > 0 then
        local active, already = 0, false
        for index, keys in ipairs(demand.siblings) do
            if redis.call('EXISTS', keys[1], keys[2]) > 0 then
                active = active + 1
                if index == demand.language_position then already = true end
            end
        end
        if not already and active >= tonumber(ARGV[9]) then return -2 end
    end
end
cap.expires_at = expiry
cap.access_revocation_key = ARGV[5]
cap.oidc_revocation_key = ARGV[6]
local ttl = expiry - now
redis.call('SET', token, cjson.encode(cap), 'EX', ttl)
redis.call('EXPIRE', public, ttl)
redis.call('SADD', parent_index, token)
if redis.call('TTL', parent_index) < ttl then redis.call('EXPIRE', parent_index, ttl) end
for _, demand in ipairs(demands) do
    local remaining = redis.call('TTL', demand.key)
    redis.call('SET', demand.key, '1', 'EX', math.max(ttl, remaining))
end
return ttl
