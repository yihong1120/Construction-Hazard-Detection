#!/usr/bin/env bash
set -euo pipefail

# Match the integration server's client version without changing runner packages.
: "${RUNNER_TEMP:?RUNNER_TEMP is required}"
: "${GITHUB_PATH:?GITHUB_PATH is required}"
: "${POSTGRES_TOOLS_IMAGE:?POSTGRES_TOOLS_IMAGE is required}"

tool_dir="$RUNNER_TEMP/chd-postgres-tools"
mkdir -p "$tool_dir"
for command in pg_dump pg_restore; do
    cat > "$tool_dir/$command" <<'WRAPPER'
#!/usr/bin/env bash
set -euo pipefail
exec docker run --rm --network host --user "$(id -u):$(id -g)" \
    --volume /tmp:/tmp --env PGPASSWORD --env PGOPTIONS --env PGSSLPASSWORD \
    "$POSTGRES_TOOLS_IMAGE" "$(basename "$0")" "$@"
WRAPPER
    chmod 0755 "$tool_dir/$command"
done
printf '%s\n' "$tool_dir" >> "$GITHUB_PATH"
