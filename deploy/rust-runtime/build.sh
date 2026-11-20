#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'Usage: %s [--tensorrt] RELEASE_DIR\n' "$0"
}

variant=cpu
features=()
case "${1:-}" in
    -h|--help)
        usage
        exit 0
        ;;
    --tensorrt)
        variant=tensorrt
        features=(--features tensorrt)
        shift
        ;;
esac
if [[ $# != 1 || -z $1 || $1 == -* ]]; then
    usage >&2
    exit 2
fi

package_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo_dir=$(cd -- "$package_dir/../.." && pwd -P)
release_dir=$1
if [[ -e $release_dir || -L $release_dir ]]; then
    printf 'Release path already exists: %s\n' "$release_dir" >&2
    exit 2
fi

# Keep the two variants separate from regular developer builds and one another.
target_dir=${CARGO_TARGET_DIR:-$repo_dir/native/pipeline/target/rust-runtime}
mkdir -p -- "$target_dir"
target_dir=$(cd -- "$target_dir" && pwd -P)
for crate in pipeline api; do
    dependencies=$(cargo tree --locked --manifest-path "$repo_dir/native/$crate/Cargo.toml" \
        "${features[@]}")
    if grep -Eq '[[:space:]]pyo3([-_]|[[:space:]])' <<< "$dependencies"; then
        printf 'construction-%s unexpectedly depends on PyO3\n' "$crate" >&2
        exit 1
    fi
done
cargo build --locked --release --bin construction-pipeline \
    --manifest-path "$repo_dir/native/pipeline/Cargo.toml" \
    --target-dir "$target_dir/$variant" "${features[@]}"
cargo build --locked --release --bin construction-api --bin construction-website-access \
    --manifest-path "$repo_dir/native/api/Cargo.toml" \
    --target-dir "$target_dir/api-$variant" "${features[@]}"

release_parent=$(dirname -- "$release_dir")
mkdir -p -- "$release_parent"
stage=$(mktemp -d -- "$release_parent/.construction-rust-preview.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
chmod 0755 "$stage"
install -d "$stage/bin" "$stage/config" "$stage/systemd" "$stage/docs/pipeline"
install -m 0755 "$target_dir/$variant/release/construction-pipeline" "$stage/bin/"
install -m 0755 "$target_dir/api-$variant/release/construction-api" "$stage/bin/"
install -m 0755 "$target_dir/api-$variant/release/construction-website-access" "$stage/bin/"
install -m 0644 "$package_dir/api.example.json" "$package_dir/api.migration.example.json" "$stage/config/"
install -m 0644 "$package_dir/systemd/"*.service "$stage/systemd/"
install -m 0644 "$package_dir/README.md" "$package_dir/README-zh-tw.md" "$stage/"
install -m 0644 "$repo_dir/LICENSE" "$stage/"
install -m 0644 "$repo_dir/native/pipeline/"{README.md,README-zh-tw.md,POSTGRES_OUTBOX.md} \
    "$stage/docs/pipeline/"

mv --no-clobber --no-target-directory -- "$stage" "$release_dir"
if [[ -d $stage ]]; then
    printf 'Release path was created during build: %s\n' "$release_dir" >&2
    exit 2
fi
printf 'Built Rust preview release (%s): %s\n' "$variant" "$release_dir"
