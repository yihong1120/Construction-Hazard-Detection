#!/bin/bash
set -euo pipefail

template=/opt/keycloak/data/visionnaire-realm.template.json
realm_file=/opt/keycloak/data/import/visionnaire-realm.json

mkdir -p /opt/keycloak/data/import

if [[ ! -f "$realm_file" ]]; then
    sed \
        -e "s|__VISIONNAIRE_WEB_CLIENT_SECRET__|${KEYCLOAK_VISIONNAIRE_WEB_CLIENT_SECRET}|g" \
        -e "s|__USER_LINKER_CLIENT_SECRET__|${KEYCLOAK_USER_LINKER_CLIENT_SECRET}|g" \
        "$template" > "$realm_file"
fi

exec /opt/keycloak/bin/kc.sh start --import-realm
