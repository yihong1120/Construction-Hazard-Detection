#!/bin/bash
set -euo pipefail

template=/opt/keycloak/data/visionnaire-realm.template.json
realm_file=/opt/keycloak/data/import/visionnaire-realm.json
realm=visionnaire
browser_flow=visionnaire-browser
browser_forms_flow=visionnaire-browser-forms
kcadm=/opt/keycloak/bin/kcadm.sh

mkdir -p /opt/keycloak/data/import

if [[ ! -f "$realm_file" ]]; then
    sed \
        -e "s|__VISIONNAIRE_WEB_CLIENT_SECRET__|${KEYCLOAK_VISIONNAIRE_WEB_CLIENT_SECRET}|g" \
        -e "s|__USER_LINKER_CLIENT_SECRET__|${KEYCLOAK_USER_LINKER_CLIENT_SECRET}|g" \
        "$template" > "$realm_file"
fi

create_execution() {
    local flow_alias=$1
    local provider_id=$2
    local requirement=$3
    local execution_id

    execution_id=$("$kcadm" create \
        "authentication/flows/${flow_alias}/executions/execution" \
        --target-realm "$realm" \
        -s "provider=${provider_id}" \
        --id)
    "$kcadm" update "authentication/flows/${flow_alias}/executions" \
        --target-realm "$realm" \
        --body "{\"id\":\"${execution_id}\",\"requirement\":\"${requirement}\"}" \
        >/dev/null
}

execution_id_for_display_name() {
    local flow_alias=$1
    local expected_display_name=$2

    local line execution_id=''
    while IFS= read -r line; do
        if [[ $line =~ \"id\"\ :\ \"([^\"]+)\" ]]; then
            execution_id=${BASH_REMATCH[1]}
        elif [[ $line =~ \"displayName\"\ :\ \"([^\"]+)\" ]] \
            && [[ ${BASH_REMATCH[1]} == "$expected_display_name" ]]; then
            printf '%s\n' "$execution_id"
            return
        fi
    done < <("$kcadm" get "authentication/flows/${flow_alias}/executions" \
        --target-realm "$realm")
}

execution_id_for_provider() {
    local flow_alias=$1
    local expected_provider_id=$2
    local line execution_id=''
    while IFS= read -r line; do
        if [[ $line =~ \"id\"\ :\ \"([^\"]+)\" ]]; then
            execution_id=${BASH_REMATCH[1]}
        elif [[ $line =~ \"providerId\"\ :\ \"([^\"]+)\" ]] \
            && [[ ${BASH_REMATCH[1]} == "$expected_provider_id" ]]; then
            printf '%s\n' "$execution_id"
            return
        fi
    done < <("$kcadm" get "authentication/flows/${flow_alias}/executions" \
        --target-realm "$realm")
}

set_execution_requirement() {
    local flow_alias=$1
    local execution_id=$2
    local requirement=$3
    "$kcadm" update "authentication/flows/${flow_alias}/executions" \
        --target-realm "$realm" \
        --body "{\"id\":\"${execution_id}\",\"requirement\":\"${requirement}\"}" \
        >/dev/null
}

ensure_execution() {
    local flow_alias=$1
    local provider_id=$2
    local requirement=$3
    local execution_id
    execution_id=$(execution_id_for_provider "$flow_alias" "$provider_id")
    if [[ -z "$execution_id" ]]; then
        create_execution "$flow_alias" "$provider_id" "$requirement"
        return
    fi
    set_execution_requirement "$flow_alias" "$execution_id" "$requirement"
}

set_execution_priority() {
    local flow_alias=$1
    local execution_id=$2
    local requirement=$3
    local priority=$4
    "$kcadm" update "authentication/flows/${flow_alias}/executions" \
        --target-realm "$realm" \
        --body "{\"id\":\"${execution_id}\",\"requirement\":\"${requirement}\",\"priority\":${priority}}" \
        >/dev/null
}

is_enabled() {
    case "${1:-}" in
        1|true|TRUE|yes|YES|on|ON) return 0 ;;
        *) return 1 ;;
    esac
}

identity_provider_exists() {
    local alias=$1
    "$kcadm" get "identity-provider/instances/${alias}" \
        --target-realm "$realm" \
        >/dev/null 2>&1
}

disable_identity_provider() {
    local alias=$1
    if identity_provider_exists "$alias"; then
        "$kcadm" update "identity-provider/instances/${alias}" \
            --target-realm "$realm" \
            -s 'enabled=false' \
            >/dev/null
    fi
}

ensure_identity_provider() {
    local alias=$1
    local provider_id=$2
    if identity_provider_exists "$alias"; then
        return
    fi
    "$kcadm" create identity-provider/instances \
        --target-realm "$realm" \
        -s "alias=${alias}" \
        -s "providerId=${provider_id}" \
        -s 'enabled=true' \
        >/dev/null
}

configure_google_identity_provider() {
    local client_id=${KEYCLOAK_GOOGLE_CLIENT_ID:-}
    local client_secret=${KEYCLOAK_GOOGLE_CLIENT_SECRET:-}
    if ! is_enabled "${KEYCLOAK_GOOGLE_ENABLED:-false}"; then
        disable_identity_provider google
        return
    fi
    if [[ -z "$client_id" || -z "$client_secret" ]]; then
        echo 'Google social login is enabled but incomplete; disabling it' >&2
        disable_identity_provider google
        return
    fi

    ensure_identity_provider google google
    "$kcadm" update identity-provider/instances/google \
        --target-realm "$realm" \
        -s 'enabled=true' \
        -s 'displayName=Google' \
        -s 'hideOnLoginPage=false' \
        -s 'trustEmail=true' \
        -s 'storeToken=false' \
        -s 'addReadTokenRoleOnCreate=false' \
        -s 'firstBrokerLoginFlowAlias=first broker login' \
        -s "config.clientId=${client_id}" \
        -s "config.clientSecret=${client_secret}" \
        -s 'config.prompt=select_account' \
        -s 'config.syncMode=IMPORT' \
        >/dev/null
}

configure_apple_identity_provider() {
    local client_id=${KEYCLOAK_APPLE_CLIENT_ID:-}
    local client_secret=${KEYCLOAK_APPLE_CLIENT_SECRET:-}
    if ! is_enabled "${KEYCLOAK_APPLE_ENABLED:-false}"; then
        disable_identity_provider apple
        return
    fi
    if [[ -z "$client_id" || -z "$client_secret" ]]; then
        echo 'Apple social login is enabled but incomplete; disabling it' >&2
        disable_identity_provider apple
        return
    fi

    # Keycloak has a maintained generic OIDC broker. Apple is configured
    # through it because Apple client secrets are short-lived signed JWTs.
    ensure_identity_provider apple oidc
    "$kcadm" update identity-provider/instances/apple \
        --target-realm "$realm" \
        -s 'enabled=true' \
        -s 'displayName=Apple' \
        -s 'hideOnLoginPage=false' \
        -s 'trustEmail=true' \
        -s 'storeToken=false' \
        -s 'addReadTokenRoleOnCreate=false' \
        -s 'firstBrokerLoginFlowAlias=first broker login' \
        -s "config.clientId=${client_id}" \
        -s "config.clientSecret=${client_secret}" \
        -s 'config.authorizationUrl=https://appleid.apple.com/auth/authorize?response_mode=form_post' \
        -s 'config.tokenUrl=https://appleid.apple.com/auth/token' \
        -s 'config.jwksUrl=https://appleid.apple.com/auth/keys' \
        -s 'config.issuer=https://appleid.apple.com' \
        -s 'config.defaultScope=openid name email' \
        -s 'config.useJwksUrl=true' \
        -s 'config.validateSignature=true' \
        -s 'config.disableUserInfo=true' \
        -s 'config.syncMode=IMPORT' \
        >/dev/null
}

configure_social_identity_providers() {
    configure_google_identity_provider
    configure_apple_identity_provider
}

configure_visionnaire_browser_flow() {
    if ! "$kcadm" get authentication/flows --target-realm "$realm" \
        | grep -Fq "\"alias\" : \"${browser_flow}\""; then
        "$kcadm" create authentication/flows \
            --target-realm "$realm" \
            -s "alias=${browser_flow}" \
            -s 'description=Visionnaire password login protected by hCaptcha' \
            -s 'providerId=basic-flow' \
            -s 'topLevel=true' \
            -s 'builtIn=false' \
            >/dev/null

    fi

    # An existing Keycloak SSO session completes the alternative branch; a
    # fresh password login enters the forms branch below.
    ensure_execution "$browser_flow" auth-cookie ALTERNATIVE
    # A native Google/Apple assertion is first verified by Visionnaire and
    # then redeemed exactly once over loopback HMAC. Put this before browser
    # brokering/forms so Keycloak still issues a normal Code + PKCE response.
    ensure_execution "$browser_flow" visionnaire-native-social-exchange ALTERNATIVE
    # A social button sends kc_idp_hint.  The redirector must appear before
    # the password subflow so social sign-in never asks for a local password
    # or hCaptcha challenge.
    ensure_execution "$browser_flow" identity-provider-redirector ALTERNATIVE

    local forms_execution_id
    forms_execution_id=$(execution_id_for_display_name \
        "$browser_flow" "$browser_forms_flow")
    if [[ -z "$forms_execution_id" ]]; then
        "$kcadm" create \
            "authentication/flows/${browser_flow}/executions/flow" \
            --target-realm "$realm" \
            -s "alias=${browser_forms_flow}" \
            -s 'description=Password followed by mandatory hCaptcha' \
            -s 'type=basic-flow' \
            -s 'provider=basic-flow' \
            >/dev/null
        forms_execution_id=$(execution_id_for_display_name \
            "$browser_flow" "$browser_forms_flow")
    fi
    if [[ -z "$forms_execution_id" ]]; then
        echo 'Unable to resolve the Visionnaire browser forms execution' >&2
        return 1
    fi
    set_execution_requirement "$browser_flow" "$forms_execution_id" ALTERNATIVE

    local native_social_execution_id identity_provider_execution_id
    native_social_execution_id=$(execution_id_for_provider \
        "$browser_flow" visionnaire-native-social-exchange)
    if [[ -z "$native_social_execution_id" ]]; then
        echo 'Unable to resolve the native social exchange execution' >&2
        return 1
    fi
    identity_provider_execution_id=$(execution_id_for_provider \
        "$browser_flow" identity-provider-redirector)
    if [[ -z "$identity_provider_execution_id" ]]; then
        echo 'Unable to resolve the identity-provider redirector execution' >&2
        return 1
    fi
    set_execution_priority \
        "$browser_flow" "$native_social_execution_id" ALTERNATIVE 1
    set_execution_priority \
        "$browser_flow" "$identity_provider_execution_id" ALTERNATIVE 2
    set_execution_priority \
        "$browser_flow" "$forms_execution_id" ALTERNATIVE 3

    # Replace Keycloak's stock form with a Keycloak-first compatibility form.
    # It only proves a Visionnaire Argon2 password after the normal Keycloak
    # credential check fails, then immediately upgrades the credential and
    # disables the old verifier through a one-use loopback acknowledgement.
    ensure_execution "$browser_forms_flow" visionnaire-legacy-password REQUIRED
    local legacy_password_execution_id stock_password_execution_id
    legacy_password_execution_id=$(execution_id_for_provider \
        "$browser_forms_flow" visionnaire-legacy-password)
    stock_password_execution_id=$(execution_id_for_provider \
        "$browser_forms_flow" auth-username-password-form)
    if [[ -z "$legacy_password_execution_id" ]]; then
        echo 'Unable to resolve the legacy password migration execution' >&2
        return 1
    fi
    set_execution_priority \
        "$browser_forms_flow" "$legacy_password_execution_id" REQUIRED 0
    if [[ -n "$stock_password_execution_id" ]]; then
        set_execution_requirement \
            "$browser_forms_flow" "$stock_password_execution_id" DISABLED
    fi
    ensure_execution "$browser_forms_flow" visionnaire-hcaptcha REQUIRED

    # Match Visionnaire's pre-OIDC minimum while legacy passwords are
    # migrated. Raising this before the migration window ends would lock out
    # valid existing accounts whose passwords were accepted at 8–13 chars.
    "$kcadm" update "realms/${realm}" \
        -s "browserFlow=${browser_flow}" \
        -s 'loginTheme=visionnaire' \
        -s 'bruteForceProtected=true' \
        -s 'permanentLockout=false' \
        -s 'failureFactor=5' \
        -s 'waitIncrementSeconds=60' \
        -s 'maxFailureWaitSeconds=900' \
        -s 'maxDeltaTimeSeconds=43200' \
        -s 'passwordPolicy=length(8) and notUsername(undefined) and passwordHistory(5)' \
        >/dev/null
}

configure_mobile_client() {
    local client_id='' line
    while IFS= read -r line; do
        if [[ $line =~ \"id\"\ :\ \"([^\"]+)\" ]]; then
            client_id=${BASH_REMATCH[1]}
            break
        fi
    done < <("$kcadm" get clients --target-realm "$realm" \
        --query clientId=visionnaire-mobile \
        --fields id)
    if [[ -z "$client_id" ]]; then
        "$kcadm" create clients --target-realm "$realm" \
            -s 'clientId=visionnaire-mobile' \
            -s 'name=Visionnaire Flutter Mobile' \
            -s 'enabled=true' \
            -s 'protocol=openid-connect' \
            -s 'publicClient=true' \
            -s 'standardFlowEnabled=true' \
            -s 'implicitFlowEnabled=false' \
            -s 'directAccessGrantsEnabled=false' \
            -s 'serviceAccountsEnabled=false' \
            -s 'redirectUris=["com.changdar.visionnaire:/oauthredirect"]' \
            >/dev/null

        while IFS= read -r line; do
            if [[ $line =~ \"id\"\ :\ \"([^\"]+)\" ]]; then
                client_id=${BASH_REMATCH[1]}
                break
            fi
        done < <("$kcadm" get clients --target-realm "$realm" \
            --query clientId=visionnaire-mobile \
            --fields id)
    fi
    if [[ -z "$client_id" ]]; then
        echo 'Unable to resolve the Visionnaire mobile client id' >&2
        return 1
    fi
    "$kcadm" update "clients/${client_id}" --target-realm "$realm" \
        -s 'name=Visionnaire Flutter Mobile' \
        -s 'enabled=true' \
        -s 'publicClient=true' \
        -s 'standardFlowEnabled=true' \
        -s 'implicitFlowEnabled=false' \
        -s 'directAccessGrantsEnabled=false' \
        -s 'serviceAccountsEnabled=false' \
        -s 'redirectUris=["com.changdar.visionnaire:/oauthredirect"]' \
        -s 'attributes={"pkce.code.challenge.method":"S256","post.logout.redirect.uris":"com.changdar.visionnaire:/oauthredirect"}' \
        >/dev/null

    if ! "$kcadm" get "clients/${client_id}/protocol-mappers/models" \
        --target-realm "$realm" \
        | grep -Fq '"name" : "visionnaire-api-audience"'; then
        "$kcadm" create "clients/${client_id}/protocol-mappers/models" \
            --target-realm "$realm" \
            -s 'name=visionnaire-api-audience' \
            -s 'protocol=openid-connect' \
            -s 'protocolMapper=oidc-audience-mapper' \
            -s 'consentRequired=false' \
            -s 'config={"included.client.audience":"visionnaire-api","id.token.claim":"false","access.token.claim":"true"}' \
            >/dev/null
    fi
}

configure_user_linker_service_account() {
    # This client is never exposed to Flutter. Its credentials are held by the
    # Visionnaire API, which always derives the target Keycloak user from a
    # freshly verified bearer token. ``manage-users`` is needed by Keycloak's
    # supported federated-identity Admin endpoint; no client/realm management
    # role is assigned.
    "$kcadm" add-roles --target-realm "$realm" \
        --uusername service-account-visionnaire-user-linker \
        --cclientid realm-management \
        --rolename view-users \
        --rolename manage-users \
        >/dev/null
}

shutdown() {
    if [[ -n "${keycloak_pid:-}" ]] && kill -0 "$keycloak_pid" 2>/dev/null; then
        kill -TERM "$keycloak_pid"
        wait "$keycloak_pid"
    fi
}

trap shutdown TERM INT

/opt/keycloak/bin/kc.sh start --optimized --import-realm &
keycloak_pid=$!

# The realm may predate this image, so importing alone cannot reliably apply
# the new flow. Configure idempotently through the supported Admin API after
# the server is ready.
for _ in $(seq 1 60); do
    if "$kcadm" config credentials \
        --server "http://127.0.0.1:${KC_HTTP_PORT:-8080}${KC_HTTP_RELATIVE_PATH:-}" \
        --realm master \
        --user "$KC_BOOTSTRAP_ADMIN_USERNAME" \
        --password "$KC_BOOTSTRAP_ADMIN_PASSWORD" \
        >/dev/null 2>&1; then
        configure_visionnaire_browser_flow
        configure_mobile_client
        configure_user_linker_service_account
        configure_social_identity_providers
        wait "$keycloak_pid"
        exit $?
    fi
    if ! kill -0 "$keycloak_pid" 2>/dev/null; then
        wait "$keycloak_pid"
        exit $?
    fi
    sleep 1
done

echo 'Timed out waiting for Keycloak Admin API' >&2
shutdown
exit 1
