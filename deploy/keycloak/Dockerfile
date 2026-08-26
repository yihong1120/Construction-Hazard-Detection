FROM maven:3.9.11-eclipse-temurin-21 AS hcaptcha-builder

WORKDIR /workspace
COPY deploy/keycloak/hcaptcha-authenticator/pom.xml ./pom.xml
COPY deploy/keycloak/hcaptcha-authenticator/src ./src
RUN mvn --batch-mode --no-transfer-progress package -DskipTests

FROM quay.io/keycloak/keycloak:26.7.2

USER root

COPY --from=hcaptcha-builder \
    /workspace/target/hcaptcha-authenticator-1.0.0.jar \
    /opt/keycloak/providers/hcaptcha-authenticator.jar
COPY --chown=keycloak:keycloak deploy/keycloak/visionnaire-realm.template.json \
    /opt/keycloak/data/visionnaire-realm.template.json
COPY --chown=keycloak:keycloak deploy/keycloak/start.sh \
    /opt/keycloak/bin/visionnaire-start.sh
COPY --chown=keycloak:keycloak deploy/keycloak/themes/visionnaire \
    /opt/keycloak/themes/visionnaire

RUN chown keycloak:keycloak /opt/keycloak/providers/hcaptcha-authenticator.jar \
    && chmod 0750 /opt/keycloak/bin/visionnaire-start.sh

USER keycloak

ENV KC_DB=postgres \
    KC_HEALTH_ENABLED=true \
    KC_HTTP_RELATIVE_PATH=/keycloak

RUN /opt/keycloak/bin/kc.sh build

ENTRYPOINT ["/opt/keycloak/bin/visionnaire-start.sh"]
