FROM quay.io/keycloak/keycloak:26.7.2

USER root

COPY --chown=keycloak:keycloak deploy/keycloak/visionnaire-realm.template.json \
    /opt/keycloak/data/visionnaire-realm.template.json
COPY --chown=keycloak:keycloak deploy/keycloak/start.sh \
    /opt/keycloak/bin/visionnaire-start.sh

RUN chmod 0750 /opt/keycloak/bin/visionnaire-start.sh

USER keycloak

ENTRYPOINT ["/opt/keycloak/bin/visionnaire-start.sh"]
