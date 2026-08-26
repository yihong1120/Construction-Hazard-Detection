package com.visionnaire.keycloak.legacypassword;

import java.util.List;
import org.keycloak.Config.Scope;
import org.keycloak.authentication.Authenticator;
import org.keycloak.authentication.AuthenticatorFactory;
import org.keycloak.models.AuthenticationExecutionModel;
import org.keycloak.models.KeycloakSession;
import org.keycloak.models.KeycloakSessionFactory;
import org.keycloak.provider.ProviderConfigProperty;

/** Registers the Keycloak-first legacy password migration form. */
public final class LegacyPasswordMigrationAuthenticatorFactory implements AuthenticatorFactory {
    private static final AuthenticationExecutionModel.Requirement[] REQUIREMENTS = {
        AuthenticationExecutionModel.Requirement.REQUIRED,
        AuthenticationExecutionModel.Requirement.DISABLED
    };

    @Override
    public String getDisplayType() {
        return "Visionnaire Legacy Password Migration";
    }

    @Override
    public String getReferenceCategory() {
        return "password";
    }

    @Override
    public boolean isConfigurable() {
        return false;
    }

    @Override
    public AuthenticationExecutionModel.Requirement[] getRequirementChoices() {
        return REQUIREMENTS;
    }

    @Override
    public boolean isUserSetupAllowed() {
        return false;
    }

    @Override
    public String getHelpText() {
        return "Upgrades a verified legacy Visionnaire password into Keycloak once.";
    }

    @Override
    public List<ProviderConfigProperty> getConfigProperties() {
        // URL and HMAC material stay only in the container environment.
        return List.of();
    }

    @Override
    public Authenticator create(KeycloakSession session) {
        return new LegacyPasswordMigrationAuthenticator(session);
    }

    @Override
    public void init(Scope config) {
        // All transitional settings are read from the process environment.
    }

    @Override
    public void postInit(KeycloakSessionFactory factory) {
        // No post-initialisation work is required.
    }

    @Override
    public void close() {
        // No factory-owned resources.
    }

    @Override
    public String getId() {
        return LegacyPasswordMigrationAuthenticator.PROVIDER_ID;
    }
}
