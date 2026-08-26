package com.visionnaire.keycloak.hcaptcha;

import java.util.List;
import org.keycloak.Config.Scope;
import org.keycloak.authentication.Authenticator;
import org.keycloak.authentication.AuthenticatorFactory;
import org.keycloak.models.AuthenticationExecutionModel;
import org.keycloak.models.KeycloakSession;
import org.keycloak.models.KeycloakSessionFactory;
import org.keycloak.provider.ProviderConfigProperty;

/** Factory registering the Visionnaire hCaptcha execution with Keycloak. */
public final class HCaptchaAuthenticatorFactory implements AuthenticatorFactory {
    public static final String PROVIDER_ID = "visionnaire-hcaptcha";
    private static final AuthenticationExecutionModel.Requirement[] REQUIREMENTS = {
        AuthenticationExecutionModel.Requirement.REQUIRED,
        AuthenticationExecutionModel.Requirement.DISABLED
    };
    private static final HCaptchaAuthenticator SINGLETON = new HCaptchaAuthenticator();

    @Override
    public String getDisplayType() {
        return "Visionnaire hCaptcha";
    }

    @Override
    public String getReferenceCategory() {
        return "hcaptcha";
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
        return "Requires a server-verified hCaptcha challenge before login completes.";
    }

    @Override
    public List<ProviderConfigProperty> getConfigProperties() {
        // Secret-bearing configuration must not be persisted in the realm DB.
        return List.of();
    }

    @Override
    public Authenticator create(KeycloakSession session) {
        return SINGLETON;
    }

    @Override
    public void init(Scope config) {
        // Runtime values are deliberately sourced only from process environment.
    }

    @Override
    public void postInit(KeycloakSessionFactory factory) {
        // No post-initialization work is required.
    }

    @Override
    public void close() {
        // The singleton has no factory-owned resources to close.
    }

    @Override
    public String getId() {
        return PROVIDER_ID;
    }
}
