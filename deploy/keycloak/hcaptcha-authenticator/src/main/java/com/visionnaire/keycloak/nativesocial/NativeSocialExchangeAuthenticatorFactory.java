package com.visionnaire.keycloak.nativesocial;

import java.util.List;
import org.keycloak.Config.Scope;
import org.keycloak.authentication.Authenticator;
import org.keycloak.authentication.AuthenticatorFactory;
import org.keycloak.models.AuthenticationExecutionModel;
import org.keycloak.models.KeycloakSession;
import org.keycloak.models.KeycloakSessionFactory;
import org.keycloak.provider.ProviderConfigProperty;

/** Factory registering the native social one-use proof authenticator. */
public final class NativeSocialExchangeAuthenticatorFactory implements AuthenticatorFactory {
    private static final AuthenticationExecutionModel.Requirement[] REQUIREMENTS = {
        AuthenticationExecutionModel.Requirement.ALTERNATIVE,
        AuthenticationExecutionModel.Requirement.DISABLED
    };
    private static final NativeSocialExchangeAuthenticator SINGLETON =
        new NativeSocialExchangeAuthenticator();

    @Override
    public String getDisplayType() {
        return "Visionnaire Native Social Exchange";
    }

    @Override
    public String getReferenceCategory() {
        return "native-social-exchange";
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
        return "Redeems a loopback-HMAC native provider proof bound to PKCE.";
    }

    @Override
    public List<ProviderConfigProperty> getConfigProperties() {
        // The HMAC secret belongs only in Keycloak/container environment.
        return List.of();
    }

    @Override
    public Authenticator create(KeycloakSession session) {
        return SINGLETON;
    }

    @Override
    public void init(Scope config) {
        // Runtime values are deliberately sourced only from environment.
    }

    @Override
    public void postInit(KeycloakSessionFactory factory) {
        // No post-initialization work is required.
    }

    @Override
    public void close() {
        // The singleton has no factory-owned resources.
    }

    @Override
    public String getId() {
        return NativeSocialExchangeAuthenticator.PROVIDER_ID;
    }
}
