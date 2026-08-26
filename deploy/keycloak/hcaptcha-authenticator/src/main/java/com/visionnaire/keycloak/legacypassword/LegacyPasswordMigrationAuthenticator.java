package com.visionnaire.keycloak.legacypassword;

import jakarta.ws.rs.core.MultivaluedMap;
import jakarta.ws.rs.core.Response;
import java.io.IOException;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.security.InvalidKeyException;
import java.security.NoSuchAlgorithmException;
import java.time.Duration;
import java.time.Instant;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.jboss.logging.Logger;
import org.keycloak.authentication.AuthenticationFlowContext;
import org.keycloak.authentication.AuthenticationFlowError;
import org.keycloak.authentication.authenticators.browser.UsernamePasswordForm;
import org.keycloak.authentication.authenticators.util.AuthenticatorUtils;
import org.keycloak.events.Errors;
import org.keycloak.models.KeycloakSession;
import org.keycloak.models.RealmModel;
import org.keycloak.models.UserCredentialModel;
import org.keycloak.models.UserModel;
import org.keycloak.protocol.oidc.OIDCLoginProtocol;
import org.keycloak.representations.idm.CredentialRepresentation;
import org.keycloak.services.managers.AuthenticationManager;

/**
 * Upgrades a successfully proved legacy Visionnaire password into Keycloak.
 *
 * <p>This is intentionally a temporary fallback of the ordinary Keycloak
 * username/password form. Keycloak always checks its own credential first.
 * Only after that fails does the plugin make a loopback HMAC request to prove
 * the old Argon2 hash. A successful proof is immediately saved as a Keycloak
 * password and acknowledged to Visionnaire, which disables the old hash.
 */
public final class LegacyPasswordMigrationAuthenticator extends UsernamePasswordForm {
    // Keycloak persists authenticator provider IDs in a varchar(36) column.
    public static final String PROVIDER_ID = "visionnaire-legacy-password";
    private static final Logger LOG = Logger.getLogger(LegacyPasswordMigrationAuthenticator.class);
    private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(5);
    private static final String SIGNATURE_CONTEXT = "visionnaire:legacy-password-migration:v1.";
    private static final Pattern MIGRATION_TOKEN = Pattern.compile(
        "\\\"migration_token\\\"\\s*:\\s*\\\"([A-Za-z0-9_-]{43,128})\\\""
    );
    private static final HttpClient HTTP = HttpClient.newBuilder()
        .connectTimeout(REQUEST_TIMEOUT)
        .followRedirects(HttpClient.Redirect.NEVER)
        .build();

    public LegacyPasswordMigrationAuthenticator(KeycloakSession session) {
        super(session);
    }

    @Override
    public boolean validatePassword(
        AuthenticationFlowContext context,
        UserModel user,
        MultivaluedMap<String, String> inputData,
        boolean clearUser
    ) {
        String password = inputData.getFirst(CredentialRepresentation.PASSWORD);
        if (password == null || password.isEmpty()) {
            // Preserve Keycloak's normal empty-password UI and event behavior.
            return super.validatePassword(context, user, inputData, clearUser);
        }
        if (isDisabledByBruteForce(context, user)) {
            return false;
        }
        if (user.credentialManager().isValid(UserCredentialModel.password(password))) {
            context.getAuthenticationSession().setAuthNote(
                AuthenticationManager.PASSWORD_VALIDATED,
                "true"
            );
            return true;
        }

        Configuration configuration = Configuration.fromEnvironment();
        if (configuration.isComplete() && migrate(user, password, configuration)) {
            context.getAuthenticationSession().setAuthNote(
                AuthenticationManager.PASSWORD_VALIDATED,
                "true"
            );
            return true;
        }
        return invalidCredentials(context, user, clearUser);
    }

    private boolean migrate(
        UserModel user,
        String password,
        Configuration configuration
    ) {
        String migrationToken = verifyLegacyPassword(
            user.getId(),
            password,
            configuration
        );
        if (migrationToken == null) {
            return false;
        }
        try {
            if (!user.credentialManager().updateCredential(
                UserCredentialModel.password(password)
            )) {
                LOG.warn("Could not store a migrated Keycloak password");
                return false;
            }
        } catch (RuntimeException exception) {
            // Never include an exception message: it can contain credential
            // implementation details and must not become a password oracle.
            LOG.warn("Could not store a migrated Keycloak password", exception);
            return false;
        }
        if (!completeMigration(user.getId(), migrationToken, configuration)) {
            // Keycloak now owns the password. The temporary local verifier is
            // intentionally retained on this rare failure rather than being
            // deleted before its replacement was durably acknowledged.
            LOG.warn("Legacy password migration acknowledgement was rejected");
            return false;
        }
        LOG.info("Legacy password migrated to Keycloak");
        return true;
    }

    private String verifyLegacyPassword(
        String keycloakSubject,
        String password,
        Configuration configuration
    ) {
        String body = "{\\\"keycloak_subject\\\":" + json(keycloakSubject)
            + ",\\\"password\\\":" + json(password) + "}";
        HttpResponse<String> response = post(configuration.verifyUri(), body, configuration);
        if (response == null || response.statusCode() != 200) {
            return null;
        }
        Matcher token = MIGRATION_TOKEN.matcher(response.body());
        return token.find() ? token.group(1) : null;
    }

    private boolean completeMigration(
        String keycloakSubject,
        String migrationToken,
        Configuration configuration
    ) {
        String body = "{\\\"keycloak_subject\\\":" + json(keycloakSubject)
            + ",\\\"migration_token\\\":" + json(migrationToken) + "}";
        HttpResponse<String> response = post(configuration.completeUri(), body, configuration);
        return response != null && response.statusCode() == 200;
    }

    private HttpResponse<String> post(
        URI uri,
        String body,
        Configuration configuration
    ) {
        try {
            String timestamp = Long.toString(Instant.now().getEpochSecond());
            HttpRequest request = HttpRequest.newBuilder(uri)
                .timeout(REQUEST_TIMEOUT)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .header("X-Visionnaire-Timestamp", timestamp)
                .header(
                    "X-Visionnaire-Legacy-Signature",
                    signature(configuration.sharedSecret(), timestamp, body)
                )
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
            return HTTP.send(request, HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8));
        } catch (IOException exception) {
            LOG.warn("Legacy password migration loopback request failed", exception);
            return null;
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            LOG.warn("Legacy password migration loopback request interrupted", exception);
            return null;
        } catch (RuntimeException | NoSuchAlgorithmException | InvalidKeyException exception) {
            LOG.warn("Legacy password migration request could not be prepared", exception);
            return null;
        }
    }

    private static String signature(String secret, String timestamp, String body)
        throws NoSuchAlgorithmException, InvalidKeyException {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
        byte[] message = (SIGNATURE_CONTEXT + timestamp + "." + body)
            .getBytes(StandardCharsets.UTF_8);
        byte[] digest = mac.doFinal(message);
        StringBuilder encoded = new StringBuilder(digest.length * 2);
        for (byte value : digest) {
            encoded.append(String.format("%02x", value & 0xff));
        }
        return encoded.toString();
    }

    private boolean invalidCredentials(
        AuthenticationFlowContext context,
        UserModel user,
        boolean clearUser
    ) {
        context.getEvent().user(user);
        context.getEvent().error(Errors.INVALID_USER_CREDENTIALS);
        AuthenticatorUtils.setupReauthenticationInUsernamePasswordFormError(context);
        Response challengeResponse = challenge(
            context,
            getDefaultChallengeMessage(context),
            "password"
        );
        context.failureChallenge(AuthenticationFlowError.INVALID_CREDENTIALS, challengeResponse);
        if (clearUser) {
            context.clearUser();
        }
        return false;
    }

    @Override
    public boolean configuredFor(KeycloakSession session, RealmModel realm, UserModel user) {
        return true;
    }

    @Override
    public void close() {
        // The shared HTTP client has no resources requiring explicit cleanup.
    }

    private static String json(String value) {
        StringBuilder escaped = new StringBuilder("\"");
        for (int index = 0; index < value.length(); index++) {
            char character = value.charAt(index);
            switch (character) {
                case '\\' -> escaped.append("\\\\");
                case '"' -> escaped.append("\\\"");
                case '\n' -> escaped.append("\\n");
                case '\r' -> escaped.append("\\r");
                case '\t' -> escaped.append("\\t");
                default -> {
                    if (character < 0x20) {
                        escaped.append(String.format("\\u%04x", (int) character));
                    } else {
                        escaped.append(character);
                    }
                }
            }
        }
        return escaped.append('"').toString();
    }

    private record Configuration(
        boolean enabled,
        URI verifyUri,
        URI completeUri,
        String sharedSecret
    ) {
        static Configuration fromEnvironment() {
            boolean enabled = "true".equalsIgnoreCase(env(
                "KC_LEGACY_PASSWORD_MIGRATION_ENABLED"
            ));
            String secret = env("KC_LEGACY_PASSWORD_MIGRATION_SHARED_SECRET");
            try {
                return new Configuration(
                    enabled,
                    URI.create(env("KC_LEGACY_PASSWORD_MIGRATION_VERIFY_URL")),
                    URI.create(env("KC_LEGACY_PASSWORD_MIGRATION_COMPLETE_URL")),
                    secret
                );
            } catch (IllegalArgumentException exception) {
                return new Configuration(enabled, null, null, secret);
            }
        }

        boolean isComplete() {
            return enabled
                && sharedSecret.length() >= 32
                && isLoopbackHttp(verifyUri)
                && isLoopbackHttp(completeUri);
        }

        private static boolean isLoopbackHttp(URI uri) {
            if (uri == null || !"http".equalsIgnoreCase(uri.getScheme())) {
                return false;
            }
            String host = uri.getHost();
            return host != null && (
                "127.0.0.1".equals(host)
                    || "::1".equals(host)
                    || "localhost".equals(host.toLowerCase(Locale.ROOT))
            );
        }

        private static String env(String name) {
            String value = System.getenv(name);
            return value == null ? "" : value.trim();
        }
    }
}
