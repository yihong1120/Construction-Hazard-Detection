package com.visionnaire.keycloak.nativesocial;

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
import java.util.Base64;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import javax.crypto.Mac;
import javax.crypto.spec.SecretKeySpec;
import org.jboss.logging.Logger;
import org.keycloak.authentication.AuthenticationFlowContext;
import org.keycloak.authentication.AuthenticationFlowError;
import org.keycloak.authentication.Authenticator;
import org.keycloak.models.ClientModel;
import org.keycloak.models.FederatedIdentityModel;
import org.keycloak.models.KeycloakSession;
import org.keycloak.models.RealmModel;
import org.keycloak.models.UserModel;

/**
 * Converts a verified native-provider proof into an ordinary Keycloak login.
 *
 * <p>Flutter never submits an external provider token to Keycloak. Visionnaire
 * validates the Google ID token or Apple code itself, writes only a one-use
 * provider subject to Redis, and this authenticator redeems it through a
 * loopback HMAC call. The eventual token response remains Keycloak's normal
 * Authorization Code + PKCE flow, rather than the deprecated external token
 * exchange grant.
 */
public final class NativeSocialExchangeAuthenticator implements Authenticator {
    public static final String PROVIDER_ID = "visionnaire-native-social-exchange";
    private static final Logger LOG = Logger.getLogger(NativeSocialExchangeAuthenticator.class);
    private static final String PARAMETER = "native_social_exchange";
    private static final String CODE_CHALLENGE_NOTE = "code_challenge";
    private static final Pattern TRANSACTION = Pattern.compile("^[A-Za-z0-9_-]{43,128}$");
    private static final Pattern PROVIDER = Pattern.compile(
        "\\\"provider\\\"\\s*:\\s*\\\"(google|apple)\\\""
    );
    private static final Pattern SUBJECT = Pattern.compile(
        "\\\"provider_subject_b64\\\"\\s*:\\s*\\\"([A-Za-z0-9_-]{1,1024})\\\""
    );
    private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(5);
    private static final HttpClient HTTP = HttpClient.newBuilder()
        .connectTimeout(REQUEST_TIMEOUT)
        .followRedirects(HttpClient.Redirect.NEVER)
        .build();

    @Override
    public void authenticate(AuthenticationFlowContext context) {
        String transactionId = context.getHttpRequest().getUri().getQueryParameters()
            .getFirst(PARAMETER);
        if (transactionId == null || transactionId.isBlank()) {
            context.attempted();
            return;
        }
        if (!TRANSACTION.matcher(transactionId).matches()) {
            fail(context, "Native social exchange transaction was malformed");
            return;
        }

        Configuration configuration = Configuration.fromEnvironment();
        if (!configuration.isComplete()) {
            fail(context, "Native social exchange is not configured");
            return;
        }
        ClientModel client = context.getAuthenticationSession().getClient();
        String redirectUri = context.getAuthenticationSession().getRedirectUri();
        String codeChallenge = context.getAuthenticationSession().getClientNote(CODE_CHALLENGE_NOTE);
        if (client == null || isBlank(redirectUri) || isBlank(codeChallenge)) {
            fail(context, "Native social exchange was not bound to a PKCE client");
            return;
        }

        RedeemedIdentity identity = redeem(
            configuration,
            transactionId,
            client.getClientId(),
            redirectUri,
            codeChallenge
        );
        if (identity == null) {
            fail(context, "Native social exchange redemption was rejected");
            return;
        }

        UserModel user = context.getSession().users().getUserByFederatedIdentity(
            context.getRealm(),
            new FederatedIdentityModel(
                identity.provider(),
                identity.providerSubject(),
                identity.providerSubject(),
                null
            )
        );
        if (user == null || !user.isEnabled()) {
            // Do not reveal whether an upstream social account exists. The
            // account must be linked by a recently reauthenticated Keycloak
            // user before a native assertion can sign it in.
            fail(context, "Native social identity is not linked to an active account");
            return;
        }
        context.setUser(user);
        context.success();
    }

    @Override
    public void action(AuthenticationFlowContext context) {
        // This execution has no HTML form and accepts no browser action.
        context.failure(AuthenticationFlowError.INVALID_CREDENTIALS);
    }

    private RedeemedIdentity redeem(
        Configuration configuration,
        String transactionId,
        String clientId,
        String redirectUri,
        String codeChallenge
    ) {
        try {
            String body = "{\"transaction_id\":" + json(transactionId)
                + ",\"client_id\":" + json(clientId)
                + ",\"redirect_uri\":" + json(redirectUri)
                + ",\"code_challenge\":" + json(codeChallenge) + "}";
            String timestamp = Long.toString(Instant.now().getEpochSecond());
            HttpRequest request = HttpRequest.newBuilder(configuration.redeemUri())
                .timeout(REQUEST_TIMEOUT)
                .header("Content-Type", "application/json")
                .header("Accept", "application/json")
                .header("X-Visionnaire-Timestamp", timestamp)
                .header(
                    "X-Visionnaire-Signature",
                    signature(configuration.sharedSecret(), timestamp, body)
                )
                .POST(HttpRequest.BodyPublishers.ofString(body, StandardCharsets.UTF_8))
                .build();
            HttpResponse<String> response = HTTP.send(
                request,
                HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
            );
            if (response.statusCode() != 200) {
                LOG.warnf("Native social exchange redemption rejected with HTTP status %d", response.statusCode());
                return null;
            }
            return parseIdentity(response.body());
        } catch (IOException exception) {
            LOG.warn("Native social exchange redemption request failed", exception);
            return null;
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            LOG.warn("Native social exchange redemption request interrupted", exception);
            return null;
        } catch (RuntimeException | NoSuchAlgorithmException | InvalidKeyException exception) {
            LOG.warn("Native social exchange redemption response could not be processed", exception);
            return null;
        }
    }

    private RedeemedIdentity parseIdentity(String body) {
        Matcher provider = PROVIDER.matcher(body);
        Matcher encodedSubject = SUBJECT.matcher(body);
        if (!provider.find() || !encodedSubject.find()) {
            return null;
        }
        try {
            String subject = new String(
                Base64.getUrlDecoder().decode(padBase64(encodedSubject.group(1))),
                StandardCharsets.UTF_8
            );
            if (subject.isBlank() || subject.getBytes(StandardCharsets.UTF_8).length > 512) {
                return null;
            }
            return new RedeemedIdentity(provider.group(1), subject);
        } catch (IllegalArgumentException exception) {
            return null;
        }
    }

    private static String signature(String secret, String timestamp, String body)
        throws NoSuchAlgorithmException, InvalidKeyException {
        Mac mac = Mac.getInstance("HmacSHA256");
        mac.init(new SecretKeySpec(secret.getBytes(StandardCharsets.UTF_8), "HmacSHA256"));
        byte[] value = mac.doFinal((timestamp + "." + body).getBytes(StandardCharsets.UTF_8));
        return Base64.getUrlEncoder().withoutPadding().encodeToString(value);
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

    private static String padBase64(String value) {
        int padding = (4 - value.length() % 4) % 4;
        return value + "=".repeat(padding);
    }

    private void fail(AuthenticationFlowContext context, String message) {
        // Never log parameters, upstream assertions, or HMAC material.
        LOG.warn(message);
        context.failure(AuthenticationFlowError.INVALID_CREDENTIALS);
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    @Override
    public boolean requiresUser() {
        return false;
    }

    @Override
    public boolean configuredFor(KeycloakSession session, RealmModel realm, UserModel user) {
        return true;
    }

    @Override
    public void setRequiredActions(KeycloakSession session, RealmModel realm, UserModel user) {
        // No credential is stored by this execution.
    }

    @Override
    public void close() {
        // The shared HTTP client has no resources that require explicit cleanup.
    }

    private record RedeemedIdentity(String provider, String providerSubject) {
    }

    private record Configuration(URI redeemUri, String sharedSecret) {
        static Configuration fromEnvironment() {
            String redeemUrl = env("KC_NATIVE_SOCIAL_EXCHANGE_REDEEM_URL");
            String secret = env("KC_NATIVE_SOCIAL_EXCHANGE_SHARED_SECRET");
            try {
                return new Configuration(URI.create(redeemUrl), secret);
            } catch (IllegalArgumentException exception) {
                return new Configuration(null, secret);
            }
        }

        boolean isComplete() {
            if (redeemUri == null || sharedSecret.length() < 32) {
                return false;
            }
            String host = redeemUri.getHost();
            return "http".equalsIgnoreCase(redeemUri.getScheme())
                && host != null
                && ("127.0.0.1".equals(host) || "::1".equals(host)
                    || "localhost".equals(host.toLowerCase(Locale.ROOT)));
        }

        private static String env(String name) {
            String value = System.getenv(name);
            return value == null ? "" : value.trim();
        }
    }
}
