package com.visionnaire.keycloak.hcaptcha;

import jakarta.ws.rs.core.MultivaluedMap;
import java.io.IOException;
import java.net.URI;
import java.net.URLEncoder;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.Locale;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import org.jboss.logging.Logger;
import org.keycloak.authentication.AuthenticationFlowContext;
import org.keycloak.authentication.AuthenticationFlowError;
import org.keycloak.authentication.Authenticator;
import org.keycloak.forms.login.LoginFormsProvider;
import org.keycloak.models.KeycloakSession;
import org.keycloak.models.RealmModel;
import org.keycloak.models.UserModel;

/**
 * A fail-closed hCaptcha authentication step.
 *
 * <p>The secret only exists in the Keycloak process environment.  The site
 * key is intentionally public and is passed to the rendered login template.
 * A challenge token is always verified server-side, with the expected
 * hostname and site key checked before the authentication flow may continue.
 */
public final class HCaptchaAuthenticator implements Authenticator {
    private static final Logger LOG = Logger.getLogger(HCaptchaAuthenticator.class);
    private static final String FORM_TEMPLATE = "hcaptcha.ftl";
    private static final String RESPONSE_PARAMETER = "h-captcha-response";
    private static final String VERIFY_URL = "https://api.hcaptcha.com/siteverify";
    private static final Duration REQUEST_TIMEOUT = Duration.ofSeconds(5);
    private static final Pattern SUCCESS = Pattern.compile(
        "\\\"success\\\"\\s*:\\s*true"
    );
    private static final Pattern HOSTNAME = Pattern.compile(
        "\\\"hostname\\\"\\s*:\\s*\\\"([^\\\"]+)\\\""
    );
    private static final Pattern SITE_KEY = Pattern.compile(
        "\\\"sitekey\\\"\\s*:\\s*\\\"([^\\\"]+)\\\""
    );
    private static final HttpClient HTTP = HttpClient.newBuilder()
        .connectTimeout(REQUEST_TIMEOUT)
        .followRedirects(HttpClient.Redirect.NEVER)
        .build();

    @Override
    public void authenticate(AuthenticationFlowContext context) {
        HCaptchaConfiguration configuration = HCaptchaConfiguration.fromEnvironment();
        if (!configuration.isComplete()) {
            LOG.error("Visionnaire hCaptcha is enabled without complete configuration");
            context.failure(AuthenticationFlowError.INTERNAL_ERROR);
            return;
        }
        challenge(context, configuration, null);
    }

    @Override
    public void action(AuthenticationFlowContext context) {
        HCaptchaConfiguration configuration = HCaptchaConfiguration.fromEnvironment();
        if (!configuration.isComplete()) {
            LOG.error("Visionnaire hCaptcha is enabled without complete configuration");
            context.failure(AuthenticationFlowError.INTERNAL_ERROR);
            return;
        }

        MultivaluedMap<String, String> form = context.getHttpRequest()
            .getDecodedFormParameters();
        String responseToken = form.getFirst(RESPONSE_PARAMETER);
        if (responseToken == null || responseToken.isBlank()) {
            challenge(context, configuration, "hcaptchaMissing");
            return;
        }

        String remoteAddress = context.getConnection().getRemoteAddr();
        if (!verify(configuration, responseToken, remoteAddress)) {
            challenge(context, configuration, "hcaptchaVerificationFailed");
            return;
        }
        context.success();
    }

    private void challenge(
        AuthenticationFlowContext context,
        HCaptchaConfiguration configuration,
        String errorMessageKey
    ) {
        LoginFormsProvider form = context.form()
            .setAttribute("hcaptchaSiteKey", configuration.siteKey());
        if (errorMessageKey != null) {
            form.setError(errorMessageKey);
        }
        context.failureChallenge(
            AuthenticationFlowError.INVALID_CREDENTIALS,
            form.createForm(FORM_TEMPLATE)
        );
    }

    private boolean verify(
        HCaptchaConfiguration configuration,
        String responseToken,
        String remoteAddress
    ) {
        try {
            String requestBody = "secret=" + encode(configuration.secret())
                + "&response=" + encode(responseToken)
                + "&sitekey=" + encode(configuration.siteKey());
            if (remoteAddress != null && !remoteAddress.isBlank()) {
                requestBody += "&remoteip=" + encode(remoteAddress);
            }

            HttpRequest request = HttpRequest.newBuilder(configuration.verifyUri())
                .timeout(REQUEST_TIMEOUT)
                .header("Content-Type", "application/x-www-form-urlencoded")
                .header("Accept", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(requestBody))
                .build();
            HttpResponse<String> response = HTTP.send(
                request,
                HttpResponse.BodyHandlers.ofString(StandardCharsets.UTF_8)
            );
            if (response.statusCode() != 200 || !SUCCESS.matcher(response.body()).find()) {
                LOG.warnf("hCaptcha verification rejected with HTTP status %d", response.statusCode());
                return false;
            }
            return responseMatchesConfiguration(configuration, response.body());
        } catch (IOException exception) {
            LOG.warn("hCaptcha verification request failed", exception);
            return false;
        } catch (InterruptedException exception) {
            Thread.currentThread().interrupt();
            LOG.warn("hCaptcha verification request interrupted", exception);
            return false;
        } catch (RuntimeException exception) {
            LOG.warn("hCaptcha verification response could not be processed", exception);
            return false;
        }
    }

    private boolean responseMatchesConfiguration(
        HCaptchaConfiguration configuration,
        String body
    ) {
        Matcher hostname = HOSTNAME.matcher(body);
        if (!hostname.find() || !configuration.expectedHostname().equals(
            hostname.group(1).toLowerCase(Locale.ROOT)
        )) {
            LOG.warn("hCaptcha response hostname did not match the configured public hostname");
            return false;
        }
        Matcher siteKey = SITE_KEY.matcher(body);
        if (!siteKey.find() || !configuration.siteKey().equals(siteKey.group(1))) {
            LOG.warn("hCaptcha response site key did not match the configured site key");
            return false;
        }
        return true;
    }

    private static String encode(String value) {
        return URLEncoder.encode(value, StandardCharsets.UTF_8);
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
        // This challenge does not persist a user credential or required action.
    }

    @Override
    public void close() {
        // The shared HTTP client has no resources that require explicit cleanup.
    }

    private record HCaptchaConfiguration(
        String siteKey,
        String secret,
        String expectedHostname,
        URI verifyUri
    ) {
        static HCaptchaConfiguration fromEnvironment() {
            String siteKey = env("KC_HCAPTCHA_SITE_KEY");
            String secret = env("KC_HCAPTCHA_SECRET_KEY");
            String expectedHostname = env("KC_HCAPTCHA_EXPECTED_HOSTNAME")
                .toLowerCase(Locale.ROOT);
            String verifyUrl = env("KC_HCAPTCHA_VERIFY_URL");
            if (verifyUrl.isBlank()) {
                verifyUrl = VERIFY_URL;
            }
            try {
                return new HCaptchaConfiguration(
                    siteKey,
                    secret,
                    expectedHostname,
                    URI.create(verifyUrl)
                );
            } catch (IllegalArgumentException exception) {
                LOG.error("KC_HCAPTCHA_VERIFY_URL is invalid", exception);
                return new HCaptchaConfiguration(siteKey, secret, expectedHostname, null);
            }
        }

        boolean isComplete() {
            return !siteKey.isBlank()
                && !secret.isBlank()
                && !expectedHostname.isBlank()
                && verifyUri != null
                && "https".equalsIgnoreCase(verifyUri.getScheme());
        }

        private static String env(String name) {
            String value = System.getenv(name);
            return value == null ? "" : value.trim();
        }
    }
}
