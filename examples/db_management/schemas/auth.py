from __future__ import annotations

from typing import Literal
from typing import NotRequired
from typing import TypedDict

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import EmailStr
from pydantic import Field


class DbUserInfo(TypedDict):
    """Define user fields cached from the database.

    Attributes:
        id: Database identifier of the user.
        username: Unique account username.
        role: Role assigned to the user.
        group_id: Optional identifier of the user's group.
        status: Current account lifecycle status.
    """

    id: int
    username: str
    role: str
    group_id: int | None
    status: str
    tenant_id: str


class UserCache(TypedDict):
    """Define the Redis representation of a user's security state.

    Attributes:
        db_user: Cached identity and authorisation fields.
        jti_list: Active access-token identifiers.
        jti_meta: Expiry times keyed by access-token identifier.
        refresh_tokens: Active raw refresh tokens for legacy cache lookups.
        refresh_token_hashes: Hashes of active refresh tokens.
        refresh_token_families: Refresh-token family keyed by token hash.
        feature_names: Feature names granted through the user's group.
    """

    db_user: DbUserInfo
    jti_list: list[str]
    jti_meta: dict[str, int]
    refresh_tokens: list[str]
    refresh_token_hashes: list[str]
    refresh_token_families: dict[str, str]
    feature_names: list[str]


class SubjectUsername(TypedDict):
    """Define the shared username claim in an application JWT subject.

    Attributes:
        username: Unique username of the token subject.
    """

    username: str


class AccessTokenSubject(SubjectUsername):
    """Define identity claims embedded in an access-token subject.

    Attributes:
        user_id: Database identifier of the authenticated user.
        role: Role granted when the token was issued.
        jti: Unique identifier for the individual access token.
        features: Feature names granted when the token was issued.
    """

    user_id: int
    role: str
    jti: str
    features: list[str]
    tenant_id: NotRequired[str]
    deployment_id: NotRequired[str]
    config_revision: NotRequired[int]


class RefreshTokenSubject(SubjectUsername):
    """Define refresh-token rotation claims.

    Attributes:
        family_id: Identifier shared by a refresh-token rotation family.
        token_id: Unique identifier for the refresh token.
    """

    family_id: str
    token_id: str
    tenant_id: NotRequired[str]
    deployment_id: NotRequired[str]
    config_revision: NotRequired[int]


class JwtSubjectModel(BaseModel):
    """Validate fields shared by application-issued JWT subjects.

    Attributes:
        username: Non-empty username of the token subject.
    """

    # Reject unknown claims before security-sensitive token processing.
    model_config = ConfigDict(extra='forbid', strict=True)

    username: str = Field(min_length=1)


class AccessTokenSubjectModel(JwtSubjectModel):
    """Validate the complete subject carried by an access token.

    Attributes:
        user_id: Database identifier of the authenticated user.
        role: Non-empty role granted when the token was issued.
        jti: Non-empty unique identifier for the access token.
        features: Feature names granted when the token was issued.
    """

    user_id: int
    role: str = Field(min_length=1)
    jti: str = Field(min_length=1)
    features: list[str]
    # These claims are optional in the schema only so old stored tokens can be
    # decoded for logout/revocation.  HTTP authentication requires all three.
    tenant_id: str | None = None
    deployment_id: str | None = None
    config_revision: int | None = Field(default=None, ge=1)


class RefreshTokenSubjectModel(JwtSubjectModel):
    """Validate the complete subject carried by a refresh token.

    Attributes:
        family_id: Non-empty identifier for the rotation family.
        token_id: Non-empty identifier for the refresh token.
    """

    family_id: str = Field(min_length=1)
    token_id: str = Field(min_length=1)
    tenant_id: str | None = None
    deployment_id: str | None = None
    config_revision: int | None = Field(default=None, ge=1)


class ProviderClaims(BaseModel):
    """Validate OpenID Connect claims used by provider sign-in flows.

    Attributes:
        sub: Stable non-empty provider subject identifier.
        aud: Optional intended OAuth client audience.
        email: Optional email address supplied by the provider.
        email_verified: Whether the provider verified the email address.
        is_private_email: Whether the provider supplied a relay address.
        nonce: Optional request nonce returned by the provider.
        name: Optional full display name.
        given_name: Optional given name.
        family_name: Optional family name.
        device_lang: Optional device language supplied by the client.
    """

    # Providers can add standard claims, but known fields remain strict.
    model_config = ConfigDict(extra='allow', strict=True)

    sub: str = Field(min_length=1)
    aud: str | None = None
    email: str | None = None
    email_verified: bool = False
    is_private_email: bool = False
    nonce: str | None = None
    name: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    device_lang: str | None = None


class AppleTokenExchangeResponse(BaseModel):
    """Validate the Apple token-exchange field used by this application.

    Attributes:
        id_token: Optional OpenID Connect identity token returned by Apple.
    """

    model_config = ConfigDict(extra='allow', strict=True)

    id_token: str | None = None


class JWTPayloadBase(TypedDict, total=False):
    """Define optional registered claims shared by JWT payloads.

    Attributes:
        exp: Expiry time expressed as a Unix timestamp.
        iat: Issue time expressed as a Unix timestamp.
        nbf: Earliest valid time expressed as a Unix timestamp.
        iss: Optional issuer identifier.
        aud: Optional audience identifier.
    """

    exp: NotRequired[int]
    iat: NotRequired[int]
    nbf: NotRequired[int]
    iss: NotRequired[str]
    aud: NotRequired[str]


class RefreshTokenPayload(JWTPayloadBase, total=False):
    """Define application claims within a refresh-token payload.

    Attributes:
        subject: Identity and rotation claims for the refresh token.
    """

    subject: RefreshTokenSubject


class UserLogin(BaseModel):
    """Define credentials submitted to the password login endpoint.

    Attributes:
        identifier: Username or email address identifying the account.
        password: Account password.
        hcaptcha_token: Optional hCaptcha response token.
    """

    identifier: str
    password: str
    hcaptcha_token: str | None = None


class VerifyEmailRequest(BaseModel):
    """Define an email-verification request.

    Attributes:
        token: Raw one-time token received in the verification link.
    """

    token: str


class ResendVerificationRequest(BaseModel):
    """Define a request to resend an email-verification link.

    Attributes:
        email: Account email address to receive a new link.
    """

    email: EmailStr


class AuthMessageResponse(BaseModel):
    """Represent a response from an authentication lifecycle operation.

    Attributes:
        message: User-facing result message.
        code: Optional stable machine-readable result code.
        status: Optional account lifecycle status.
    """

    message: str
    code: str | None = None
    status: str | None = None


class LegalConsentFields(BaseModel):
    """Define legal consents supplied with provider account registration.

    Attributes:
        accepted_terms: Whether the general terms were accepted.
        terms_version: Accepted general terms version.
        privacy_version: Accepted privacy notice version.
        notification_consent: Whether notifications were accepted.
        ai_terms_accepted: Whether AI-specific terms were accepted.
        ai_terms_version: Accepted AI-specific terms version.
    """

    accepted_terms: bool = False
    terms_version: str | None = None
    privacy_version: str | None = None
    notification_consent: bool = False
    ai_terms_accepted: bool = False
    ai_terms_version: str | None = None


class GoogleAuthRequest(LegalConsentFields):
    """Define a Google Sign-In authentication request.

    Attributes:
        id_token: Google OpenID Connect identity token.
        email: Optional email asserted by the client.
        display_name: Optional display name asserted by the client.
        device_lang: Optional device language supplied by the client.
    """

    id_token: str
    email: str | None = None
    display_name: str | None = None
    device_lang: str | None = None


class AppleAuthRequest(LegalConsentFields):
    """Define a Sign in with Apple authentication request.

    Attributes:
        identity_token: Optional Apple OpenID Connect identity token.
        authorization_code: Apple authorisation code to exchange.
        email: Optional email supplied only during the initial Apple sign-in.
        given_name: Optional given name supplied by Apple.
        family_name: Optional family name supplied by Apple.
        nonce: Optional nonce expected in the identity token.
        device_lang: Optional device language supplied by the client.
    """

    identity_token: str | None = None
    authorization_code: str
    email: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    nonce: str | None = None
    device_lang: str | None = None


NativeSocialProvider = Literal['google', 'apple']


class NativeSocialExchangeBeginRequest(BaseModel):
    """Start a short-lived native social assertion exchange.

    The client supplies its normal Authorization Code + PKCE parameters before
    it talks to Google or Apple.  The server returns a nonce that must be
    supplied to the native provider SDK and binds the resulting identity
    assertion to this exact future Keycloak authorisation request.
    """

    model_config = ConfigDict(extra='forbid', strict=True)

    provider: NativeSocialProvider
    client_id: str = Field(min_length=1, max_length=128)
    redirect_uri: str = Field(min_length=1, max_length=2048)
    code_challenge: str = Field(
        min_length=43,
        max_length=128,
        pattern=r'^[A-Za-z0-9_-]+$',
    )
    code_challenge_method: Literal['S256']
    state: str = Field(min_length=1, max_length=2048)


class NativeSocialExchangeBeginResponse(BaseModel):
    """Return the opaque exchange ID and provider nonce to Flutter."""

    transaction_id: str = Field(min_length=43, max_length=128)
    nonce: str = Field(min_length=43, max_length=128)
    expires_in: int = Field(ge=30, le=300)


class NativeSocialCredential(BaseModel):
    """Provider credentials returned by an official native SDK.

    Google completes with ``id_token``.  Apple must include its one-use
    ``authorization_code`` and may additionally include the identity token
    returned by the platform API.
    """

    model_config = ConfigDict(extra='forbid', strict=True)

    id_token: str | None = Field(default=None, min_length=1, max_length=16384)
    authorization_code: str | None = Field(
        default=None,
        min_length=1,
        max_length=4096,
    )


class NativeSocialExchangeCompleteRequest(NativeSocialCredential):
    """Complete a PKCE-bound native social exchange."""

    transaction_id: str = Field(min_length=43, max_length=128)


class NativeSocialExchangeCompleteResponse(BaseModel):
    """Return the Keycloak URL that creates a normal OIDC code."""

    authorization_url: str = Field(min_length=1, max_length=4096)
    expires_in: int = Field(ge=30, le=300)


class NativeSocialLinkBeginRequest(BaseModel):
    """Start a freshly-authenticated social-identity linking transaction."""

    model_config = ConfigDict(extra='forbid', strict=True)

    provider: NativeSocialProvider


class NativeSocialLinkBeginResponse(BaseModel):
    """Return the nonce bound to a recent Keycloak session."""

    transaction_id: str = Field(min_length=43, max_length=128)
    nonce: str = Field(min_length=43, max_length=128)
    expires_in: int = Field(ge=30, le=300)


class NativeSocialLinkCompleteRequest(NativeSocialCredential):
    """Submit the provider proof that will be linked to the current account."""

    transaction_id: str = Field(min_length=43, max_length=128)


class NativeSocialEmailLinkConfirmRequest(BaseModel):
    """Confirm a verified-email link after fresh Keycloak authentication."""

    model_config = ConfigDict(extra='forbid', strict=True)

    transaction_id: str = Field(min_length=43, max_length=128)


class NativeSocialLinkResponse(BaseModel):
    """Report whether provider subject was newly linked or already present."""

    provider: NativeSocialProvider
    status: Literal['linked', 'already_linked']


class IdentityRead(BaseModel):
    """Represent an external identity linked to a user account.

    Attributes:
        id: Database identifier of the linked identity.
        provider: External identity provider name.
        email: Optional email held by the provider.
        display_name: Optional display name held by the provider.
        linked_at: ISO 8601 time at which the identity was linked.
    """

    id: int
    provider: str
    email: str | None = None
    display_name: str | None = None
    linked_at: str


class IdentityListResponse(BaseModel):
    """Represent all authentication methods linked to the current user.

    Attributes:
        identities: Linked external identities.
        has_password: Whether the account also has a password credential.
    """

    identities: list[IdentityRead]
    has_password: bool


class LogoutRequest(BaseModel):
    """Define a logout request.

    Attributes:
        refresh_token: Optional refresh token to revoke with the session.
    """

    refresh_token: str | None = None


class RefreshRequest(BaseModel):
    """Define a token-refresh request.

    Attributes:
        refresh_token: Optional refresh token supplied outside a web cookie.
    """

    refresh_token: str | None = None


class TokenPairData(TypedDict):
    """Define fields accepted when building a token-pair response.

    Attributes:
        access_token: Newly issued access token.
        feature_names: Features granted to the authenticated user.
        refresh_token: Newly issued refresh token.
        username: Optional account username.
        role: Optional role granted to the user.
        user_id: Optional database identifier of the user.
        group_id: Optional identifier of the user's group.
    """

    access_token: str
    feature_names: list[str]
    refresh_token: str
    username: NotRequired[str]
    role: NotRequired[str]
    user_id: NotRequired[int]
    group_id: NotRequired[int | None]
    deployment: NotRequired[DeploymentInfo]


class DeploymentInfo(BaseModel):
    """Managed deployment identity returned after authentication succeeds."""

    model_config = ConfigDict(extra='forbid', strict=True)

    deployment_id: str
    tenant_id: str
    config_revision: int = Field(ge=1)


class TokenPair(BaseModel):
    """Represent issued tokens and the authenticated user's public details.

    Attributes:
        access_token: Newly issued access token.
        refresh_token: Optional newly issued refresh token.
        username: Optional authenticated account username.
        role: Optional role granted to the user.
        user_id: Optional database identifier of the user.
        group_id: Optional identifier of the user's group.
        feature_names: Features granted to the authenticated user.
    """

    access_token: str
    refresh_token: str | None = None
    username: str | None = None
    role: str | None = None
    user_id: int | None = None
    group_id: int | None = None
    feature_names: list[str] = []
    deployment: DeploymentInfo | None = None
