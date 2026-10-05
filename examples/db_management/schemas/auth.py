from __future__ import annotations

from typing import Literal
from typing import NotRequired
from typing import TypedDict

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


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


class DeploymentInfo(BaseModel):
    """Managed deployment identity returned after authentication succeeds."""

    model_config = ConfigDict(extra='forbid', strict=True)

    deployment_id: str
    tenant_id: str
    config_revision: int = Field(ge=1)
