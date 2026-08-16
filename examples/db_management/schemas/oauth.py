from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import RootModel


class OAuthTokenResponse(BaseModel):
    """Represent tokens issued after a successful native OAuth exchange.

    Attributes:
        access_token: Newly issued access token.
        refresh_token: Newly issued refresh token.
        token_type: OAuth token type for the authorisation header.
        expires_in: Access-token lifetime in seconds.
    """

    access_token: str
    refresh_token: str
    token_type: Literal['Bearer'] = 'Bearer'
    expires_in: int


class NativeOAuthClients(RootModel[dict[str, list[str]]]):
    """Validate native OAuth client redirect-URI configuration.

    Attributes:
        root: Mapping from client identifier to permitted redirect URIs.
    """

    model_config = ConfigDict(strict=True)


class OAuthRequestParameters(RootModel[dict[str, str]]):
    """Validate string-only OAuth parameters from a form or JSON request.

    Attributes:
        root: Mapping of submitted OAuth parameter names to values.
    """

    model_config = ConfigDict(strict=True)


class AuthSessionUser(BaseModel):
    """Validate user identity stored in a BFF authentication session.

    Attributes:
        id: Database identifier of the authenticated user.
    """

    # Session stores may add non-security metadata without invalidating a user.
    model_config = ConfigDict(extra='allow', strict=True)

    id: int


class AuthSession(BaseModel):
    """Validate the authenticated BFF-session fields consumed by OAuth.

    Attributes:
        user: Authenticated user held by the session.
    """

    model_config = ConfigDict(extra='allow', strict=True)

    user: AuthSessionUser


class OAuthAuthorizationCode(BaseModel):
    """Validate a one-use authorisation-code record persisted in Redis.

    Attributes:
        user_id: Database identifier of the authorising user.
        client_id: Native OAuth client that requested the code.
        redirect_uri: Redirect URI bound to the authorisation code.
        code_challenge: PKCE challenge bound to the authorisation code.
    """

    # A code record must not silently accept unrecognised security fields.
    model_config = ConfigDict(extra='forbid', strict=True)

    user_id: int
    client_id: str = Field(min_length=1)
    redirect_uri: str = Field(min_length=1)
    code_challenge: str = Field(min_length=1)


class OAuthTokenRequest(BaseModel):
    """Validate fields shared by native OAuth grant requests.

    Attributes:
        grant_type: OAuth grant type requested by the client.
        client_id: Optional native OAuth client identifier.
    """

    model_config = ConfigDict(extra='allow', strict=True)

    grant_type: str = Field(min_length=1)
    client_id: str | None = None


class OAuthAuthorizationCodeRequest(OAuthTokenRequest):
    """Validate a PKCE authorisation-code exchange request.

    Attributes:
        grant_type: Fixed grant type for an authorisation-code exchange.
        redirect_uri: Redirect URI originally bound to the code.
        code: One-use authorisation code to exchange.
        code_verifier: PKCE verifier matching the stored code challenge.
    """

    grant_type: Literal['authorization_code']
    redirect_uri: str = Field(min_length=1)
    code: str = Field(min_length=1)
    code_verifier: str = Field(min_length=1)


class OAuthRefreshTokenRequest(OAuthTokenRequest):
    """Validate a native refresh-token rotation request.

    Attributes:
        grant_type: Fixed grant type for a refresh-token exchange.
        refresh_token: Non-empty refresh token to rotate.
    """

    grant_type: Literal['refresh_token']
    refresh_token: str = Field(min_length=1)


class OAuthRevocationRequest(BaseModel):
    """Validate a best-effort OAuth token-revocation request.

    Attributes:
        token: Optional token to revoke.
        token_type_hint: Optional hint describing the token type.
    """

    model_config = ConfigDict(extra='forbid', strict=True)

    token: str | None = None
    token_type_hint: str | None = None


class MeResponse(BaseModel):
    """Represent the public profile returned by native OAuth ``/me``.

    Attributes:
        id: Database identifier of the authenticated user.
        username: Unique account username.
        display_name: User's display name.
        role: Role granted to the user.
        group_id: Optional identifier of the user's group.
        status: Current account lifecycle status.
        feature_names: Features granted to the user.
    """

    id: int
    username: str
    display_name: str
    role: str
    group_id: int | None = None
    status: str
    feature_names: list[str] = Field(default_factory=list)
