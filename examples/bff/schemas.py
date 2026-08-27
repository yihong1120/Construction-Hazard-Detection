from __future__ import annotations

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from examples.db_management.schemas.auth import DeploymentInfo
from examples.db_management.schemas.auth import UserLogin


class UserSummary(BaseModel):
    """Represent the token-free user data exposed by a BFF session.

    Attributes:
        id: Database identifier of the authenticated user.
        username: Account username.
        display_name: Name displayed by the browser client.
        role: Role granted to the user.
        group_id: Optional assigned group identifier.
        status: Current account lifecycle status.
    """

    model_config = ConfigDict(extra='forbid', strict=True)

    id: int
    username: str
    display_name: str
    role: str
    group_id: int | None = None
    status: str


class BffSessionResponse(BaseModel):
    """Represent a browser session without exposing bearer credentials.

    Attributes:
        authenticated: Whether an active BFF session is present.
        user: Public authenticated user summary.
        feature_names: Features granted to the authenticated user.
    """

    # Redis session records contain encrypted tokens, which must never be
    # serialised even when this schema validates the complete record.
    model_config = ConfigDict(extra='ignore', strict=True)

    authenticated: bool = True
    user: UserSummary
    feature_names: list[str] = Field(default_factory=list)
    deployment: DeploymentInfo | None = None


class CsrfResponse(BaseModel):
    """Represent the CSRF token issued for an active BFF session.

    Attributes:
        csrf_token: Secret required by subsequent mutating browser requests.
    """

    csrf_token: str


class BffLogoutResponse(BaseModel):
    """Return the optional one-use URL for Keycloak browser logout."""

    global_logout_url: str | None = None


class BffLoginRequest(UserLogin):
    """Define browser login credentials for a token-private BFF session.

    Tokens are retained server-side and are never included in the response.
    """
