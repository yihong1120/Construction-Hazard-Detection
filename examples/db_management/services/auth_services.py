from __future__ import annotations

import datetime
from functools import partial
from typing import cast
from uuid import uuid4

import httpx
import jwt
from fastapi import HTTPException
from fastapi import Request
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from examples.auth.cache import rate_limiter_service
from examples.auth.config import Settings
from examples.auth.deployment_context import DeploymentBinding
from examples.auth.deployment_context import resolve_request_deployment
from examples.auth.identity_provider import require_local_login
from examples.auth.jwt_config import access_token_subject_from_payload
from examples.auth.jwt_config import jwt_access
from examples.auth.jwt_config import jwt_refresh
from examples.auth.jwt_config import refresh_token_subject_from_payload
from examples.auth.models import Feature
from examples.auth.models import group_features_table
from examples.auth.models import User
from examples.auth.models import USER_STATUS_ACTIVE
from examples.auth.models import USER_STATUS_EMAIL_UNVERIFIED
from examples.auth.models import USER_STATUS_PENDING_ADMIN_APPROVAL
from examples.auth.models import USER_STATUS_REJECTED
from examples.auth.models import USER_STATUS_SUSPENDED
from examples.auth.models import UserProfile
from examples.auth.token_cleanup import prune_user_cache
from examples.auth.token_revocation import AccessTokenRevocationPayload
from examples.auth.token_revocation import revoke_access_token
from examples.auth.token_revocation import revoke_access_token_jtis
from examples.db_management.schemas.auth import DbUserInfo
from examples.db_management.schemas.auth import RefreshRequest
from examples.db_management.schemas.auth import RefreshTokenPayload
from examples.db_management.schemas.auth import TokenPairData
from examples.db_management.schemas.auth import UserCache
from examples.db_management.schemas.auth import UserLogin
from examples.db_management.services.auth_login_guard import check_login_guard
from examples.db_management.services.auth_login_guard import clear_login_guard
from examples.db_management.services.auth_login_guard import (
    clear_login_guard_for_identifier,
)
from examples.db_management.services.auth_login_guard import (
    record_failed_login,
)
from examples.db_management.services.auth_refresh_state import (
    _cache_contains_refresh_token,
)
from examples.db_management.services.auth_refresh_state import (
    _refresh_family_revoked_key,
)
from examples.db_management.services.auth_refresh_state import (
    _remove_refresh_token_from_cache,
)
from examples.db_management.services.auth_refresh_state import (
    _store_refresh_token_in_cache,
)
from examples.db_management.services.auth_refresh_state import (
    consume_refresh_token_state,
)
from examples.db_management.services.auth_refresh_state import (
    register_refresh_token_state,
)
from examples.db_management.services.auth_refresh_state import (
    revoke_refresh_family,
)
from examples.db_management.services.auth_refresh_state import (
    revoke_user_access_tokens,
)
from examples.db_management.services.auth_token_issuer import (
    issue_access_token,
)
from examples.db_management.services.auth_token_issuer import (
    issue_refresh_token,
)
from src.http_client_pool import get_application_http_client

# Configuration settings for JWT authentication
settings = Settings()
SECRET_KEY = settings.authjwt_secret_key
HCAPTCHA_SECRET_KEY = settings.hcaptcha_secret_key
HCAPTCHA_SITE_KEY = settings.hcaptcha_site_key
HCAPTCHA_BYPASS_KEY = settings.hcaptcha_bypass_key
HCAPTCHA_VERIFY_URL = 'https://api.hcaptcha.com/siteverify'
ALGORITHM = 'HS256'
SUPER_ADMIN = 'ChangDar'
ACCESS_TTL = datetime.timedelta(minutes=15)  # Short-lived API capability
REFRESH_TTL = datetime.timedelta(days=30)  # Refresh token expiry time


async def clear_login_guard_for_identifiers(
    redis_pool: Redis,
    identifiers: list[str],
) -> None:
    """Clear login guards for a user's known aliases."""
    seen: set[str] = set()
    for identifier in identifiers:
        normalized = identifier.strip().lower()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        await clear_login_guard_for_identifier(redis_pool, normalized)


async def _load_feature_names(
    db: AsyncSession,
    group_id: int | None,
) -> list[str]:
    """Retrieve feature names associated with a specified group.

    Args:
        db (AsyncSession): Database session for queries.
        group_id (Optional[int]): ID of the group.

    Returns:
        list[str]: List of feature names linked to the group.
    """
    if group_id is None:
        return []

    rows = await db.execute(
        select(Feature.feature_name)
        .join(
            group_features_table,
            Feature.id == group_features_table.c.feature_id,
        )
        .where(group_features_table.c.group_id == group_id),
    )

    return [r.feature_name for r in rows]


async def _authenticate(
    db: AsyncSession,
    identifier: str,
    password: str,
) -> User:
    """Authenticate user credentials and verify active status.

    Args:
        db (AsyncSession): Database session for queries.
        identifier (str): User's username or profile e-mail address.
        password (str): User's password.

    Returns:
        User: Authenticated user object.

    Raises:
        HTTPException: When credentials are incorrect or user is inactive.
    """
    login_identifier = identifier.strip()
    user = await db.scalar(
        select(User).where(User.username == login_identifier),
    )

    if user is None:
        user = await db.scalar(
            select(User)
            .join(UserProfile, UserProfile.user_id == User.id)
            .where(UserProfile.email == login_identifier.lower()),
        )

    if not user or not await user.check_password(password):
        raise HTTPException(
            status_code=401,
            detail='Wrong username/e-mail or password',
        )

    if user.status == USER_STATUS_EMAIL_UNVERIFIED:
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'email_unverified',
                'status': user.status,
                'message': 'Please verify your email before logging in.',
            },
        )

    if user.status == USER_STATUS_PENDING_ADMIN_APPROVAL:
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'pending_admin_approval',
                'status': user.status,
                'message': 'Account is waiting for administrator approval.',
            },
        )

    if user.status == USER_STATUS_REJECTED:
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'account_rejected',
                'status': user.status,
                'message': 'Account application was rejected.',
            },
        )

    if user.status == USER_STATUS_SUSPENDED:
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'account_suspended',
                'status': user.status,
                'message': 'Account is suspended.',
            },
        )

    if user.status != USER_STATUS_ACTIVE:
        raise HTTPException(
            status_code=403,
            detail={'code': 'user_not_active', 'status': user.status},
        )

    return user


async def _verify_hcaptcha(
    hcaptcha_token: str | None,
    hcaptcha_bypass_key: str | None = None,
) -> None:
    """Verify an hCaptcha token before credential authentication.

    Args:
        hcaptcha_token: Client hCaptcha response token.
        hcaptcha_bypass_key: Trusted server-side bypass credential.

    Raises:
        HTTPException: If hCaptcha validation is required and fails.
    """
    if not settings.hcaptcha_enabled:
        return

    if (
        HCAPTCHA_BYPASS_KEY
        and hcaptcha_bypass_key
        and hcaptcha_bypass_key == HCAPTCHA_BYPASS_KEY
    ):
        return

    token = (hcaptcha_token or '').strip()
    if not token:
        raise HTTPException(status_code=400, detail='hCaptcha token required')

    if not HCAPTCHA_SECRET_KEY or not HCAPTCHA_SITE_KEY:
        raise HTTPException(status_code=500, detail='hCaptcha not configured')

    try:
        client = await get_application_http_client(
            'hcaptcha',
            timeout=10.0,
        )
        if client is not None:
            response = await client.post(
                HCAPTCHA_VERIFY_URL,
                data={
                    'secret': HCAPTCHA_SECRET_KEY,
                    'response': token,
                    'sitekey': HCAPTCHA_SITE_KEY,
                },
            )
            response.raise_for_status()
            result = response.json()
        else:
            async with httpx.AsyncClient(timeout=10.0) as ephemeral_client:
                response = await ephemeral_client.post(
                    HCAPTCHA_VERIFY_URL,
                    data={
                        'secret': HCAPTCHA_SECRET_KEY,
                        'response': token,
                        'sitekey': HCAPTCHA_SITE_KEY,
                    },
                )
                response.raise_for_status()
                result = response.json()
    except (httpx.HTTPError, ValueError):
        raise HTTPException(
            status_code=403,
            detail='hCaptcha verification failed',
        )

    if result.get('success') is not True:
        raise HTTPException(
            status_code=403,
            detail='hCaptcha verification failed',
        )


async def verify_refresh_token(
    refresh_token: str,
    redis_pool: Redis,
    deployment: DeploymentBinding | None = None,
) -> RefreshTokenPayload:
    """Verify and decode a JWT refresh token.

    Args:
        refresh_token (str): Refresh token to validate.
        redis_pool (Redis): Redis connection pool for caching.

    Returns:
        RefreshTokenPayload: Decoded token payload.

    Raises:
        HTTPException: If token is invalid, expired, or not recognised.
    """
    try:
        # Decode and verify JWT refresh token and its purpose.
        payload = jwt_refresh.decode_token(
            refresh_token,
            expected_issuer=deployment.issuer if deployment else None,
            expected_audience=deployment.audience if deployment else None,
        )
        subject = refresh_token_subject_from_payload(payload)
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail='Refresh token has expired',
        )
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=401, detail='Invalid refresh token')

    username = subject['username']
    if deployment is not None and (
        subject.get('tenant_id') != str(deployment.tenant_id)
        or subject.get('deployment_id') != str(deployment.deployment_id)
        or subject.get('config_revision') != deployment.config_revision
    ):
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'deployment_configuration_changed',
                'message': 'Deployment configuration changed; sign in again.',
            },
        )
    family_id = subject['family_id']
    if await redis_pool.get(
        _refresh_family_revoked_key(family_id),
    ):
        raise HTTPException(status_code=401, detail='Refresh token reused')

    # Retrieve user's data from Redis cache (and prune expired entries)
    await prune_user_cache(redis_pool, username)
    user_data = cast(
        UserCache | None,
        await rate_limiter_service.get_user_data(redis_pool, username),
    )
    if not user_data or not _cache_contains_refresh_token(
        user_data, refresh_token,
    ):
        if family_id:
            await revoke_refresh_family(
                redis_pool,
                family_id,
                refresh_ttl=REFRESH_TTL,
            )
            await revoke_user_access_tokens(
                redis_pool,
                username,
                get_user_data_fn=rate_limiter_service.get_user_data,
                revoke_access_token_jtis_fn=revoke_access_token_jtis,
            )
            raise HTTPException(status_code=401, detail='Refresh token reused')
        raise HTTPException(
            status_code=401,
            detail='Refresh token not recognised',
        )

    return cast(RefreshTokenPayload, payload)


async def login_user(
    payload: UserLogin,
    db: AsyncSession,
    redis_pool: Redis,
    hcaptcha_bypass_key: str | None = None,
    client_ip: str | None = None,
    hash_refresh_token: bool = False,
    deployment: DeploymentBinding | None = None,
    request: Request | None = None,
) -> TokenPairData:
    """Authenticate user, issue JWT tokens, and store session in Redis cache.

    Args:
        payload (UserLogin): Login credentials (username/e-mail and password).
        db (AsyncSession): Database session.
        redis_pool (Redis): Redis connection pool for caching sessions.

    Returns:
        TokenPairData: Generated tokens and user-related details.
    """
    require_local_login()
    if deployment is None and request is not None:
        deployment = await resolve_request_deployment(request, db)

    await _verify_hcaptcha(payload.hcaptcha_token, hcaptcha_bypass_key)

    await check_login_guard(
        redis_pool,
        payload.identifier,
        client_ip,
        policy=settings,
    )

    try:
        user = await _authenticate(
            db,
            payload.identifier,
            payload.password,
        )
    except HTTPException as exc:
        if exc.status_code == 401:
            await record_failed_login(
                redis_pool,
                payload.identifier,
                client_ip,
                policy=settings,
            )
        raise

    await clear_login_guard(redis_pool, payload.identifier, client_ip)

    return await issue_token_pair_for_user(
        user,
        db,
        redis_pool,
        hash_refresh_token=hash_refresh_token,
        deployment=deployment,
    )


async def issue_token_pair_for_user(
    user: User,
    db: AsyncSession,
    redis_pool: Redis,
    hash_refresh_token: bool = False,
    deployment: DeploymentBinding | None = None,
) -> TokenPairData:
    """Issue local JWT tokens for an already verified user.

    Args:
        user: Authenticated user receiving the token pair.
        db: Database session used to resolve permissions.
        redis_pool: Redis connection used to register token state.
        hash_refresh_token: Whether to cache the refresh token as a hash.

    Returns:
        Token-pair data for the authentication response.
    """
    if deployment is not None and user.tenant_id != deployment.tenant_id:
        raise HTTPException(
            status_code=403,
            detail={
                'code': 'tenant_access_denied',
                'message': 'This account does not belong to this deployment.',
            },
        )

    await prune_user_cache(redis_pool, user.username)
    cache = cast(
        UserCache | None,
        await rate_limiter_service.get_user_data(redis_pool, user.username),
    )
    if cache is None:
        cache = UserCache(
            db_user=DbUserInfo(
                id=user.id,
                username=user.username,
                role=user.role,
                group_id=user.group_id,
                status=user.status,
                tenant_id=str(user.tenant_id) if deployment else '',
            ),
            jti_list=[],
            jti_meta={},
            refresh_tokens=[],
            refresh_token_hashes=[],
            refresh_token_families={},
            feature_names=[],
        )

    # Load feature names for user's group
    feature_names = await _load_feature_names(db, user.group_id)
    cache['feature_names'] = feature_names

    # Generate JWT tokens
    new_jti = str(uuid4())
    access_token = issue_access_token(
        jwt_access,
        username=user.username,
        user_id=user.id,
        role=user.role,
        jti=new_jti,
        feature_names=feature_names,
        expires_delta=ACCESS_TTL,
        deployment=deployment,
    )
    refresh_family_id = str(uuid4())
    refresh_token = issue_refresh_token(
        jwt_refresh,
        username=user.username,
        family_id=refresh_family_id,
        token_id=str(uuid4()),
        expires_delta=REFRESH_TTL,
        deployment=deployment,
    )

    # Update cache and store in Redis
    cache['jti_list'].append(new_jti)

    # store access token expiry timestamp for pruning (epoch seconds)
    at_payload = jwt_access.decode_token(
        access_token,
        verify_exp=False,
        expected_issuer=deployment.issuer if deployment else None,
        expected_audience=deployment.audience if deployment else None,
    )
    cache['jti_meta'][new_jti] = int(at_payload['exp'])
    _store_refresh_token_in_cache(
        cache,
        refresh_token,
        hash_refresh_token=hash_refresh_token,
    )
    await rate_limiter_service.set_user_data(
        redis_pool,
        user.username,
        cast(dict[str, object], cache),
    )
    await register_refresh_token_state(
        redis_pool,
        refresh_token,
        user.username,
        refresh_family_id,
        refresh_ttl=REFRESH_TTL,
    )

    return cast(
        TokenPairData,
        {
            'access_token': access_token,
            'refresh_token': refresh_token,
            'username': user.username,
            'role': user.role,
            'user_id': user.id,
            'group_id': user.group_id,
            'feature_names': feature_names,
            **({'deployment': deployment.as_response()} if deployment else {}),
        },
    )


async def logout_user(
    refresh_token: str | None,
    authorization: str | None,
    redis_pool: Redis,
) -> None:
    """Invalidate user's tokens on logout.

    Args:
        refresh_token (str): Refresh token to invalidate.
        authorization (Optional[str]): JWT access token from request headers.
        redis_pool (Redis): Redis connection pool.
    """
    username, jti, access_payload = _access_logout_context(authorization)
    refresh_context = _refresh_logout_context(refresh_token)
    if refresh_token and refresh_context is None and access_payload is None:
        return
    if refresh_context is not None:
        refresh_username, refresh_family_id = refresh_context
        if not username and refresh_username:
            username = refresh_username
    else:
        refresh_family_id = None

    if username is None:
        return

    # Revocation must happen before cache maintenance so a logout takes
    # effect immediately, even if the user cache was already evicted.
    if access_payload is not None:
        await revoke_access_token(redis_pool, access_payload)
    if refresh_family_id:
        await revoke_refresh_family(
            redis_pool,
            refresh_family_id,
            refresh_ttl=REFRESH_TTL,
        )

    # Remove the tokens from Redis cache
    await prune_user_cache(redis_pool, username)
    cache = cast(
        UserCache | None,
        await rate_limiter_service.get_user_data(redis_pool, username),
    )
    if not cache:
        return

    # Refresh-only logout requests do not identify a single access token;
    # revoke the user's current access capabilities rather than leaving them
    # valid until their natural expiry.
    if access_payload is None:
        await revoke_user_access_tokens(
            redis_pool,
            username,
            get_user_data_fn=rate_limiter_service.get_user_data,
            revoke_access_token_jtis_fn=revoke_access_token_jtis,
        )

    _remove_logout_tokens_from_cache(cache, jti, refresh_token)
    await rate_limiter_service.set_user_data(
        redis_pool,
        username,
        cast(dict[str, object], cache),
    )


def _access_logout_context(
    authorization: str | None,
) -> tuple[str | None, str | None, AccessTokenRevocationPayload | None]:
    """Decode an access token into logout identity and JTI context.

    Args:
        authorization: Optional HTTP authorisation header.

    Returns:
        Username, access-token identifier, and decoded payload when valid.
    """
    if not authorization:
        return None, None, None
    parts = authorization.split()
    if len(parts) != 2:
        return None, None, None
    try:
        payload = jwt_access.decode_token_for_lifecycle(
            parts[1],
            verify_exp=False,
        )
        subject = access_token_subject_from_payload(payload)
    except jwt.PyJWTError:
        return None, None, None
    return (
        subject['username'],
        subject['jti'],
        {
            'jti': subject['jti'],
            'exp': cast(int, payload['exp']),
        },
    )


def _refresh_logout_context(
    refresh_token: str | None,
) -> tuple[str | None, str | None] | None:
    """Decode refresh-token identity and family data for logout.

    Args:
        refresh_token: Optional raw refresh token.

    Returns:
        Username and rotation family, or ``None`` when invalid.
    """
    if not refresh_token:
        return None
    try:
        payload = jwt_refresh.decode_token_for_lifecycle(
            refresh_token,
            verify_exp=False,
        )
        subject = refresh_token_subject_from_payload(payload)
    except jwt.PyJWTError:
        return None
    return subject['username'], subject['family_id']


def _remove_logout_tokens_from_cache(
    cache: UserCache,
    jti: str | None,
    refresh_token: str | None,
) -> None:
    """Remove logout token references while preserving cache consistency.

    Args:
        cache: Mutable cached user token state.
        jti: Optional access-token identifier to remove.
        refresh_token: Optional refresh token to remove.
    """
    cache['jti_list'] = [token for token in cache['jti_list'] if token != jti]
    if jti:
        cache['jti_meta'].pop(jti, None)
    if refresh_token:
        _remove_refresh_token_from_cache(cache, refresh_token)


async def refresh_tokens(
    payload: RefreshRequest,
    redis_pool: Redis,
    hash_refresh_token: bool = False,
    deployment: DeploymentBinding | None = None,
    request: Request | None = None,
    db: AsyncSession | None = None,
) -> TokenPairData:
    """Issue new JWT tokens using a refresh token.

    Args:
        payload (RefreshRequest): Contains the refresh token.
        redis_pool (Redis): Redis connection pool.

    Returns:
        TokenPairData: New JWT access and refresh tokens.

    Raises:
        HTTPException: If refresh token is invalid or missing.
    """
    require_local_login()
    old_refresh = payload.refresh_token or ''
    if not old_refresh:
        raise HTTPException(status_code=401, detail='Missing refresh token')

    if deployment is None and request is not None and db is not None:
        deployment = await resolve_request_deployment(request, db)

    # Verify provided refresh token against the current API deployment.
    data = await verify_refresh_token(old_refresh, redis_pool, deployment)
    username = data['subject']['username']
    family_id = str(data['subject'].get('family_id') or '')

    await prune_user_cache(redis_pool, username)
    cache = cast(
        UserCache | None,
        await rate_limiter_service.get_user_data(redis_pool, username),
    )
    if not cache or not _cache_contains_refresh_token(cache, old_refresh):
        if family_id:
            await revoke_refresh_family(
                redis_pool,
                family_id,
                refresh_ttl=REFRESH_TTL,
            )
            await revoke_user_access_tokens(
                redis_pool,
                username,
                get_user_data_fn=rate_limiter_service.get_user_data,
                revoke_access_token_jtis_fn=revoke_access_token_jtis,
            )
            raise HTTPException(status_code=401, detail='Refresh token reused')
        raise HTTPException(status_code=401, detail='Refresh token invalid')
    db_user = cast(DbUserInfo, cache['db_user'])
    if deployment is not None and db_user.get('tenant_id') != str(
        deployment.tenant_id,
    ):
        raise HTTPException(
            status_code=409,
            detail={
                'code': 'deployment_configuration_changed',
                'message': 'Deployment configuration changed; sign in again.',
            },
        )

    if family_id:
        await consume_refresh_token_state(
            redis_pool,
            old_refresh,
            family_id,
            username,
            refresh_ttl=REFRESH_TTL,
            revoke_refresh_family_fn=partial(
                revoke_refresh_family,
                refresh_ttl=REFRESH_TTL,
            ),
            revoke_user_access_tokens_fn=partial(
                revoke_user_access_tokens,
                get_user_data_fn=rate_limiter_service.get_user_data,
                revoke_access_token_jtis_fn=revoke_access_token_jtis,
            ),
        )

    _remove_refresh_token_from_cache(cache, old_refresh)

    # Generate new JWT tokens
    new_jti = str(uuid4())
    access_token = issue_access_token(
        jwt_access,
        username=username,
        user_id=db_user['id'],
        role=db_user['role'],
        jti=new_jti,
        feature_names=cache['feature_names'],
        expires_delta=ACCESS_TTL,
        deployment=deployment,
    )
    new_family_id = family_id or str(uuid4())
    new_refresh = issue_refresh_token(
        jwt_refresh,
        username=username,
        family_id=new_family_id,
        token_id=str(uuid4()),
        expires_delta=REFRESH_TTL,
        deployment=deployment,
    )

    # Update and store new tokens in Redis cache
    cache['jti_list'].append(new_jti)
    # store access token expiry for pruning
    at_payload = jwt_access.decode_token(
        access_token,
        verify_exp=False,
        expected_issuer=deployment.issuer if deployment else None,
        expected_audience=deployment.audience if deployment else None,
    )
    cache['jti_meta'][new_jti] = int(at_payload['exp'])
    _store_refresh_token_in_cache(
        cache,
        new_refresh,
        hash_refresh_token=hash_refresh_token,
    )
    await rate_limiter_service.set_user_data(
        redis_pool,
        username,
        cast(dict[str, object], cache),
    )
    await register_refresh_token_state(
        redis_pool,
        new_refresh,
        username,
        new_family_id,
        refresh_ttl=REFRESH_TTL,
        enforce_family_active=bool(family_id),
    )

    return cast(
        TokenPairData,
        {
            'access_token': access_token,
            'refresh_token': new_refresh,
            'feature_names': cache['feature_names'],
            **({'deployment': deployment.as_response()} if deployment else {}),
        },
    )
