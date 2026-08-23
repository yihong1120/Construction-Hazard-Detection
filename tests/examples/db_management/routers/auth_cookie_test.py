from __future__ import annotations

import unittest
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

from fastapi import Response
from starlette.datastructures import FormData
from starlette.datastructures import QueryParams
from starlette.requests import Request

from examples.db_management.routers import auth
from examples.db_management.schemas.auth import AppleAuthRequest
from examples.db_management.schemas.auth import GoogleAuthRequest
from examples.db_management.schemas.auth import UserLogin
from examples.db_management.services import web_auth_services as web_auth


def make_request(
    *,
    method: str = 'POST',
    headers: dict[str, str] | None = None,
) -> Request:
    """Build the minimal HTTP request used by auth route unit tests."""
    header_items = [
        (key.lower().encode(), value.encode())
        for key, value in (headers or {}).items()
    ]
    return Request(
        {
            'type': 'http',
            'method': method,
            'headers': header_items,
            'client': ('127.0.0.1', 5000),
            'scheme': 'https',
            'server': ('testserver', 443),
        },
    )


class TestAuthCookieCoverage(unittest.IsolatedAsyncioTestCase):

    """Provide TestAuthCookieCoverage.
    """

    def test_web_request_detection_accepts_explicit_modes(self) -> None:
        """Explicit platform and cookie mode headers select the Web flow."""
        self.assertTrue(
            web_auth.is_web_auth_request(
                make_request(headers={'x-client-platform': 'browser'}),
            ),
        )
        self.assertTrue(
            web_auth.is_web_auth_request(
                make_request(headers={'x-auth-mode': 'web_cookie'}),
            ),
        )

    def test_token_pair_omits_refresh_token_for_cookie_response(self) -> None:
        """Test token pair omits refresh token for cookie response.
        """
        result = web_auth.token_pair_response(
            {'access_token': 'access', 'refresh_token': 'refresh', 'feature_names': []},
            omit_refresh_token=True,
        )

        self.assertEqual(result.access_token, 'access')
        self.assertIsNone(result.refresh_token)

    async def test_web_login_sets_refresh_cookie(self) -> None:
        """Legacy-compatible Web login stores refresh tokens in HttpOnly
        cookie."""
        response = Response()
        request = make_request(headers={'x-client-platform': 'web'})
        with (
            patch.object(web_auth, 'LEGACY_WEB_TOKEN_ENDPOINTS_ENABLED', True),
            patch.object(
                auth,
                'login_user',
                new=AsyncMock(
                    return_value={
                        'access_token': 'access',
                        'refresh_token': 'refresh',
                    },
                ),
            ),
        ):
            result = await auth.login(
                UserLogin(identifier='user', password='password'),
                request,
                response,
                db=MagicMock(),
                redis=MagicMock(),
            )

        self.assertEqual(result.access_token, 'access')
        self.assertIsNone(result.refresh_token)
        self.assertIn(
            'refresh_session=refresh',
            response.headers['set-cookie'],
        )

    async def test_web_provider_logins_set_refresh_cookie(self) -> None:
        """Google and Apple Web responses use the same cookie-only contract."""
        request = make_request(headers={'x-auth-mode': 'cookie'})
        google_response = Response()
        apple_response = Response()
        token_pair = {'access_token': 'access', 'refresh_token': 'refresh'}

        with (
            patch.object(web_auth, 'LEGACY_WEB_TOKEN_ENDPOINTS_ENABLED', True),
            patch.object(
                auth,
                'login_with_google',
                new=AsyncMock(return_value=token_pair),
            ),
            patch.object(
                auth,
                'login_with_apple',
                new=AsyncMock(return_value=token_pair),
            ),
        ):
            google_result = await auth.google_login(
                GoogleAuthRequest(id_token='google-token'),
                request,
                google_response,
                db=MagicMock(),
                redis=MagicMock(),
            )
            apple_result = await auth.apple_login(
                AppleAuthRequest(
                    identity_token='apple-token',
                    authorization_code='apple-code',
                ),
                request,
                apple_response,
                db=MagicMock(),
                redis=MagicMock(),
            )

        self.assertIsNone(google_result.refresh_token)
        self.assertIsNone(apple_result.refresh_token)
        self.assertIn(
            'refresh_session=refresh',
            google_response.headers['set-cookie'],
        )
        self.assertIn(
            'refresh_session=refresh',
            apple_response.headers['set-cookie'],
        )

    async def test_refresh_rotates_cookie_token(self) -> None:
        """Cookie refresh ignores the body and replaces the browser cookie."""
        request = make_request(
            headers={
                'x-auth-mode': 'cookie',
                'cookie': 'refresh_session=old-refresh',
            },
        )
        response = Response()
        refresh_tokens = AsyncMock(
            return_value={
                'access_token': 'new-access',
                'refresh_token': 'new-refresh',
            },
        )

        with (
            patch.object(web_auth, 'LEGACY_WEB_TOKEN_ENDPOINTS_ENABLED', True),
            patch.object(auth, 'refresh_tokens', new=refresh_tokens),
        ):
            result = await auth.refresh(
                request,
                response,
                payload=None,
                redis=MagicMock(),
            )

        self.assertEqual(result.access_token, 'new-access')
        self.assertIsNone(result.refresh_token)
        assert refresh_tokens.await_args is not None
        self.assertEqual(
            refresh_tokens.await_args.args[0].refresh_token,
            'old-refresh',
        )
        self.assertIn(
            'refresh_session=new-refresh',
            response.headers['set-cookie'],
        )

    async def test_apple_callback_accepts_posted_form_data(self) -> None:
        """Apple POST callbacks retain form parameters in the Android
        intent."""

        class FormRequest:
            """Provide FormRequest.
            """
            method = 'POST'
            query_params = QueryParams()

            async def form(self) -> FormData:
                """Perform form.

                Returns:
                    The callable result.
                """
                return FormData(
                    [('code', 'apple-code'), ('state', 'signed-state')],
                )

        response = await auth.apple_callback(FormRequest())

        self.assertEqual(response.status_code, 302)
        self.assertIn('code=apple-code', response.headers['location'])
        self.assertIn('state=signed-state', response.headers['location'])
