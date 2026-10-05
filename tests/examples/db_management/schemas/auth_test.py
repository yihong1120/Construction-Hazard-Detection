from __future__ import annotations

import pytest
from pydantic import ValidationError

from examples.db_management.schemas.auth import (
    NativeSocialExchangeCompleteRequest,
)


def test_native_social_exchange_does_not_accept_local_login_credentials():
    with pytest.raises(ValidationError):
        NativeSocialExchangeCompleteRequest(
            identifier='alice', password='password',
        )
