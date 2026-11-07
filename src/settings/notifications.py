from __future__ import annotations

from pydantic import Field
from pydantic import SecretStr

from src.settings.base import EnvironmentSettings


class MessengerSettings(EnvironmentSettings):
    page_access_token: SecretStr = Field(
        '',
        validation_alias='FACEBOOK_PAGE_ACCESS_TOKEN',
        repr=False,
        exclude=True,
    )


class TelegramSettings(EnvironmentSettings):
    telegram_bot_token: SecretStr = Field('', repr=False, exclude=True)


class WeChatSettings(EnvironmentSettings):
    model_config = {'env_prefix': 'WECHAT_'}
    corp_id: str = ''
    corp_secret: SecretStr = Field('', repr=False, exclude=True)
    agent_id: int = Field(0, ge=0)


class LineSettings(EnvironmentSettings):
    channel_access_token: SecretStr = Field(
        '',
        validation_alias='LINE_CHANNEL_ACCESS_TOKEN',
        repr=False,
        exclude=True,
    )
    cloudinary_cloud_name: str | None = None
    cloudinary_api_key: SecretStr | None = Field(
        None, repr=False, exclude=True,
    )
    cloudinary_api_secret: SecretStr | None = Field(
        None, repr=False, exclude=True,
    )
