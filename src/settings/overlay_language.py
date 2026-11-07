from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated

from pydantic import Field
from pydantic import field_validator
from pydantic_settings import NoDecode

from examples.streaming_web.overlay_labels import LANGUAGE_ALIASES
from examples.streaming_web.overlay_labels import SUPPORTED_LABEL_LANGUAGES
from src.settings.base import EnvironmentSettings


def canonical_language(value: str) -> str:
    code = value.strip().replace('_', '-')
    if code in SUPPORTED_LABEL_LANGUAGES:
        return code
    canonical = LANGUAGE_ALIASES.get(code.lower())
    if canonical is None:
        raise ValueError('Unsupported overlay language')
    return canonical


class OverlayLanguagePolicy(EnvironmentSettings):
    model_config = {'frozen': True}
    allowed: Annotated[tuple[str, ...], NoDecode] = Field(
        SUPPORTED_LABEL_LANGUAGES,
        validation_alias='MEDIA_OVERLAY_ALLOWED_LANGUAGES',
    )
    default: str = Field(
        'zh-TW',
        validation_alias='MEDIA_DEFAULT_OVERLAY_LANGUAGE',
    )

    @field_validator('allowed', mode='before')
    @classmethod
    def canonical_allowed(cls, value):
        values = value.split(',') if isinstance(value, str) else value
        allowed = tuple(
            dict.fromkeys(
                canonical_language(code) for code in values if code.strip()
            ),
        )
        if not allowed:
            raise ValueError('At least one overlay language is required')
        return allowed

    @field_validator('default', mode='before')
    @classmethod
    def canonical_default(cls, value):
        return canonical_language(value)

    def default_language(self) -> str:
        if self.default not in self.allowed:
            raise ValueError(
                'Default overlay language must be an allowed language',
            )
        return self.default


@lru_cache(maxsize=16)
def _policy(allowed: str, default: str) -> OverlayLanguagePolicy:
    return OverlayLanguagePolicy(allowed=allowed, default=default)


def get_overlay_language_policy() -> OverlayLanguagePolicy:
    # Environment changes take effect on the next owner lifecycle; callers
    # retain this immutable policy instead of parsing settings per frame.
    return _policy(
        os.getenv(
            'MEDIA_OVERLAY_ALLOWED_LANGUAGES',
            ','.join(SUPPORTED_LABEL_LANGUAGES),
        ),
        os.getenv('MEDIA_DEFAULT_OVERLAY_LANGUAGE', 'zh-TW'),
    )
