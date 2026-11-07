from __future__ import annotations

from typing import get_args
from typing import Self

from pydantic import AliasChoices
from pydantic import field_validator
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pydantic_settings import SettingsConfigDict


class EnvironmentSettings(BaseSettings):
    model_config = SettingsConfigDict(
        populate_by_name=True,
        extra='ignore',
        str_strip_whitespace=True,
        hide_input_in_errors=True,
    )

    @field_validator('*', mode='before')
    @classmethod
    def strip_environment_value(cls, value: object, info) -> object:
        if isinstance(value, str):
            annotation = cls.model_fields[info.field_name].annotation
            if SecretStr in (annotation, *get_args(annotation)):
                return SecretStr(value)
            return value.strip()
        return value

    @classmethod
    def from_overrides(cls, **values) -> Self:
        """Explicit constructor values override environment aliases."""
        overrides = {}
        for name, value in values.items():
            if value is None:
                continue
            alias = cls.model_fields[name].validation_alias
            if isinstance(alias, AliasChoices):
                alias = alias.choices[0]
            overrides[alias if isinstance(alias, str) else name] = value
        return cls(**overrides)
