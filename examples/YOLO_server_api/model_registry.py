from __future__ import annotations

import asyncio
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Annotated
from typing import Literal

from fastapi import HTTPException
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator
from pydantic import StringConstraints

from examples.shared.class_colors import normalize_color
from examples.shared.class_colors import published_color

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class ModelClass(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: int = Field(ge=0, strict=True)
    code: Text
    display_name: Text
    color: str | None = None

    @field_validator('color', mode='before')
    @classmethod
    def optional_color(cls, value):
        return normalize_color(value)

    @model_validator(mode='after')
    def valid_code(self):
        if self.code.lstrip('+-').isdecimal():
            raise ValueError('Class codes cannot be integer strings')
        self.color = self.color or published_color(self.code)
        return self


class ModelEntry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: Text
    display_name: Text
    version: Text
    hf_repo: str | None = None
    hf_filename: str | None = None
    hf_revision: str | None = Field(default=None, pattern=r'^[0-9a-f]{40}$')
    artifact: Text
    sha256: str = Field(pattern=r'^[0-9a-f]{64}$')
    enabled: bool = True
    capabilities: list[Literal['image', 'stream']]
    tenant_ids: list[str] = Field(default_factory=list)
    site_ids: list[int] = Field(default_factory=list)
    roles: list[str] = Field(
        default_factory=lambda: ['admin', 'user', 'guest', 'super_admin'],
    )
    classes: list[ModelClass]

    @model_validator(mode='after')
    def unique_classes(self):
        if len({c.id for c in self.classes}) != len(self.classes):
            raise ValueError('Duplicate class id')
        if len({c.code for c in self.classes}) != len(self.classes):
            raise ValueError('Duplicate class code')
        return self

    def allowed(
        self,
        tenant_id: str,
        role: str,
        capability: str,
        site_id: int | None = None,
    ) -> bool:
        return (
            self.enabled
            and capability in self.capabilities
            and tenant_id in self.tenant_ids
            and role in self.roles
            and (capability != 'stream' or site_id in self.site_ids)
        )


class ModelRegistry(BaseModel):
    model_config = ConfigDict(extra='forbid')
    revision: Text
    default_id: str | None = None
    models: list[ModelEntry]

    @model_validator(mode='after')
    def unique_models(self):
        if len({m.id for m in self.models}) != len(self.models):
            raise ValueError('Duplicate model id')
        if self.default_id is not None and self.default_id not in {
            m.id for m in self.models
        }:
            raise ValueError('Unknown default')
        return self


def unavailable(code: str = 'OPTIONS_UNAVAILABLE') -> HTTPException:
    return HTTPException(503, detail={'code': code})


async def _read_catalog_rows_async():
    from sqlalchemy import select
    from sqlalchemy.ext.asyncio import create_async_engine
    from sqlalchemy.pool import NullPool
    from examples.auth.config import Settings
    from examples.auth.models import DetectionModelCatalog

    # A short-lived connection belongs to this worker's event loop.
    engine = create_async_engine(
        Settings().sqlalchemy_database_uri,
        poolclass=NullPool,
        connect_args={'timeout': 5, 'command_timeout': 5},
    )
    try:
        async with engine.connect() as connection:
            result = await connection.execute(
                select(
                    DetectionModelCatalog.model_key,
                    DetectionModelCatalog.definition,
                    DetectionModelCatalog.is_default,
                ).order_by(
                    DetectionModelCatalog.display_order,
                    DetectionModelCatalog.model_key,
                ),
            )
            return list(result.mappings())
    finally:
        await engine.dispose()


def _read_catalog_rows():
    return asyncio.run(_read_catalog_rows_async())


def load_registry() -> ModelRegistry:
    """Read the database catalog from a synchronous worker thread."""
    try:
        rows = _read_catalog_rows()
        models = [
            ModelEntry.model_validate(
                {**r['definition'], 'id': r['model_key']},
            )
            for r in rows
        ]
        defaults = [r['model_key'] for r in rows if r['is_default']]
        if len(defaults) > 1:
            raise ValueError('Multiple default models')
        return ModelRegistry(
            revision=revision_for([m.model_dump() for m in models] + defaults),
            default_id=defaults[0] if defaults else None,
            models=models,
        )
    except Exception as exc:
        raise unavailable('MODEL_REGISTRY_UNAVAILABLE') from exc


def artifact_path(entry: ModelEntry) -> Path:
    """Use local weights or fetch the published Hugging Face commit."""
    path = Path(entry.artifact)
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    if path.is_file():
        return path
    if entry.hf_repo and entry.hf_filename and entry.hf_revision:
        from huggingface_hub import hf_hub_download

        try:
            return Path(
                hf_hub_download(
                    repo_id=entry.hf_repo,
                    filename=entry.hf_filename,
                    revision=entry.hf_revision,
                ),
            )
        except Exception as exc:
            raise unavailable('MODEL_DOWNLOAD_UNAVAILABLE') from exc
    return path


@lru_cache(maxsize=128)
def _artifact_digest(path: Path, fingerprint: tuple[int, ...]) -> str:
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def verify_artifact(entry: ModelEntry) -> Path:
    path = artifact_path(entry)
    try:
        stat = path.stat()
        digest = _artifact_digest(
            path,
            (
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            ),
        )
    except OSError as exc:
        raise HTTPException(409, detail={'code': 'MODEL_UNAVAILABLE'}) from exc
    if digest != entry.sha256:
        raise HTTPException(409, detail={'code': 'MODEL_UNAVAILABLE'})
    return path


def require_model(
    model_id: str,
    version: str | None,
    tenant_id: str,
    role: str,
    capability: str,
    site_id: int | None = None,
) -> ModelEntry:
    registry = load_registry()
    entry = next((m for m in registry.models if m.id == model_id), None)
    if (
        entry is None
        or not entry.enabled
        or (version is not None and version != entry.version)
    ):
        raise HTTPException(409, detail={'code': 'MODEL_UNAVAILABLE'})
    if not entry.allowed(tenant_id, role, capability, site_id):
        raise HTTPException(
            403, detail='Model is not allowed for this resource',
        )
    verify_artifact(entry)
    return entry


def revision_for(value: object) -> str:
    serialized = json.dumps(
        value, sort_keys=True, ensure_ascii=False, default=str,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()
