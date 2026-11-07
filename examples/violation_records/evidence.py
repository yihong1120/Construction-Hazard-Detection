from __future__ import annotations

from dataclasses import dataclass

from starlette.datastructures import UploadFile

from src.evidence import EvidenceMetadata


@dataclass(frozen=True, kw_only=True)
class ViolationEvidence(EvidenceMetadata):
    image_file: UploadFile
    model_classes: list[dict] | None = None


class EvidenceEventConflictError(ValueError):
    pass
