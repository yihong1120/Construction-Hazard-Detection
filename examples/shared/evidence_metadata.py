from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from examples.shared.overlay_geometry import decode_polygons
from examples.violation_records.schemas import ViolationDetectionRows
from examples.violation_records.schemas import ViolationWarning
from examples.violation_records.violation_types import WARNING_PAYLOAD_ADAPTER
from src.evidence import EvidenceMetadata


@dataclass(frozen=True)
class ValidatedMetadata[T: EvidenceMetadata]:
    source: T
    warnings: dict[str, ViolationWarning] | None
    detections: list[list[float]] | None
    cone_polygon: list | None
    pole_polygon: list | None

    def with_source(self, **changes):
        # Only scope/snapshot enrichment is allowed after payload validation.
        if not set(changes) <= {
            'site_id',
            'model_id',
            'model_version',
            'model_classes',
        }:
            raise ValueError('Cannot change validated payload fields')
        return replace(self, source=replace(self.source, **changes))


def validate_metadata[T: EvidenceMetadata](
    evidence: T | ValidatedMetadata[T],
) -> ValidatedMetadata[T]:
    if isinstance(evidence, ValidatedMetadata):
        return evidence
    warnings = (
        WARNING_PAYLOAD_ADAPTER.validate_json(evidence.warnings_json)
        if evidence.warnings_json is not None
        else None
    )
    detections = (
        ViolationDetectionRows.model_validate_json(
            evidence.detections_json,
        ).root
        if evidence.detections_json is not None
        else None
    )
    cones = decode_polygons(evidence.cone_polygon_json)
    poles = decode_polygons(evidence.pole_polygon_json)
    canonical = replace(
        evidence,
        warnings_json=(
            WARNING_PAYLOAD_ADAPTER.dump_json(warnings).decode()
            if warnings is not None
            else None
        ),
        detections_json=(
            ViolationDetectionRows.model_construct(
                root=detections,
            ).model_dump_json()
            if detections is not None
            else None
        ),
    )
    return ValidatedMetadata(canonical, warnings, detections, cones, poles)
