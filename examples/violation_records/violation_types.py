from __future__ import annotations

from dataclasses import dataclass

from pydantic import TypeAdapter

from examples.violation_records.schemas import ViolationWarning


@dataclass(frozen=True)
class ViolationTypeDefinition:
    """Define canonical metadata for one violation type.

    Attributes:
        code: Stable API and database identifier.
        label: Localised display label for clients.
        warning_keys: Detector warning keys that activate the type.
    """

    code: str
    label: str
    warning_keys: tuple[str, ...]


VIOLATION_TYPE_DEFINITIONS: tuple[ViolationTypeDefinition, ...] = (
    ViolationTypeDefinition(
        code='no_safety_helmet',
        label='未戴安全帽',
        warning_keys=('warning_no_hardhat',),
    ),
    ViolationTypeDefinition(
        code='no_safety_vest',
        label='未穿安全背心',
        warning_keys=('warning_no_safety_vest',),
    ),
    ViolationTypeDefinition(
        code='near_vehicle',
        label='人員靠近車輛',
        warning_keys=('warning_close_to_vehicle',),
    ),
    ViolationTypeDefinition(
        code='near_machinery',
        label='人員靠近機具',
        warning_keys=('warning_close_to_machinery',),
    ),
    ViolationTypeDefinition(
        code='restricted_area',
        label='進入管制區',
        warning_keys=('warning_people_in_controlled_area',),
    ),
    ViolationTypeDefinition(
        code='utility_pole_restricted_area',
        label='進入電桿管制區',
        warning_keys=('warning_people_in_utility_pole_controlled_area',),
    ),
    ViolationTypeDefinition(
        code='machinery_close_to_pole',
        label='機具靠近電桿',
        warning_keys=('detect_machinery_close_to_pole',),
    ),
)

# Index canonical definitions once for request-time filter validation.
VIOLATION_TYPE_BY_CODE: dict[str, ViolationTypeDefinition] = {
    definition.code: definition for definition in VIOLATION_TYPE_DEFINITIONS
}
WARNING_PAYLOAD_ADAPTER: TypeAdapter[dict[str, ViolationWarning]] = (
    TypeAdapter(dict[str, ViolationWarning])
)


def normalise_violation_type(code: str) -> str | None:
    """Return a supported canonical violation-type code.

    Args:
        code: Client-supplied type code.

    Returns:
        Canonical code, or ``None`` when unsupported or blank.
    """
    normalized = code.strip()
    if not normalized:
        return None
    return normalized if normalized in VIOLATION_TYPE_BY_CODE else None


def violation_type_codes(
    warnings: dict[str, ViolationWarning],
) -> list[str]:
    """Derive canonical codes from parsed detector warnings.

    Args:
        warnings: Parsed detector warning mapping.

    Returns:
        Active canonical violation-type codes.
    """
    return [
        definition.code
        for definition in VIOLATION_TYPE_DEFINITIONS
        if any(
            warning is not None and warning.count > 0
            for key in definition.warning_keys
            for warning in [warnings.get(key)]
        )
    ]
