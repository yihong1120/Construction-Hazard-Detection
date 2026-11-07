from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class DeliveryOutcome(StrEnum):
    completed = 'completed'
    skipped = 'skipped'
    retryable = 'retryable'
    permanent_failure = 'permanent_failure'


@dataclass(frozen=True)
class DeliveryResult:
    outcome: DeliveryOutcome
    reason: str = ''

    def __bool__(self) -> bool:
        """A completed or deliberately skipped delivery is successful."""
        return self.outcome in (
            DeliveryOutcome.completed,
            DeliveryOutcome.skipped,
        )

    @classmethod
    def from_response(cls, body: dict[str, object]) -> DeliveryResult:
        value = body.get('outcome')
        if value is None:
            value = (
                DeliveryOutcome.completed
                if body.get(
                    'success',
                )
                else DeliveryOutcome.retryable
            )
        return cls(DeliveryOutcome(str(value)), str(body.get('message') or ''))


def retryable_status(status: int) -> bool:
    return status in (401, 408, 425, 429) or status >= 500
