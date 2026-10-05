"""Validate human annotations against the original record snapshot."""
from __future__ import annotations

from fastapi import HTTPException

from examples.violation_records.record_policy import historical_classes


def validate_feedback_label(payload, violation) -> None:
    codes = {c.code for c in historical_classes(violation)}
    if (
        payload.corrected_label is not None
        and payload.corrected_label not in codes
    ):
        raise HTTPException(422, detail='Unknown feedback category')
    if (
        payload.type in {'false_negative', 'wrong_class'}
        and payload.corrected_label not in codes
    ):
        raise HTTPException(422, detail='A feedback category is required')
