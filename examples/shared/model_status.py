from __future__ import annotations


def model_status_key(stream_id: int) -> str:
    return f'stream-model-status:{stream_id}'


def model_challenge_key(model_id: str) -> str:
    return f'stream-model-challenge:{model_id}'
