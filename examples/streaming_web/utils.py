from __future__ import annotations

import base64

import redis


def get_labels(r: redis.Redis) -> list[str]:
    """
    Retrieve and decode unique labels from Redis keys, excluding 'test'.

    Args:
        r (redis.Redis): The Redis connection.

    Returns:
        list: Sorted list of unique labels.
    """
    cursor, keys = r.scan()
    decoded_keys = [key.decode('utf-8') for key in keys]
    labels = {
        key.split('_')[0]
        for key in decoded_keys
        if key.count(
            '_',
        )
        == 1
        and not key.startswith('_')
        and not key.endswith('_')
        and key.split('_')[0] != 'test'
    }
    return sorted(labels)


def get_image_data(r: redis.Redis, label: str) -> list[tuple[str, str]]:
    """
    Retrieve and process image data for a specific label.

    Args:
        r (redis.Redis): The Redis connection.
        label (str): The label/category of the images.

    Returns:
        list: List of tuples containing base64 encoded images and their names.
    """
    cursor, keys = r.scan(match=f"{label}_*")
    image_data = [
        (
            base64.b64encode(image).decode('utf-8'),
            key.decode('utf-8').split('_')[1],
        )
        for key in keys
        if (image := r.get(key)) is not None  # Ensure image is not None
    ]
    return sorted(image_data, key=lambda x: x[1])
