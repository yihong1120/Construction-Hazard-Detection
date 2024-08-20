from __future__ import annotations

import base64
from functools import lru_cache

import redis


@lru_cache(maxsize=1024)
def encode_image(image: bytes) -> str:
    """
    Encode the image data to a base64 string,
    with caching to optimize performance.

    Args:
        image (bytes): The image data in bytes.

    Returns:
        str: The base64 encoded string of the image.
    """
    return base64.b64encode(image).decode('utf-8')


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
        if key.count('_') == 1
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
    image_data = []

    for key in keys:
        image = r.get(key)
        if image is not None:
            encoded_image = encode_image(image)
            image_name = key.decode('utf-8').split('_')[1]
            image_data.append((encoded_image, image_name))

    return sorted(image_data, key=lambda x: x[1])
