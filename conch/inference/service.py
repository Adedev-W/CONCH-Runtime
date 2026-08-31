"""Request decoding and inference orchestration shared by HTTP adapters."""

from __future__ import annotations

import base64
import binascii
import io
import os
from typing import Any, Mapping

from PIL import Image, UnidentifiedImageError


DEFAULT_MAX_IMAGE_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_QUERIES = 128


def decode_image(value: Any) -> Image.Image:
    """Decode a raw base64 string or a data URI into an RGB image."""
    if not isinstance(value, str) or not value:
        raise ValueError("image_base64 must be a non-empty base64 string")

    if value.startswith("data:"):
        parts = value.split(",", 1)
        # Data URIs contain metadata before the comma and encoded bytes after it.
        if len(parts) != 2:
            raise ValueError("image_base64 has an invalid data URI")
        encoded = parts[1]
    else:
        encoded = value

    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc

    max_bytes = int(os.getenv("CONCH_MAX_IMAGE_BYTES", DEFAULT_MAX_IMAGE_BYTES))
    if len(image_bytes) > max_bytes:
        raise ValueError(f"image exceeds the {max_bytes} byte limit")

    try:
        with Image.open(io.BytesIO(image_bytes)) as image:
            return image.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("image_base64 does not contain a valid image") from exc


def validate_queries(queries: Any, top_k: Any = None) -> tuple[list[str], int | None]:
    """Validate ranking parameters shared by HTTP and compatibility callers."""
    if not isinstance(queries, list) or not queries:
        raise ValueError("queries must be a non-empty list")

    max_queries = int(os.getenv("CONCH_MAX_QUERIES", DEFAULT_MAX_QUERIES))
    if len(queries) > max_queries:
        raise ValueError(f"queries cannot contain more than {max_queries} items")
    if any(not isinstance(query, str) or not query.strip() for query in queries):
        raise ValueError("every query must be a non-empty string")

    if top_k is not None:
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        if top_k < 1 or top_k > len(queries):
            raise ValueError("top_k must be between 1 and the number of queries")

    return queries, top_k


def prepare_request(
    payload: Mapping[str, Any],
) -> tuple[Image.Image, list[str], int | None]:
    """Decode and validate a request before requiring a loaded model."""
    queries, top_k = validate_queries(payload.get("queries"), payload.get("top_k"))
    image = decode_image(payload.get("image_base64"))
    return image, queries, top_k


def rank_request(runtime: Any, payload: Mapping[str, Any]) -> list[dict[str, float | int | str]]:
    """Decode one request and execute ranking through an already loaded runtime."""
    image, queries, top_k = prepare_request(payload)
    return runtime.rank(image, queries, top_k=top_k)
