"""Queue-free compatibility shim for the former job-shaped interface."""

from __future__ import annotations

import os
from typing import Any

from conch.inference import get_runtime
from conch.inference.service import prepare_request


def handler(job: dict[str, Any]) -> dict[str, Any]:
    """Process the legacy ``{"input": ...}`` shape without starting a queue."""
    payload = job.get("input")
    if not isinstance(payload, dict):
        raise ValueError("job.input must be an object")
    image, queries, top_k = prepare_request(payload)
    return {
        "model": os.getenv("CONCH_MODEL_NAME", "conch_ViT-B-16"),
        "results": get_runtime().rank(image, queries, top_k=top_k),
    }
