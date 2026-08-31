"""Process-local CONCH model loading and inference runtime."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from conch.open_clip_custom import create_model_from_pretrained, get_tokenizer

from .ranking import rank_queries


DEFAULT_MODEL_ID = "MahmoodLab/conch"
DEFAULT_MODEL_NAME = "conch_ViT-B-16"
DEFAULT_MODEL_CACHE = "/runpod-volume/huggingface-cache/hub"


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _select_device() -> torch.device:
    requested = os.getenv("CONCH_DEVICE", "auto").strip().lower()
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CONCH_DEVICE requests CUDA, but no CUDA device is available")
    return torch.device(requested)


@dataclass
class ConchRuntime:
    """Loaded model and preprocessing objects reused by every request."""

    model: Any
    preprocess: Any
    tokenizer: Any
    device: torch.device
    autocast_dtype: torch.dtype | None = None

    def rank(
        self,
        image: Image.Image,
        queries: list[str],
        top_k: int | None = None,
    ) -> list[dict[str, float | int | str]]:
        image_tensor = self.preprocess(image).unsqueeze(0)
        return rank_queries(
            model=self.model,
            image=image_tensor,
            queries=queries,
            tokenizer=self.tokenizer,
            device=self.device,
            top_k=top_k,
            autocast_dtype=self.autocast_dtype,
        )


def load_runtime() -> ConchRuntime:
    """Load CONCH from an explicit checkpoint or a configured HF cache."""
    device = _select_device()
    model_id = os.getenv("CONCH_MODEL_ID", DEFAULT_MODEL_ID)
    model_name = os.getenv("CONCH_MODEL_NAME", DEFAULT_MODEL_NAME)
    allow_download = _env_flag("ALLOW_MODEL_DOWNLOAD", default=False)
    configured_checkpoint = os.getenv("CONCH_CHECKPOINT_PATH")

    if configured_checkpoint:
        checkpoint_path = Path(configured_checkpoint).expanduser()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"CONCH checkpoint does not exist: {checkpoint_path}"
            )
        checkpoint_ref = str(checkpoint_path)
    elif allow_download:
        checkpoint_ref = f"hf_hub:{model_id}"
    else:
        raise FileNotFoundError(
            "CONCH checkpoint is not configured. Set "
            "CONCH_CHECKPOINT_PATH to an existing file or explicitly enable "
            "ALLOW_MODEL_DOWNLOAD=true."
        )

    cache_dir = (
        os.getenv("CONCH_MODEL_CACHE")
        or os.getenv("HF_HOME")
        or DEFAULT_MODEL_CACHE
    )

    model, preprocess = create_model_from_pretrained(
        model_name,
        checkpoint_path=checkpoint_ref,
        device=device,
        cache_dir=cache_dir,
        hf_auth_token=os.getenv("HF_TOKEN"),
    )
    model.eval()

    use_autocast = _env_flag("CONCH_AUTOCAST", default=device.type == "cuda")
    autocast_dtype = torch.float16 if use_autocast and device.type == "cuda" else None
    return ConchRuntime(
        model=model,
        preprocess=preprocess,
        tokenizer=get_tokenizer(),
        device=device,
        autocast_dtype=autocast_dtype,
    )


_runtime: ConchRuntime | None = None
_runtime_lock = threading.Lock()


def get_runtime() -> ConchRuntime:
    """Return the process-local runtime, loading it at most once."""
    global _runtime
    if _runtime is None:
        with _runtime_lock:
            if _runtime is None:
                _runtime = load_runtime()
    return _runtime


def reset_runtime() -> None:
    """Reset the singleton for tests and local development."""
    global _runtime
    with _runtime_lock:
        _runtime = None
