"""Image-to-text query ranking for the CONCH model."""

from contextlib import nullcontext
from typing import Sequence

import torch

from conch.open_clip_custom import tokenize


def rank_queries(
    model,
    image: torch.Tensor,
    queries: Sequence[str],
    tokenizer,
    device: torch.device | str,
    top_k: int | None = None,
    autocast_dtype: torch.dtype | None = None,
) -> list[dict[str, float | int | str]]:
    """Rank text queries by cosine similarity to one preprocessed image."""
    query_list = list(queries)
    if not query_list:
        return []
    if top_k is not None and (top_k < 1 or top_k > len(query_list)):
        raise ValueError("top_k must be between 1 and the number of queries")

    device = torch.device(device)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError("image must have shape (1, channels, height, width)")

    text_tokens = tokenize(tokenizer, query_list).to(device)
    image = image.to(device)

    if autocast_dtype is None:
        autocast_context = nullcontext()
    else:
        autocast_context = torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
        )

    with torch.inference_mode(), autocast_context:
        image_features = model.encode_image(
            image,
            normalize=True,
            proj_contrast=True,
        )
        text_features = model.encode_text(
            text_tokens,
            normalize=True,
        )
        scores = (image_features @ text_features.T)[0].float()
        probabilities = scores.softmax(dim=0)
        ranked_indices = torch.argsort(scores, descending=True)

    if top_k is not None:
        ranked_indices = ranked_indices[:top_k]

    return [
        {
            "rank": rank,
            "query": query_list[index],
            "score": scores[index].item(),
            "probability": probabilities[index].item(),
        }
        for rank, index in enumerate(ranked_indices.tolist(), start=1)
    ]
