import torch

from conch.open_clip_custom.custom_tokenizer import get_tokenizer, tokenize


def test_tokenize_preserves_conch_context_length():
    tokenizer = get_tokenizer()
    tokens = tokenize(tokenizer, ["human pathology", "vision language model"])

    assert tokens.shape == (2, 128)
    assert tokens.dtype == torch.long
    assert torch.all(tokens[:, -1] == tokenizer.pad_token_id)
