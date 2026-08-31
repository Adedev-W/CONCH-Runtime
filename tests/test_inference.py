import base64
import io

import pytest
import torch
from PIL import Image

import handler as serverless_handler
from conch.inference import ranking, runtime


class FakeModel:
    def encode_image(self, image, normalize=True, proj_contrast=True):
        return torch.tensor([[1.0, 0.0]], device=image.device)

    def encode_text(self, tokens, normalize=True):
        return torch.tensor(
            [[1.0, 0.0], [0.0, 1.0]],
            device=tokens.device,
        )


def test_rank_queries_sorts_scores_and_applies_top_k(monkeypatch):
    monkeypatch.setattr(
        ranking,
        "tokenize",
        lambda tokenizer, queries: torch.ones((len(queries), 4), dtype=torch.long),
    )

    results = ranking.rank_queries(
        model=FakeModel(),
        image=torch.zeros((1, 3, 8, 8)),
        queries=["second", "first"],
        tokenizer=object(),
        device="cpu",
        top_k=1,
    )

    assert results[0]["query"] == "second"
    assert results[0]["rank"] == 1
    assert results[0]["score"] == pytest.approx(1.0)


def test_handler_validates_and_decodes_base64(monkeypatch):
    image_buffer = io.BytesIO()
    Image.new("RGB", (4, 4), color="white").save(image_buffer, format="PNG")
    image_base64 = base64.b64encode(image_buffer.getvalue()).decode()

    class FakeRuntime:
        def rank(self, image, queries, top_k=None):
            assert image.mode == "RGB"
            assert queries == ["tumor"]
            assert top_k == 1
            return [{"rank": 1, "query": "tumor", "score": 1.0, "probability": 1.0}]

    monkeypatch.setattr(serverless_handler, "get_runtime", lambda: FakeRuntime())
    response = serverless_handler.handler(
        {
            "input": {
                "image_base64": f"data:image/png;base64,{image_base64}",
                "queries": ["tumor"],
                "top_k": 1,
            }
        }
    )

    assert response["results"][0]["query"] == "tumor"


def test_handler_rejects_invalid_queries():
    with pytest.raises(ValueError, match="non-empty list"):
        serverless_handler.handler({"input": {"queries": [], "image_base64": "x"}})


def test_runtime_is_loaded_once(monkeypatch):
    loaded = object()
    calls = []
    monkeypatch.setattr(runtime, "load_runtime", lambda: calls.append(loaded) or loaded)
    runtime.reset_runtime()

    assert runtime.get_runtime() is loaded
    assert runtime.get_runtime() is loaded
    assert len(calls) == 1

    runtime.reset_runtime()


def test_load_runtime_uses_local_checkpoint_without_download(monkeypatch, tmp_path):
    checkpoint = tmp_path / "pytorch_model.bin"
    checkpoint.write_bytes(b"checkpoint")
    calls = []

    class FakeLoadedModel:
        def eval(self):
            return self

    def fake_create_model(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return FakeLoadedModel(), object()

    monkeypatch.setenv("CONCH_CHECKPOINT_PATH", str(checkpoint))
    monkeypatch.setattr(runtime, "create_model_from_pretrained", fake_create_model)
    monkeypatch.setattr(runtime, "get_tokenizer", lambda: object())

    loaded = runtime.load_runtime()

    assert isinstance(loaded, runtime.ConchRuntime)
    assert calls[0][1]["checkpoint_path"] == str(checkpoint)


def test_load_runtime_rejects_missing_explicit_checkpoint(monkeypatch, tmp_path):
    monkeypatch.setenv(
        "CONCH_CHECKPOINT_PATH",
        str(tmp_path / "missing" / "pytorch_model.bin"),
    )
    monkeypatch.delenv("ALLOW_MODEL_DOWNLOAD", raising=False)

    with pytest.raises(FileNotFoundError, match="does not exist"):
        runtime.load_runtime()


def test_load_runtime_uses_configured_cache_for_download(monkeypatch, tmp_path):
    calls = []

    class FakeLoadedModel:
        def eval(self):
            return self

    def fake_create_model(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return FakeLoadedModel(), object()

    monkeypatch.delenv("CONCH_CHECKPOINT_PATH", raising=False)
    monkeypatch.setenv("ALLOW_MODEL_DOWNLOAD", "true")
    monkeypatch.setenv("CONCH_MODEL_CACHE", str(tmp_path / "huggingface"))
    monkeypatch.setattr(runtime, "create_model_from_pretrained", fake_create_model)
    monkeypatch.setattr(runtime, "get_tokenizer", lambda: object())

    runtime.load_runtime()

    assert calls[0][1]["checkpoint_path"] == "hf_hub:MahmoodLab/conch"
    assert calls[0][1]["cache_dir"] == str(tmp_path / "huggingface")
