import json

from examples import rank_image


class FakeResponse:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return b'{"model":"conch_ViT-B-16","results":[]}'


def test_rank_image_builds_base64_payload_for_local_file(monkeypatch, tmp_path):
    image_path = tmp_path / "slide.png"
    image_path.write_bytes(b"image-bytes")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(rank_image, "urlopen", fake_urlopen)

    response = rank_image.rank_image(
        image_path=image_path,
        queries=["tumor tissue", "normal tissue"],
        api_url="http://localhost:8000/",
        top_k=1,
        timeout=12,
    )

    payload = json.loads(captured["request"].data.decode("utf-8"))
    assert captured["request"].full_url == "http://localhost:8000/rank"
    assert captured["timeout"] == 12
    assert payload["image_base64"].startswith("data:image/png;base64,")
    assert payload["queries"] == ["tumor tissue", "normal tissue"]
    assert payload["top_k"] == 1
    assert response["model"] == "conch_ViT-B-16"
