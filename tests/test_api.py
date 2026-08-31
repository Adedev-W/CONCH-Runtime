import base64
import os
from pathlib import Path

import httpx


API_URL = os.getenv("CONCH_API_URL", "http://127.0.0.1:8000").rstrip("/")
IMAGE_PATH = Path(__file__).resolve().parents[1] / "steptodown.com151127.jpg"


def _image_data_uri() -> str:
    encoded = base64.b64encode(IMAGE_PATH.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def check_rank_endpoint_as_external_client() -> None:
    """Call the already-running API through httpx, like a real client."""
    response = httpx.post(
        f"{API_URL}/rank",
        json={
            "image_base64": _image_data_uri(),
            "queries": [
    "healthy normal tissue",
    "malignant tumor cells",
    "stromal tissue",
    "kidney tissue",
    "liver tissue",
    "Invasive Ductal Carcinoma",
    "Invasive Lobular Carcinoma"
],
            "top_k": 1,
        },
        timeout=float(os.getenv("CONCH_API_REQUEST_TIMEOUT", "30")),
    )
    if response.status_code != 200:
        raise RuntimeError(
            f"POST /rank returned HTTP {response.status_code}: {response.text}"
        )

    payload = response.json()
    if not isinstance(payload.get("model"), str):
        raise AssertionError("Response field 'model' must be a string")
    if len(payload.get("results", [])) != 1:
        raise AssertionError("Response must contain exactly one ranking result")
    result = payload["results"][0]
    print(result)
    if result.get("rank") != 1:
        raise AssertionError("The top result must have rank 1")
    if result.get("query") not in {"tumor", "normal"}:
        raise AssertionError("Unexpected query in ranking result")


if __name__ == "__main__":
    try:
        check_rank_endpoint_as_external_client()
    except Exception as exc:
        print(f"API smoke test failed: {exc}")
        raise SystemExit(1) from exc
    print("API smoke test passed")
