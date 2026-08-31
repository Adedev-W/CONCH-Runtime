"""Send one local pathology image to the CONCH ranking endpoint."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


DEFAULT_API_URL = "http://127.0.0.1:8000"
DEFAULT_TIMEOUT = 60.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank pathology text queries against a local image."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to the image file to submit.",
    )
    parser.add_argument(
        "--query",
        action="append",
        required=True,
        help="Candidate pathology description; repeat this option for more queries.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Return only the highest-scoring queries.",
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_API_URL,
        help=f"Base URL of the CONCH API (default: {DEFAULT_API_URL}).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {DEFAULT_TIMEOUT}).",
    )
    return parser.parse_args()


def _image_data_uri(image_path: Path) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    mime_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def rank_image(
    image_path: Path,
    queries: list[str],
    api_url: str = DEFAULT_API_URL,
    top_k: int | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> dict:
    """Submit one image and return the decoded ranking response."""
    payload = {
        "image_base64": _image_data_uri(image_path),
        "queries": queries,
        "top_k": top_k,
    }
    request = Request(
        url=f"{api_url.rstrip('/')}/rank",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    args = _parse_args()
    try:
        response = rank_image(
            image_path=args.image,
            queries=args.query,
            api_url=args.url,
            top_k=args.top_k,
            timeout=args.timeout,
        )
    except FileNotFoundError as exc:
        print(f"Client error: {exc}")
        return 2
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"API error ({exc.code}): {detail}")
        return 1
    except URLError as exc:
        print(f"Connection error: {exc.reason}")
        return 1
    except (TimeoutError, ValueError) as exc:
        print(f"Client error: {exc}")
        return 2

    print(json.dumps(response, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
