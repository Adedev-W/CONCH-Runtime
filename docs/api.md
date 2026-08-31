# API Reference

This document describes the public HTTP contract implemented by `app.py`.

Interactive FastAPI documentation is intentionally disabled. The service does not expose `/docs`, `/redoc`, or `/openapi.json`; this document is the API reference for clients.

See the [architecture overview](Architecture_overview.png) and [architecture guide](architecture.md) for the request lifecycle and deployment architecture.

## Base URL

For local development, use `http://127.0.0.1:8000`. In Docker, the default port is `80`. RunPod Load Balancing uses the endpoint URL assigned to the deployed worker.

## `GET /ping`

The readiness endpoint is intended for load balancers and health checks. It has no response body.

| Status | Meaning | Client action |
| --- | --- | --- |
| `200` | The model is loaded and ready. | Send inference requests. |
| `204` | Model initialization is still running. | Poll again. |
| `503` | Model initialization failed. | Inspect worker logs and configuration. |

## `POST /rank`

Ranks candidate text queries against one image.

### Request body

```json
{
  "image_base64": "data:image/jpeg;base64,...",
  "queries": ["tumor tissue", "normal tissue"],
  "top_k": 1
}
```

| Field | Type | Required | Rules |
| --- | --- | --- | --- |
| `image_base64` | string | Yes | Raw valid Base64 or a Base64 data URI containing an image. |
| `queries` | array of strings | Yes | Non-empty; maximum is `CONCH_MAX_QUERIES`, default `128`. |
| `top_k` | integer or null | No | Between `1` and the number of supplied queries. Defaults to all queries. |

Images are converted to RGB before preprocessing. The default maximum decoded payload size is 8 MiB and can be changed with `CONCH_MAX_IMAGE_BYTES`.

### Response body

```json
{
  "model": "conch_ViT-B-16",
  "results": [
    {
      "rank": 1,
      "query": "tumor tissue",
      "score": 0.82,
      "probability": 0.71
    }
  ]
}
```

Results are sorted by descending similarity score. `probability` is normalized only across the submitted query list and should not be interpreted as a calibrated disease probability.

### Error responses

| Status | Cause |
| --- | --- |
| `422` | Invalid JSON fields, Base64, image, query list, or `top_k`. |
| `503` | The model is still initializing or failed to initialize. |
| `500` | An unexpected inference failure occurred. |

When the service returns `503`, respect the `Retry-After` header when present. Retry transient `5xx` failures with exponential backoff and a bounded number of attempts.

## Example client

Use the single runnable client example in `examples/rank_image.py` to submit an image from your own filesystem:

```bash
python examples/rank_image.py \
  --image ./my_pathology_image.jpg \
  --query "malignant tumor cells" \
  --query "normal tissue" \
  --top-k 1
```

## Compatibility handler

`handler.py` retains a synchronous job-shaped interface for integrations that submit:

```json
{
  "input": {
    "image_base64": "...",
    "queries": ["tumor tissue"],
    "top_k": 1
  }
}
```

New HTTP integrations should use `/rank`.
