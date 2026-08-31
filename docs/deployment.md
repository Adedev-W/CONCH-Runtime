# Deployment Guide

This guide covers local processes, checkpoint loading, container deployment, and RunPod Load Balancing. The service listens on `PORT` and exposes `/ping` for readiness and `/rank` for inference.

## Local process

Install the API extra and point the runtime at a local checkpoint:

```bash
python3 -m pip install -e ".[dev,api]"

CONCH_CHECKPOINT_PATH=/models/pytorch_model.bin \
CONCH_DEVICE=auto \
PORT=8000 \
python app.py
```

Wait for `GET /ping` to return `200` before sending requests to `/rank`.

## Checkpoint loading

Production deployments should mount an existing checkpoint and leave model downloading disabled:

```bash
CONCH_CHECKPOINT_PATH=/models/pytorch_model.bin \
ALLOW_MODEL_DOWNLOAD=false \
python app.py
```

For development, automatic Hugging Face loading can be enabled explicitly:

```bash
CONCH_MODEL_CACHE=/models/huggingface \
ALLOW_MODEL_DOWNLOAD=true \
HF_TOKEN=... \
python app.py
```

The checkpoint is gated. Each user must satisfy the upstream Hugging Face access requirements. Never place `HF_TOKEN` in source code, a container image, or a client-side application.

## Docker

The image supports CPU fallback and NVIDIA GPU execution. Build it using a PyTorch index compatible with the target host driver:

```bash
docker build --platform linux/amd64 -t conch-runtime:dev .
```

Run on CPU:

```bash
docker run --rm \
  -e CONCH_DEVICE=cpu \
  -e CONCH_CHECKPOINT_PATH=/models/pytorch_model.bin \
  -v /path/to/checkpoint:/models:ro \
  -p 8000:80 \
  conch-runtime:dev
```

Run with an NVIDIA GPU:

```bash
docker run --gpus all --rm \
  -e CONCH_DEVICE=cuda \
  -e CONCH_CHECKPOINT_PATH=/models/pytorch_model.bin \
  -v /path/to/checkpoint:/models:ro \
  -p 8000:80 \
  conch-runtime:dev
```

For a CPU-only image, set the build argument to the CPU PyTorch index:

```bash
docker build \
  --build-arg PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
  --platform linux/amd64 \
  -t conch-runtime:cpu .
```

Keep the checkpoint mounted read-only. Do not copy weights or credentials into the image. Pin the final production image by digest.

## RunPod Load Balancing

Configure the worker with `PORT=80` and `PORT_HEALTH=80`. Use `/ping` as the readiness check and send application requests to `/rank`.

Example request:

```bash
curl -X POST "https://ENDPOINT_ID.api.runpod.ai/rank" \
  -H "Authorization: Bearer $RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  --data @request.json
```

Keep `RUNPOD_API_KEY` in the backend that calls RunPod. The worker is queue-free, so callers should implement bounded retries for `503`, `429`, and transient `5xx` responses.

## Configuration

| Variable | Default | Description |
| --- | --- | --- |
| `CONCH_CHECKPOINT_PATH` | unset | Exact local path to `pytorch_model.bin`. |
| `CONCH_MODEL_ID` | `MahmoodLab/conch` | Hugging Face repository used for opt-in downloads. |
| `CONCH_MODEL_NAME` | `conch_ViT-B-16` | Bundled model configuration name. |
| `CONCH_MODEL_CACHE` | `/runpod-volume/huggingface-cache/hub` | Hugging Face cache directory. |
| `HF_HOME` | unset | Fallback cache directory when `CONCH_MODEL_CACHE` is unset. |
| `HF_TOKEN` | unset | Token used for an explicitly enabled model download. |
| `ALLOW_MODEL_DOWNLOAD` | `false` | Enables checkpoint download fallback when set to `true`. |
| `CONCH_DEVICE` | `auto` | `auto`, `cpu`, or a CUDA device such as `cuda`. |
| `CONCH_AUTOCAST` | CUDA-dependent | Enables CUDA autocasting when set to `true`. |
| `CONCH_MAX_CONCURRENCY` | `1` | Maximum concurrent inference requests per process. |
| `CONCH_MAX_IMAGE_BYTES` | `8388608` | Maximum decoded image payload size. |
| `CONCH_MAX_QUERIES` | `128` | Maximum candidate queries in one request. |
| `PORT` | `80` in Docker | HTTP port used by Uvicorn. |

## Operational settings

- Set `CONCH_DEVICE=cuda` when GPU availability is required; use `auto` for fallback behavior.
- Keep `CONCH_MAX_CONCURRENCY` conservative because each request performs model inference.
- Increase `CONCH_MAX_IMAGE_BYTES` only when the deployment's request limits support it.
- Use a persistent `CONCH_MODEL_CACHE` for development downloads, not for secrets.
- Monitor `/ping`, worker logs, request latency, and inference error rates.
