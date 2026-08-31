# The CUDA-enabled PyTorch wheel also runs on CPU-only hosts. Set
# PYTORCH_INDEX_URL to a compatible PyTorch index when the deployment driver
# targets a different CUDA release.
FROM python:3.13-slim

ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu130

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TOKENIZERS_PARALLELISM=false \
    CONCH_MODEL_CACHE=/runpod-volume/huggingface-cache/hub \
    ALLOW_MODEL_DOWNLOAD=false \
    CONCH_DEVICE=auto \
    CONCH_AUTOCAST=true

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.lock ./
RUN python -m pip install --no-cache-dir \
        --index-url "${PYTORCH_INDEX_URL}" \
        --extra-index-url https://pypi.org/simple \
        torch==2.13.0 torchvision==0.28.0 \
    && python -m pip install --no-cache-dir -r requirements-runtime.lock

COPY pyproject.toml MANIFEST.in README.md ./
COPY conch ./conch
COPY app.py ./app.py
COPY handler.py ./handler.py
RUN python -m pip install --no-cache-dir --no-deps --no-build-isolation .

EXPOSE 80

CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-80}"]
