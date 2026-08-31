# CONCH Runtime

<p align="center">
  <a href="https://github.com/Mahmoodlab/CONCH"><img src="https://img.shields.io/badge/CONCH-research%20model-7f52a2" alt="CONCH research model"></a>
  <a href="https://www.python.org/downloads/release/python-3130/"><img src="https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white" alt="Python 3.13 or newer"></a>
  <a href="https://fastapi.tiangolo.com/"><img src="https://img.shields.io/badge/FastAPI-0.141.1-009688?logo=fastapi&logoColor=white" alt="FastAPI 0.141.1"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-CC%20BY--NC--ND%204.0-blue" alt="CC BY-NC-ND 4.0 license"></a>
</p>

Production-oriented FastAPI runtime for [CONCH](https://github.com/Mahmoodlab/CONCH), a vision-language foundation model for human pathology. Send one pathology image and a list of candidate descriptions to rank the descriptions by image-text similarity.

<p align="center">
  <img src="docs/Architecture_overview.png" width="720" alt="CONCH Runtime architecture overview">
</p>

> **Research-use notice:** CONCH outputs are not medical diagnoses and must not replace qualified clinical review.

## Highlights

- Zero-shot image-to-text ranking through `POST /rank`.
- CPU fallback and CUDA inference with optional autocasting.
- Readiness-aware service with bounded inference concurrency.
- Explicit local checkpoint loading with opt-in Hugging Face downloads.
- Base64-encoded JPEG, PNG, and other RGB-convertible image inputs.

The HTTP service currently exposes ranking only. Caption generation and localization remain available only through the lower-level Python model API; see the [model guide](docs/model.md).

## Quick start

Install the package with development and API dependencies:

```bash
python3 -m pip install -e ".[dev,api]"
```

Start with an existing checkpoint:

```bash
CONCH_CHECKPOINT_PATH=/models/pytorch_model.bin \
CONCH_DEVICE=auto \
PORT=8000 \
python app.py
```

Wait for `GET /ping` to return `200` before sending inference requests. For Hugging Face downloads, deployment configuration, and RunPod setup, see the [deployment guide](docs/deployment.md).

## API example

```bash
python examples/rank_image.py \
  --image ./my_pathology_image.jpg \
  --query "malignant tumor cells" \
  --query "normal tissue" \
  --top-k 1
```

The service accepts an image and comparable candidate queries, then returns ranked similarity scores and probabilities relative to the submitted query list. See the [API reference](docs/api.md) for the request contract, validation rules, response schema, and errors.

## Documentation

- [Architecture overview](docs/architecture.md)
- [API reference](docs/api.md)
- [Deployment guide](docs/deployment.md)
- [Model and capability guide](docs/model.md)
- [Upstream CONCH repository](https://github.com/Mahmoodlab/CONCH)

## Development

Run the lightweight tests and syntax check:

```bash
python3 -m pytest
python3 -m compileall conch
```

The external API smoke test requires a running service and an image fixture:

```bash
CONCH_API_URL=http://127.0.0.1:8000 python tests/test_api.py
```

## License and citation

CONCH and its associated upstream code/model are released by Mahmood Lab under the [CC BY-NC-ND 4.0 license](https://github.com/Mahmoodlab/CONCH/blob/main/LICENSE). Review the [model guide](docs/model.md) for usage limitations and the [upstream repository](https://github.com/Mahmoodlab/CONCH) for model access requirements.

If you use CONCH in research, cite the upstream paper:

```bibtex
@article{lu2024avisionlanguage,
  title={A visual-language foundation model for computational pathology},
  author={Lu, Ming Y. and Chen, Bowen and Williamson, Drew F. K. and Chen, Richard J. and Liang, Ivy and Ding, Tong and Jaume, Guillaume and Odintsov, Igor and Le, Long Phi and Gerber, Georg and others},
  journal={Nature Medicine},
  volume={30},
  pages={863--874},
  year={2024},
  publisher={Nature Publishing Group}
}
```
