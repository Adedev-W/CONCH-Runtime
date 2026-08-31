from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def test_dockerfile_builds_one_cpu_gpu_compatible_runtime():
    dockerfile = (REPOSITORY_ROOT / "Dockerfile").read_text()

    assert "ARG PYTORCH_INDEX_URL=" in dockerfile
    assert "torch==2.13.0 torchvision==0.28.0" in dockerfile
    assert "CONCH_DEVICE=auto" in dockerfile
    assert 'uvicorn app:app --host 0.0.0.0 --port ${PORT:-80}' in dockerfile


def test_docker_context_excludes_credentials_and_model_weights():
    dockerignore = (REPOSITORY_ROOT / ".dockerignore").read_text()

    for entry in (".env", ".env.*", "*.pt", "*.pth", "*.bin", "*.safetensors"):
        assert entry in dockerignore
