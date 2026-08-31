import copy

import pytest
import torch

from conch.open_clip_custom.coca_model import CoCa
from conch.open_clip_custom.factory import create_model, read_state_dict


def _tiny_config():
    return {
        "embed_dim": 8,
        "embed_dim_caption": 8,
        "vision_cfg": {
            "layers": 1,
            "width": 16,
            "num_heads": 2,
            "mlp_ratio": 2,
            "image_size": 8,
            "patch_size": 4,
            "attentional_pool_contrast": True,
            "attn_pooler_heads": 2,
            "output_tokens": False,
        },
        "text_cfg": {
            "context_length": 4,
            "vocab_size": 32,
            "width": 16,
            "heads": 2,
            "layers": 1,
            "embed_cls": False,
            "output_tokens": False,
        },
        "multimodal_cfg": {
            "context_length": 4,
            "width": 16,
            "heads": 2,
            "layers": 0,
        },
    }


def test_read_state_dict_keeps_module_prefix_compatibility(tmp_path):
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"state_dict": {"module.weight": torch.ones(2)}}, checkpoint_path)

    state_dict = read_state_dict(str(checkpoint_path))

    assert list(state_dict) == ["weight"]
    torch.testing.assert_close(state_dict["weight"], torch.ones(2))


def test_jit_flag_exports_with_example_inputs(tmp_path):
    config = _tiny_config()
    source_model = CoCa(**copy.deepcopy(config)).eval()
    checkpoint_path = tmp_path / "model.pt"
    torch.save(source_model.state_dict(), checkpoint_path)

    image = torch.randn(1, 3, 8, 8)
    text = torch.tensor([[1, 2, 3, 0]], dtype=torch.long)
    exported = create_model(
        copy.deepcopy(config),
        checkpoint_path=str(checkpoint_path),
        jit=True,
        example_inputs=(image, text),
    )

    assert isinstance(exported, torch.export.ExportedProgram)
    with torch.no_grad():
        expected = source_model(image, text)
        actual = exported.module()(image, text)

    assert set(actual) == set(expected)
    for key in expected:
        assert actual[key].shape == expected[key].shape


def test_jit_flag_requires_example_inputs():
    with pytest.raises(ValueError, match="example_inputs"):
        create_model(copy.deepcopy(_tiny_config()), jit=True)
