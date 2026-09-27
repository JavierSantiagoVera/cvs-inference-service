"""Carga de checkpoints con modelos pequeños y pesos aleatorios, sin datos clínicos."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
safetensors_torch = pytest.importorskip("safetensors.torch")

from cvs_serve.model import encode, load_encoder, load_perceiver  # noqa: E402
from cvs_serve.perceiver import PerceiverLiteTemporalGated  # noqa: E402

TINY_PERCEIVER = {
    "d_in": 16,
    "d_model": 16,
    "nhead": 2,
    "num_layers": 1,
    "dim_ff": 8,
    "K": 4,
    "mlp_hidden": 8,
    "pe_max_len": 32,
}


def _save_training_checkpoint(model: torch.nn.Module, path: Path) -> None:
    """Guarda como el Trainer original: pesos bajo `model.` más `pos_weight` de la pérdida."""
    state = {f"model.{k}": v.contiguous() for k, v in model.state_dict().items()}
    state["pos_weight"] = torch.ones(3)
    path.parent.mkdir(parents=True)
    safetensors_torch.save_file(state, str(path))


def test_perceiver_loads_latest_training_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    old, latest = (
        PerceiverLiteTemporalGated(**TINY_PERCEIVER),
        PerceiverLiteTemporalGated(**TINY_PERCEIVER),
    )
    _save_training_checkpoint(old, tmp_path / "checkpoint-9" / "model.safetensors")
    _save_training_checkpoint(latest, tmp_path / "checkpoint-10" / "model.safetensors")
    (tmp_path / "config.json").write_text(json.dumps({**TINY_PERCEIVER, "wandb": True}))

    loaded = load_perceiver(tmp_path)

    x = torch.randn(2, 5, 16)
    key_index = torch.tensor([4, 4])
    positions = torch.arange(5).repeat(2, 1)
    with torch.inference_mode():
        expected = latest.eval()(x, key_index, positions)
        np.testing.assert_array_equal(loaded(x, key_index, positions), expected)


def test_perceiver_missing_checkpoint_raises(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps(TINY_PERCEIVER))

    with pytest.raises(FileNotFoundError):
        load_perceiver(tmp_path)


def test_encoder_loads_prefixed_checkpoint_and_returns_pre_logits(tmp_path: Path) -> None:
    name = "eva02_tiny_patch14_224"
    source = timm.create_model(name, pretrained=False, num_classes=3)
    checkpoint = tmp_path / "encoder.pt"
    torch.save({f"model.{k}": v for k, v in source.state_dict().items()}, checkpoint)

    encoder = load_encoder(checkpoint, model_name=name)
    pixels = np.random.default_rng(0).standard_normal((2, 3, 224, 224), dtype=np.float32)
    embeddings = encode(encoder, pixels)

    assert embeddings.shape == (2, source.num_features)
    with torch.inference_mode():
        logits = encoder.get_classifier()(torch.from_numpy(embeddings))
        np.testing.assert_allclose(logits, source.eval()(torch.from_numpy(pixels)), atol=1e-5)
