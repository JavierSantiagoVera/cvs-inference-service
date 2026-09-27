"""Export a ONNX con modelos pequeños y pesos aleatorios, sin datos clínicos."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
timm = pytest.importorskip("timm")
pytest.importorskip("onnxscript")

from cvs_serve.onnx_export import (  # noqa: E402
    EncoderEmbedding,
    ParityError,
    check_parity,
    export_encoder,
    export_perceiver,
    quantize,
    run_onnx,
)
from cvs_serve.perceiver import PerceiverLiteTemporalGated  # noqa: E402

WINDOW, DIM = 5, 16
TINY_PERCEIVER = {
    "d_in": DIM,
    "d_model": DIM,
    "nhead": 2,
    "num_layers": 1,
    "dim_ff": 8,
    "K": 4,
    "mlp_hidden": 8,
    "pe_max_len": 32,
}


def _window_inputs(batch: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(batch)
    return {
        "x": rng.standard_normal((batch, WINDOW, DIM), dtype=np.float32),
        "key_index": np.full(batch, WINDOW - 1, dtype=np.int64),
        "positions": np.clip(np.arange(WINDOW) + rng.integers(-3, 20, (batch, 1)), 0, None),
    }


@pytest.fixture(scope="module")
def perceiver_onnx(tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, Path]:
    torch.manual_seed(0)
    model = PerceiverLiteTemporalGated(**TINY_PERCEIVER).eval()
    path = tmp_path_factory.mktemp("onnx") / "perceiver.onnx"
    export_perceiver(model, path, window_size=WINDOW, embed_dim=DIM)
    return model, path


@pytest.mark.parametrize("batch", [1, 4])
def test_perceiver_export_matches_pytorch_for_any_batch(
    perceiver_onnx: tuple[torch.nn.Module, Path], batch: int
) -> None:
    model, path = perceiver_onnx

    assert check_parity(model, path, _window_inputs(batch)) < 1e-5


def test_parity_check_detects_a_diverging_export(
    perceiver_onnx: tuple[torch.nn.Module, Path],
) -> None:
    _, path = perceiver_onnx
    other = PerceiverLiteTemporalGated(**TINY_PERCEIVER).eval()

    with pytest.raises(ParityError):
        check_parity(other, path, _window_inputs(2))


def test_quantized_perceiver_runs_and_stays_close(
    perceiver_onnx: tuple[torch.nn.Module, Path], tmp_path: Path
) -> None:
    _, path = perceiver_onnx
    inputs = _window_inputs(3)

    quantized = quantize(path, tmp_path / "perceiver.int8.onnx")

    np.testing.assert_allclose(run_onnx(quantized, inputs), run_onnx(path, inputs), atol=0.1)


def test_encoder_export_outputs_pre_logits_embedding(tmp_path: Path) -> None:
    encoder = timm.create_model("eva02_tiny_patch14_224", pretrained=False, num_classes=3).eval()
    path = export_encoder(encoder, tmp_path / "encoder.onnx", img_size=224)

    for batch in (1, 2):
        pixels = np.random.default_rng(batch).standard_normal(
            (batch, 3, 224, 224), dtype=np.float32
        )
        check_parity(EncoderEmbedding(encoder), path, {"pixels": pixels})
        assert run_onnx(path, {"pixels": pixels}).shape == (batch, encoder.num_features)
