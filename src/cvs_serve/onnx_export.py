"""Export a ONNX del encoder y del Perceiver, verificación de paridad y cuantización.

Uso:
    uv run python -m cvs_serve.onnx_export --weights perceva_cvs_weights/sages2024 --out models
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from torch import nn

from cvs_serve.config import (
    EMBED_DIM,
    ENCODER_ONNX,
    IMG_SIZE,
    MODEL_FILES,
    ONNX_OPSET,
    PARITY_ATOL,
    PARITY_RTOL,
    PERCEIVER_ONNX,
    WINDOW_SIZE,
)
from cvs_serve.model import TemporalWindow, load_encoder, load_perceiver

logger = logging.getLogger(__name__)


class ParityError(RuntimeError):
    """El modelo ONNX no reproduce la salida de PyTorch dentro de la tolerancia."""


class EncoderEmbedding(nn.Module):
    """Envuelve el encoder timm para que su salida sea el embedding pre-logits."""

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        feats = self.encoder.forward_features(pixels)
        pre = self.encoder.forward_head(feats, pre_logits=True)
        return pre.flatten(1)


def _export(
    model: nn.Module,
    example: dict[str, torch.Tensor],
    output_name: str,
    path: Path,
) -> Path:
    """Exporta con el exportador dynamo y el batch como único eje dinámico."""
    path.parent.mkdir(parents=True, exist_ok=True)
    batch = torch.export.Dim("batch", min=1, max=64)
    dynamic_shapes = {name: {0: batch} for name in example}
    program = torch.onnx.export(
        model.eval(),
        kwargs=example,
        input_names=list(example),
        output_names=[output_name],
        dynamic_shapes=dynamic_shapes,
        opset_version=ONNX_OPSET,
        dynamo=True,
        external_data=False,
        verbose=False,
    )
    program.save(str(path))
    logger.info("Exportado %s (%.1f MB)", path, path.stat().st_size / 2**20)
    return path


def export_encoder(encoder: nn.Module, path: Path, img_size: int = IMG_SIZE) -> Path:
    """Exporta el encoder a ONNX: entrada `pixels` [B, 3, H, W], salida `embedding` [B, D]."""
    example = {"pixels": torch.randn(2, 3, img_size, img_size)}
    return _export(EncoderEmbedding(encoder), example, "embedding", path)


def export_perceiver(
    perceiver: nn.Module,
    path: Path,
    window_size: int = WINDOW_SIZE,
    embed_dim: int = EMBED_DIM,
) -> Path:
    """Exporta el Perceiver a ONNX: entradas `x`, `key_index`, `positions`; salida `logits`."""
    example = {
        "x": torch.randn(2, window_size, embed_dim),
        "key_index": torch.full((2,), window_size - 1, dtype=torch.int64),
        "positions": torch.arange(window_size, dtype=torch.int64).repeat(2, 1),
    }
    return _export(perceiver, example, "logits", path)


def run_onnx(path: Path, inputs: dict[str, np.ndarray]) -> np.ndarray:
    """Ejecuta un modelo ONNX en CPU y devuelve su única salida."""
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    return session.run(None, inputs)[0]


def check_parity(model: nn.Module, onnx_path: Path, inputs: dict[str, np.ndarray]) -> float:
    """Compara PyTorch contra ONNX Runtime con la misma entrada.

    Devuelve la diferencia absoluta máxima. Lanza ParityError si supera
    rtol=PARITY_RTOL, atol=PARITY_ATOL.
    """
    with torch.inference_mode():
        expected = model.eval()(**{k: torch.from_numpy(v) for k, v in inputs.items()}).numpy()
    actual = run_onnx(onnx_path, inputs)
    max_diff = float(np.abs(expected - actual).max())
    try:
        np.testing.assert_allclose(actual, expected, rtol=PARITY_RTOL, atol=PARITY_ATOL)
    except AssertionError as err:
        raise ParityError(f"{onnx_path.name}: diferencia máxima {max_diff:.2e}") from err
    logger.info("Paridad OK %s batch=%d max_diff=%.2e", onnx_path.name, len(actual), max_diff)
    return max_diff


def quantize(src: Path, dst: Path) -> Path:
    """Cuantización dinámica INT8 de los pesos; las activaciones se cuantizan en ejecución.

    Antes borra las formas intermedias (value_info) que anota el exportador dynamo:
    la inferencia de formas del cuantizador las recalcula y a veces no coinciden.
    """
    model = onnx.load(str(src))
    del model.graph.value_info[:]
    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / src.name
        onnx.save(model, str(clean))
        quantize_dynamic(str(clean), str(dst), weight_type=QuantType.QInt8)
    logger.info("Cuantizado %s (%.1f MB)", dst, dst.stat().st_size / 2**20)
    return dst


def _perceiver_inputs(rng: np.random.Generator, batch: int) -> dict[str, np.ndarray]:
    """Ventanas como las del servicio: relleno de ceros al inicio y posiciones variadas."""
    x = rng.standard_normal((batch, WINDOW_SIZE, EMBED_DIM), dtype=np.float32)
    frame_ids = rng.integers(-WINDOW_SIZE + 1, 2000, size=(batch, 1)) + np.arange(WINDOW_SIZE)
    x[frame_ids < 0] = 0.0
    positions = np.clip(frame_ids, 0, None).astype(np.int64)
    key_index = np.full(batch, WINDOW_SIZE - 1, dtype=np.int64)
    return {"x": x, "key_index": key_index, "positions": positions}


def _pipeline_probs(
    encoder: Path, perceiver: Path, pixels: np.ndarray
) -> np.ndarray:  # [n_frames, n_classes]
    """Probabilidades frame a frame usando solo ONNX, como las calculará la API."""
    window = TemporalWindow()
    probs = []
    for frame in pixels:
        inputs = window.push(run_onnx(encoder, {"pixels": frame[np.newaxis]})[0])
        logits = run_onnx(
            perceiver,
            {"x": inputs.x, "key_index": inputs.key_index, "positions": inputs.positions},
        )
        probs.append(1.0 / (1.0 + np.exp(-logits[0])))
    return np.stack(probs)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, required=True, help="directorio sages2024")
    parser.add_argument("--out", type=Path, default=Path("models"))
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    encoder = EncoderEmbedding(load_encoder(args.weights / "eva02_encoder/best_eva02_enc_cvs.pt"))
    perceiver = load_perceiver(args.weights / "perceiver")
    enc_path = export_encoder(encoder.encoder, args.out / ENCODER_ONNX)
    per_path = export_perceiver(perceiver, args.out / PERCEIVER_ONNX)

    parity = {}
    for batch in (1, 3):
        pixels = rng.standard_normal((batch, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
        parity[f"encoder_b{batch}"] = check_parity(encoder, enc_path, {"pixels": pixels})
        parity[f"perceiver_b{batch}"] = check_parity(
            perceiver, per_path, _perceiver_inputs(rng, batch)
        )

    enc_q_name, per_q_name = MODEL_FILES["int8"]
    enc_q = quantize(enc_path, args.out / enc_q_name)
    per_q = quantize(per_path, args.out / per_q_name)

    # Sin frames de validación, la deriva se mide sobre entradas sintéticas:
    # es una señal de alarma, no un sustituto de la mAP.
    pixels = rng.standard_normal((WINDOW_SIZE, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    fp32 = _pipeline_probs(enc_path, per_path, pixels)
    int8 = _pipeline_probs(enc_q, per_q, pixels)
    agree = float(((fp32 >= 0.5) == (int8 >= 0.5)).mean())

    results = json.loads(args.results.read_text()) if args.results.exists() else {}
    results["export"] = {
        "opset": ONNX_OPSET,
        "parity_max_abs_diff": parity,
        "parity_tolerance": {"rtol": PARITY_RTOL, "atol": PARITY_ATOL},
        "size_mb": {
            p.name: round(p.stat().st_size / 2**20, 1) for p in (enc_path, per_path, enc_q, per_q)
        },
        "int8_vs_fp32_synthetic": {
            "max_abs_prob_diff": round(float(np.abs(fp32 - int8).max()), 4),
            "threshold_agreement": agree,
            "n_frames": WINDOW_SIZE,
        },
        "int8_map_drop": None,
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    logger.info("Resultados en %s", args.results)


if __name__ == "__main__":
    main()
