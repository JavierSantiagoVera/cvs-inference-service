"""Carga del modelo, preprocesamiento de frames y ventana temporal para PercEVA-CVS."""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from cvs_serve.config import (
    EMBED_DIM,
    ENCODER_NAME,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    N_CLASSES,
    PE_MAX_LEN,
    PERCEIVER_CONFIG_KEYS,
    WINDOW_SIZE,
)

if TYPE_CHECKING:
    import onnxruntime as ort
    from torch import nn

    from cvs_serve.perceiver import PerceiverLiteTemporalGated

logger = logging.getLogger(__name__)

_MEAN = np.array(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
_STD = np.array(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)


def preprocess(image: Image.Image) -> np.ndarray:
    """Convierte un frame en el tensor de entrada del encoder, forma [1, 3, 448, 448].

    Replica el transform de inference_image_encoder.py: RGB, Resize((448, 448))
    bilineal de PIL, escala a [0, 1] y normalización ImageNet.
    """
    rgb = image.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR)
    chw = np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return ((chw - _MEAN) / _STD)[np.newaxis]


def _submodule_state(state_dict: dict, prefix: str = "model.") -> dict:
    """Extrae los pesos del submódulo `model` del wrapper de entrenamiento.

    El resto de claves (p. ej. `pos_weight` de la pérdida) no son del modelo.
    """
    dropped = sorted(k for k in state_dict if not k.startswith(prefix))
    if dropped:
        logger.info("Claves del checkpoint que no son del modelo, ignoradas: %s", dropped)
    return {k.removeprefix(prefix): v for k, v in state_dict.items() if k.startswith(prefix)}


def load_encoder(checkpoint: Path, model_name: str = ENCODER_NAME) -> nn.Module:
    """Carga el encoder EVA-02 de la etapa 1 desde un .pt, en modo eval y en CPU."""
    import timm
    import torch

    model = timm.create_model(model_name, pretrained=False, num_classes=N_CLASSES)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(_submodule_state(state), strict=True)
    logger.info("Encoder cargado desde %s", checkpoint)
    return model.eval()


def find_perceiver_weights(run_dir: Path) -> Path:
    """Busca model.safetensors en el directorio del run o en su checkpoint-N más reciente."""
    direct = run_dir / "model.safetensors"
    if direct.is_file():
        return direct
    candidates = sorted(
        run_dir.glob("checkpoint-*/model.safetensors"),
        key=lambda p: int(p.parent.name.removeprefix("checkpoint-")),
    )
    if not candidates:
        raise FileNotFoundError(f"No hay model.safetensors en {run_dir}")
    return candidates[-1]


def load_perceiver(run_dir: Path) -> PerceiverLiteTemporalGated:
    """Carga el Perceiver de la etapa 2 desde un directorio con config.json y checkpoint."""
    from safetensors.torch import load_file

    from cvs_serve.perceiver import PerceiverLiteTemporalGated

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    kwargs = {k: config[k] for k in PERCEIVER_CONFIG_KEYS if k in config}
    model = PerceiverLiteTemporalGated(**kwargs)
    weights = find_perceiver_weights(run_dir)
    model.load_state_dict(_submodule_state(load_file(weights)), strict=True)
    logger.info("Perceiver cargado desde %s", weights)
    return model.eval()


def encode(encoder: nn.Module, pixels: np.ndarray) -> np.ndarray:
    """Devuelve los embeddings pre-logits del encoder, [B, 1024], como en extract_ft.py."""
    import torch

    with torch.inference_mode():
        feats = encoder.forward_features(torch.from_numpy(pixels))
        pre = encoder.forward_head(feats, pre_logits=True)
    return pre.reshape(pre.shape[0], -1).numpy()


def make_onnx_session(path: Path, num_threads: int = 0) -> ort.InferenceSession:
    """Crea una sesión de ONNX Runtime en CPU con la configuración del servicio.

    Desactiva el spinning de los hilos intra-op: con dos sesiones alternándose
    (encoder y Perceiver), los hilos ociosos de una en espera activa le quitan
    CPU a la otra. En un Ryzen 7 9800X3D bajó el p50 por frame de 1076 a 707 ms
    en fp32 y de 474 a 276 ms en int8.
    """
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = num_threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


class OnnxPredictor:
    """Ejecuta el encoder y el Perceiver exportados con ONNX Runtime, sin torch.

    Las sesiones se crean una vez; `InferenceSession.run` es seguro entre hilos.
    """

    def __init__(self, encoder_path: Path, perceiver_path: Path, num_threads: int = 0) -> None:
        self._encoder = make_onnx_session(encoder_path, num_threads)
        self._perceiver = make_onnx_session(perceiver_path, num_threads)
        logger.info("Modelos ONNX cargados: %s, %s", encoder_path, perceiver_path)

    def embed(self, pixels: np.ndarray) -> np.ndarray:
        """Embedding [1024] de un frame ya preprocesado, [1, 3, 448, 448]."""
        return self._encoder.run(None, {"pixels": pixels})[0][0]

    def probabilities(self, inputs: WindowInputs) -> np.ndarray:
        """Probabilidad de cada criterio [3] para la ventana que termina en el frame actual."""
        logits = self._perceiver.run(
            None,
            {"x": inputs.x, "key_index": inputs.key_index, "positions": inputs.positions},
        )[0][0]
        return 1.0 / (1.0 + np.exp(-logits))


@dataclass(frozen=True)
class WindowInputs:
    """Entradas del Perceiver para una ventana, con batch de 1."""

    x: np.ndarray  # [1, T, D] float32, ceros en las posiciones sin frame
    valid_mask: np.ndarray  # [1, T] bool; el modelo no la usa, solo informa
    positions: np.ndarray  # [1, T] int64, segundos desde el inicio del video
    key_index: np.ndarray  # [1] int64, siempre T - 1 en modo online

    @property
    def n_valid(self) -> int:
        return int(self.valid_mask.sum())


class TemporalWindow:
    """Ventana causal de embeddings de un video, alimentada a 1 fps.

    Replica el modo online de TemporalWindowCVSDataset: el frame actual ocupa la
    última posición, los huecos al inicio del video se rellenan con ceros y se
    marcan en valid_mask, y las posiciones son índices absolutos acotados en 0.
    Los embeddings se redondean a float16 porque así se guardaron las features
    con las que se entrenó el Perceiver. Las posiciones se acotan en
    pe_max_len - 1, porque la tabla aprendida no cubre más allá.
    """

    def __init__(
        self,
        window_size: int = WINDOW_SIZE,
        embed_dim: int = EMBED_DIM,
        pe_max_len: int = PE_MAX_LEN,
    ) -> None:
        self.window_size = window_size
        self.embed_dim = embed_dim
        self.pe_max_len = pe_max_len
        self._embeddings: deque[np.ndarray] = deque(maxlen=window_size)
        self._frames_seen = 0

    def push(self, embedding: np.ndarray) -> WindowInputs:
        """Agrega el embedding del frame actual y devuelve la ventana que termina en él."""
        embedding = np.asarray(embedding).reshape(-1).astype(np.float16).astype(np.float32)
        if embedding.shape[0] != self.embed_dim:
            raise ValueError(
                f"embedding de dimensión {embedding.shape[0]}, se esperaba {self.embed_dim}"
            )
        self._embeddings.append(embedding)
        self._frames_seen += 1
        if self._frames_seen == self.pe_max_len + 1:
            logger.warning(
                "La sesión superó %d frames; las posiciones quedan fijas en %d",
                self.pe_max_len,
                self.pe_max_len - 1,
            )
        return self._build()

    def _build(self) -> WindowInputs:
        t = self.window_size
        n_valid = len(self._embeddings)
        x = np.zeros((1, t, self.embed_dim), dtype=np.float32)
        x[0, t - n_valid :] = np.stack(self._embeddings)
        valid_mask = np.zeros((1, t), dtype=bool)
        valid_mask[0, t - n_valid :] = True
        key_frame = self._frames_seen - 1
        positions = np.arange(key_frame - t + 1, key_frame + 1)
        positions = np.clip(positions, 0, self.pe_max_len - 1)
        return WindowInputs(
            x=x,
            valid_mask=valid_mask,
            positions=positions.astype(np.int64)[np.newaxis],
            key_index=np.array([t - 1], dtype=np.int64),
        )
