"""Preprocesamiento de frames y ventana temporal para PercEVA-CVS."""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
from PIL import Image

from cvs_serve.config import EMBED_DIM, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE, WINDOW_SIZE

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


@dataclass(frozen=True)
class WindowInputs:
    """Entradas del Perceiver para una ventana, con batch de 1."""

    x: np.ndarray  # [1, T, D] float32, ceros en las posiciones sin frame
    valid_mask: np.ndarray  # [1, T] bool
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
    """

    def __init__(self, window_size: int = WINDOW_SIZE, embed_dim: int = EMBED_DIM) -> None:
        self.window_size = window_size
        self.embed_dim = embed_dim
        self._embeddings: deque[np.ndarray] = deque(maxlen=window_size)
        self._frames_seen = 0

    def push(self, embedding: np.ndarray) -> WindowInputs:
        """Agrega el embedding del frame actual y devuelve la ventana que termina en él."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if embedding.shape[0] != self.embed_dim:
            raise ValueError(
                f"embedding de dimensión {embedding.shape[0]}, se esperaba {self.embed_dim}"
            )
        self._embeddings.append(embedding)
        self._frames_seen += 1
        return self._build()

    def _build(self) -> WindowInputs:
        t = self.window_size
        n_valid = len(self._embeddings)
        x = np.zeros((1, t, self.embed_dim), dtype=np.float32)
        x[0, t - n_valid :] = np.stack(self._embeddings)
        valid_mask = np.zeros((1, t), dtype=bool)
        valid_mask[0, t - n_valid :] = True
        key_frame = self._frames_seen - 1
        positions = np.clip(np.arange(key_frame - t + 1, key_frame + 1), 0, None)
        return WindowInputs(
            x=x,
            valid_mask=valid_mask,
            positions=positions.astype(np.int64)[np.newaxis],
            key_index=np.array([t - 1], dtype=np.int64),
        )
