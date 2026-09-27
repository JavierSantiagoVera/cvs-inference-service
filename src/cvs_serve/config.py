"""Constantes del modelo PercEVA-CVS (configs/eva02.yaml y configs/perceiver_sages.yaml)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

# Etapa 1: encoder EVA-02 por frame.
ENCODER_NAME = "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k"
IMG_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBED_DIM = 1024

# Etapa 2: Perceiver temporal en modo online.
WINDOW_SIZE = 15
FPS = 1.0
# Tamaño de la tabla de posiciones aprendidas: 2048 s de video a 1 fps.
PE_MAX_LEN = 2048
# Hiperparámetros del Perceiver que se leen del config.json del checkpoint.
PERCEIVER_CONFIG_KEYS = (
    "d_in",
    "d_model",
    "nhead",
    "num_layers",
    "dim_ff",
    "K",
    "mlp_hidden",
    "pe_max_len",
)

# Export a ONNX. Tolerancias de paridad definidas en la skill export-onnx.
ONNX_OPSET = 18
PARITY_RTOL = 1e-3
PARITY_ATOL = 1e-5
ENCODER_ONNX = "encoder.onnx"
PERCEIVER_ONNX = "perceiver.onnx"

N_CLASSES = 3
CRITERIA = ("c1", "c2", "c3")
PRED_THRESHOLD = 0.5

# Archivos ONNX por precisión, dentro del directorio de modelos.
MODEL_FILES = {
    "fp32": (ENCODER_ONNX, PERCEIVER_ONNX),
    "int8": ("encoder.int8.onnx", "perceiver.int8.onnx"),
}


@dataclass(frozen=True)
class Settings:
    """Configuración del servicio, leída de variables de entorno CVS_*."""

    models_dir: Path = Path("models")
    # fp32 por defecto: la caída de mAP de int8 aún no está medida.
    precision: str = "fp32"
    max_sessions: int = 32
    session_ttl_s: float = 300.0
    num_threads: int = 0  # 0 = lo decide ONNX Runtime

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        settings = cls(
            models_dir=Path(env.get("CVS_MODELS_DIR", cls.models_dir)),
            precision=env.get("CVS_PRECISION", cls.precision),
            max_sessions=int(env.get("CVS_MAX_SESSIONS", cls.max_sessions)),
            session_ttl_s=float(env.get("CVS_SESSION_TTL_S", cls.session_ttl_s)),
            num_threads=int(env.get("CVS_NUM_THREADS", cls.num_threads)),
        )
        if settings.precision not in MODEL_FILES:
            raise ValueError(f"CVS_PRECISION debe ser uno de {sorted(MODEL_FILES)}")
        return settings
