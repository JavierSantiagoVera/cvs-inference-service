"""Constantes del modelo PercEVA-CVS (configs/eva02.yaml y configs/perceiver_sages.yaml)."""

from __future__ import annotations

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

N_CLASSES = 3
CRITERIA = ("c1", "c2", "c3")
PRED_THRESHOLD = 0.5
