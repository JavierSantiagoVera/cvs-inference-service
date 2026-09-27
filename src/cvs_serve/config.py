"""Constantes del modelo PercEVA-CVS (configs/eva02.yaml y configs/perceiver_sages.yaml)."""

from __future__ import annotations

# Etapa 1: encoder EVA-02 por frame.
IMG_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
EMBED_DIM = 1024

# Etapa 2: Perceiver temporal en modo online.
WINDOW_SIZE = 15
FPS = 1.0

N_CLASSES = 3
CRITERIA = ("c1", "c2", "c3")
PRED_THRESHOLD = 0.5
