from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from cvs_serve.config import EMBED_DIM, IMAGENET_MEAN, IMAGENET_STD, IMG_SIZE
from cvs_serve.model import TemporalWindow, preprocess


def _solid_frame(color: tuple[int, int, int], size: tuple[int, int] = (640, 360)) -> Image.Image:
    return Image.new("RGB", size, color)


def test_preprocess_outputs_normalized_square_tensor() -> None:
    out = preprocess(_solid_frame((255, 0, 128)))

    assert out.shape == (1, 3, IMG_SIZE, IMG_SIZE)
    assert out.dtype == np.float32
    expected = (np.array([1.0, 0.0, 128 / 255]) - IMAGENET_MEAN) / IMAGENET_STD
    np.testing.assert_allclose(out[0, :, 0, 0], expected, rtol=1e-5)


def test_preprocess_accepts_grayscale_and_rgba() -> None:
    for mode in ("L", "RGBA"):
        assert preprocess(Image.new(mode, (100, 50))).shape == (1, 3, IMG_SIZE, IMG_SIZE)


def test_window_pads_the_past_until_filled() -> None:
    window = TemporalWindow(window_size=4, embed_dim=2)

    first = window.push(np.array([1.0, 1.0]))
    second = window.push(np.array([2.0, 2.0]))

    assert first.n_valid == 1
    assert second.valid_mask.tolist() == [[False, False, True, True]]
    np.testing.assert_array_equal(second.x[0], [[0, 0], [0, 0], [1, 1], [2, 2]])
    assert second.positions.tolist() == [[0, 0, 0, 1]]
    assert second.key_index.tolist() == [3]


def test_window_slides_keeping_most_recent_frames() -> None:
    window = TemporalWindow(window_size=3, embed_dim=1)

    for value in range(6):
        inputs = window.push(np.array([float(value)]))

    assert inputs.n_valid == 3
    np.testing.assert_array_equal(inputs.x[0, :, 0], [3, 4, 5])
    assert inputs.positions.tolist() == [[3, 4, 5]]


def test_window_caps_positions_beyond_the_learned_table() -> None:
    window = TemporalWindow(window_size=3, embed_dim=1, pe_max_len=5)

    for value in range(8):
        inputs = window.push(np.array([float(value)]))

    assert inputs.positions.tolist() == [[4, 4, 4]]


def test_window_rounds_embeddings_to_float16_like_training_features() -> None:
    window = TemporalWindow(window_size=1, embed_dim=1)

    inputs = window.push(np.array([1.0001]))

    assert inputs.x.dtype == np.float32
    assert inputs.x[0, 0, 0] == np.float32(np.float16(1.0001))


def test_window_rejects_wrong_embedding_dim() -> None:
    window = TemporalWindow()

    with pytest.raises(ValueError, match=str(EMBED_DIM)):
        window.push(np.zeros(EMBED_DIM - 1))
