"""Endpoints de la API con un predictor de prueba y frames sintéticos."""

from __future__ import annotations

import io
from collections.abc import Iterator

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from cvs_serve.api import create_app
from cvs_serve.config import EMBED_DIM, WINDOW_SIZE, Settings
from cvs_serve.model import WindowInputs


class StubPredictor:
    """Devuelve probabilidades fijas y registra las ventanas que recibe."""

    def __init__(self, probs: tuple[float, float, float] = (0.9, 0.7, 0.2)) -> None:
        self.probs = np.array(probs)
        self.windows: list[WindowInputs] = []

    def embed(self, pixels: np.ndarray) -> np.ndarray:
        return np.full(EMBED_DIM, pixels.mean(), dtype=np.float32)

    def probabilities(self, inputs: WindowInputs) -> np.ndarray:
        self.windows.append(inputs)
        return self.probs


def _png(color: tuple[int, int, int] = (120, 30, 30)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _client(predictor: StubPredictor, **settings: object) -> TestClient:
    return TestClient(create_app(Settings(**settings), predictor=predictor))


@pytest.fixture
def client() -> Iterator[TestClient]:
    with _client(StubPredictor()) as c:
        yield c


def _send(client: TestClient, session_id: str, image: bytes | None = None) -> httpx.Response:
    files = {"file": ("frame.png", image or _png(), "image/png")}
    return client.post(f"/sessions/{session_id}/frames", files=files)


def test_session_accumulates_temporal_context_frame_by_frame() -> None:
    predictor = StubPredictor()
    with _client(predictor) as client:
        session_id = client.post("/sessions").json()["session_id"]

        results = [_send(client, session_id).json() for _ in range(3)]

    assert [r["frame_index"] for r in results] == [0, 1, 2]
    assert [r["window_filled"] for r in results] == [1, 2, 3]
    assert results[-1]["window_size"] == WINDOW_SIZE
    assert predictor.windows[-1].positions[0, -3:].tolist() == [0, 1, 2]


def test_frame_result_thresholds_each_criterion() -> None:
    with _client(StubPredictor((0.9, 0.7, 0.2))) as client:
        session_id = client.post("/sessions").json()["session_id"]
        partial = _send(client, session_id).json()

    with _client(StubPredictor((0.9, 0.7, 0.6))) as client:
        session_id = client.post("/sessions").json()["session_id"]
        achieved = _send(client, session_id).json()

    assert partial["criteria"] == {"c1": True, "c2": True, "c3": False}
    assert partial["probabilities"]["c3"] == pytest.approx(0.2)
    assert partial["cvs_achieved"] is False
    assert achieved["cvs_achieved"] is True


def test_sessions_are_independent(client: TestClient) -> None:
    first = client.post("/sessions").json()["session_id"]
    second = client.post("/sessions").json()["session_id"]

    _send(client, first)
    _send(client, first)

    assert _send(client, second).json()["window_filled"] == 1


def test_deleted_or_unknown_session_returns_404(client: TestClient) -> None:
    session_id = client.post("/sessions").json()["session_id"]

    assert client.delete(f"/sessions/{session_id}").status_code == 204
    assert _send(client, session_id).status_code == 404
    assert client.delete(f"/sessions/{session_id}").status_code == 404


def test_invalid_image_returns_400(client: TestClient) -> None:
    session_id = client.post("/sessions").json()["session_id"]

    assert _send(client, session_id, image=b"no es una imagen").status_code == 400


def test_session_limit_returns_503() -> None:
    with _client(StubPredictor(), max_sessions=2) as client:
        codes = [client.post("/sessions").status_code for _ in range(3)]

    assert codes == [201, 201, 503]


def test_inactive_sessions_expire() -> None:
    with _client(StubPredictor(), session_ttl_s=0.0) as client:
        session_id = client.post("/sessions").json()["session_id"]

        assert _send(client, session_id).status_code == 404


def test_health_reports_precision_and_active_sessions(client: TestClient) -> None:
    client.post("/sessions")

    assert client.get("/health").json() == {
        "status": "ok",
        "precision": "int8",
        "active_sessions": 1,
    }


def test_settings_reject_unknown_precision(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CVS_PRECISION", "fp16")

    with pytest.raises(ValueError, match="CVS_PRECISION"):
        Settings.from_env()
