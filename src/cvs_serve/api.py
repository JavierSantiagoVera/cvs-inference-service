"""API de inferencia CVS por sesión de video.

El cliente abre una sesión por video, envía un frame por segundo desde el inicio
del procedimiento y recibe, para cada frame, la evaluación de los tres criterios
de la Critical View of Safety usando los últimos 15 segundos de contexto.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile, status
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel

from cvs_serve.config import CRITERIA, MODEL_FILES, PRED_THRESHOLD, WINDOW_SIZE, Settings
from cvs_serve.model import OnnxPredictor, TemporalWindow, preprocess

logger = logging.getLogger(__name__)

MAX_UPLOAD_BYTES = 10 * 2**20

DESCRIPTION = (
    "Evaluación automática de la Critical View of Safety en colecistectomía "
    "laparoscópica con PercEVA-CVS (Cañar, Vera, Tovar, Arbeláez — SafeSurg, "
    "MICCAI 2026). Herramienta de investigación: no es un dispositivo médico ni "
    "debe usarse para decisiones clínicas."
)


class SessionCreated(BaseModel):
    session_id: str
    window_size: int
    expected_fps: float = 1.0


class FrameResult(BaseModel):
    session_id: str
    frame_index: int
    probabilities: dict[str, float]
    criteria: dict[str, bool]
    cvs_achieved: bool
    window_filled: int
    window_size: int


class Health(BaseModel):
    status: str
    precision: str
    active_sessions: int


@dataclass
class _Session:
    window: TemporalWindow = field(default_factory=TemporalWindow)
    lock: threading.Lock = field(default_factory=threading.Lock)
    last_seen: float = field(default_factory=time.monotonic)
    frames: int = 0


class SessionStore:
    """Sesiones en memoria con límite de cantidad y expiración por inactividad."""

    def __init__(self, max_sessions: int, ttl_s: float) -> None:
        self.max_sessions = max_sessions
        self.ttl_s = ttl_s
        self._sessions: dict[str, _Session] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._sessions)

    def _purge_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, s in self._sessions.items() if now - s.last_seen >= self.ttl_s]
        for key in expired:
            del self._sessions[key]
        if expired:
            logger.info("Sesiones expiradas: %d", len(expired))

    def create(self) -> str | None:
        """Crea una sesión y devuelve su id, o None si se alcanzó el límite."""
        with self._lock:
            self._purge_expired()
            if len(self._sessions) >= self.max_sessions:
                return None
            session_id = uuid.uuid4().hex
            self._sessions[session_id] = _Session()
            return session_id

    def get(self, session_id: str) -> _Session | None:
        with self._lock:
            self._purge_expired()
            session = self._sessions.get(session_id)
            if session is not None:
                session.last_seen = time.monotonic()
            return session

    def delete(self, session_id: str) -> bool:
        with self._lock:
            return self._sessions.pop(session_id, None) is not None


def _read_frame(upload: UploadFile) -> Image.Image:
    data = upload.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Frame mayor a {MAX_UPLOAD_BYTES} bytes"
        )
    try:
        image = Image.open(io.BytesIO(data))
        image.load()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as err:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "El archivo no es una imagen válida"
        ) from err
    return image


def create_app(settings: Settings | None = None, predictor: OnnxPredictor | None = None) -> FastAPI:
    """Crea la app. Si no se pasa un predictor, carga los ONNX de settings.models_dir."""
    settings = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if predictor is None:
            encoder_file, perceiver_file = MODEL_FILES[settings.precision]
            app.state.predictor = OnnxPredictor(
                settings.models_dir / encoder_file,
                settings.models_dir / perceiver_file,
                num_threads=settings.num_threads,
            )
        else:
            app.state.predictor = predictor
        app.state.sessions = SessionStore(settings.max_sessions, settings.session_ttl_s)
        yield

    app = FastAPI(title="CVS Inference Service", description=DESCRIPTION, lifespan=lifespan)

    @app.get("/health")
    def health(request: Request) -> Health:
        return Health(
            status="ok",
            precision=settings.precision,
            active_sessions=len(request.app.state.sessions),
        )

    @app.post("/sessions", status_code=status.HTTP_201_CREATED)
    def create_session(request: Request) -> SessionCreated:
        session_id = request.app.state.sessions.create()
        if session_id is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE, "Límite de sesiones activas alcanzado"
            )
        return SessionCreated(session_id=session_id, window_size=WINDOW_SIZE)

    @app.post("/sessions/{session_id}/frames")
    def add_frame(
        session_id: str, request: Request, file: Annotated[UploadFile, File()]
    ) -> FrameResult:
        session = request.app.state.sessions.get(session_id)
        if session is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Sesión inexistente o expirada")
        predictor: OnnxPredictor = request.app.state.predictor

        # El encoder no depende del estado: corre fuera del candado de la sesión.
        embedding = predictor.embed(preprocess(_read_frame(file)))
        with session.lock:
            inputs = session.window.push(embedding)
            probs = predictor.probabilities(inputs)
            frame_index = session.frames
            session.frames += 1

        criteria = {
            name: bool(p >= PRED_THRESHOLD) for name, p in zip(CRITERIA, probs, strict=True)
        }
        return FrameResult(
            session_id=session_id,
            frame_index=frame_index,
            probabilities={name: float(p) for name, p in zip(CRITERIA, probs, strict=True)},
            criteria=criteria,
            cvs_achieved=all(criteria.values()),
            window_filled=inputs.n_valid,
            window_size=WINDOW_SIZE,
        )

    @app.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_session(session_id: str, request: Request) -> Response:
        if not request.app.state.sessions.delete(session_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Sesión inexistente o expirada")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    return app


# Para `uvicorn cvs_serve.api:app`. Los modelos se cargan al arrancar, no al importar.
app = create_app()
