# Imagen de producción: solo ONNX Runtime, sin torch.
# Los modelos no se copian a la imagen; se montan en /models al arrancar:
#   docker run -p 8000:8000 -v "$(pwd)/models:/models:ro" cvs-serve

FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:0.12.19 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app

# Dependencias primero, para reutilizar la capa cuando solo cambia el código.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
RUN uv sync --frozen --no-default-groups --no-editable


FROM python:3.12-slim-bookworm
RUN useradd --system --uid 1000 --no-create-home app
COPY --from=builder --chown=app:app /app/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CVS_MODELS_DIR=/models \
    CVS_PRECISION=int8
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=4)"]

# Un solo worker: las sesiones viven en la memoria del proceso.
CMD ["uvicorn", "cvs_serve.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
