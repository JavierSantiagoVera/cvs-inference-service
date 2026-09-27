# CVS Inference Service

Servicio de inferencia para evaluación del Critical View of Safety en video
laparoscópico. Toma el modelo de investigación PercEVA-CVS y lo lleva a un
servicio desplegable: export a ONNX, cuantización, medición de latencia y API.

## Comandos

```bash
uv sync                      # instalar dependencias
uv run pytest                # tests
uv run ruff check . --fix    # lint
uv run ruff format .         # formato
uv run uvicorn cvs_serve.api:app --reload   # API en local
docker build -t cvs-serve .  # imagen
```

## Estructura

- `src/cvs_serve/` — código del paquete.
  - `model.py` — carga del modelo y preprocesamiento.
  - `onnx_export.py` — export y verificación de paridad numérica.
  - `api.py` — endpoints FastAPI.
- `benchmarks/` — scripts y resultados de latencia. `results.json` es la fuente
  de verdad de la tabla del README.
- `tests/` — pytest. Un test por comportamiento, no por función.
- `scripts/` — utilidades de un solo uso.

## Reglas del proyecto

### Datos y privacidad
- **Nunca** commitear datos de pacientes, frames de video quirúrgico, ni
  archivos `.npz`, `.pt`, `.onnx` con pesos entrenados sobre datos clínicos.
- Los datos de prueba van en `tests/fixtures/` y deben ser sintéticos o
  públicos, nunca clínicos.
- Antes de cualquier commit que agregue binarios, verificar `.gitignore`.

### Licencia
- El modelo original (PercEVA-CVS) es **CC BY-NC-SA 4.0**. Cualquier derivado
  mantiene atribución y la misma licencia. No publicar pesos sin confirmar con
  los coautores.
- Citar siempre: Cañar, Vera, Tovar, Arbeláez — SafeSurg Workshop, MICCAI 2026.

### Cifras
- Las métricas de rendimiento publicadas son: **65.1% macro mAP** en SAGES 2024,
  7.5 puntos sobre el baseline LG-CVS, promediado sobre tres semillas (42, 1, 50).
- **No usar** las cifras de la tesis (66.04% / +11.7), que corresponden a una
  versión anterior del modelo y a otro protocolo de evaluación.
- Las cifras de latencia del README se generan con `/bench`, nunca a mano.

### Código
- Python 3.11+. Type hints en toda función pública.
- Formato y lint con `ruff`. No introducir black, isort ni flake8.
- Sin `print()` en `src/`: usar `logging`.
- Toda función que toque el modelo debe tener al menos un test.

### Git
- Rama por cambio, mensajes en imperativo y en inglés.
- No commitear si `pytest` o `ruff check` fallan.

## Notas de contexto

- Este repo es portafolio público además de código: el README y la limpieza del
  historial importan tanto como la implementación.
- Objetivo de publicación: **19 de octubre de 2026**.
