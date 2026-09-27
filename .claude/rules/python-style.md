---
paths: ["**/*.py"]
---

# Estilo de Python en este repo

- Type hints en toda función pública. `from __future__ import annotations` arriba.
- Docstrings en una línea para funciones simples; formato Google para las que
  tengan más de dos parámetros.
- Sin `print()` en `src/`: usar `logging.getLogger(__name__)`.
- Rutas con `pathlib.Path`, nunca concatenando strings.
- Excepciones específicas, nunca `except Exception:` desnudo salvo en el borde
  de la API, y ahí siempre con log.
- Constantes de configuración en `config.py`, no dispersas en el código.
- Un test por comportamiento observable, no uno por función.
