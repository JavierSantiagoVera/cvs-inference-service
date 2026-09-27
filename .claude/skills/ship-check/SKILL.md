---
name: ship-check
description: Verificación previa a publicar o compartir el repositorio. Corre tests, lint y auditoría de publicación, y reporta si está listo.
when_to_use: El usuario va a hacer público el repo, compartir el enlace, o pregunta si está listo para publicar
allowed-tools: Bash(uv run *), Bash(git status*), Bash(git log*)
---

## Estado del repositorio

```!
git status --short
git log --oneline -5
```

## Procedimiento

1. Corre `uv run ruff check .` y `uv run pytest`. Si algo falla, **para aquí** y
   reporta qué falla. No sigas con el resto.

2. Invoca al subagente `release-auditor` para la auditoría de secretos, datos,
   licencia y README.

3. Verifica que el README tenga:
   - Qué hace el proyecto, en las primeras tres líneas.
   - Un GIF o captura del demo.
   - La tabla de resultados generada desde `benchmarks/results.json`.
   - Instrucciones de instalación que funcionen desde cero.
   - Atribución y licencia.

4. Verifica que las cifras del README coincidan con `benchmarks/results.json` y
   con las publicadas en el paper (65.1% mAP, +7.5 puntos). Si no coinciden,
   es un bloqueo.

## Salida

Un veredicto de una línea: **LISTO** o **NO LISTO**, seguido de la lista de lo
que falta. Sin rodeos.
