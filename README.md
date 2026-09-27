# CVS Inference Service

> Plantilla. Reemplaza esto por: qué hace el proyecto, en tres líneas, antes de publicar.

Servicio de inferencia para evaluación automática del Critical View of Safety
(CVS) en video de colecistectomía laparoscópica. Lleva el modelo de
investigación PercEVA-CVS a un servicio desplegable, con export a ONNX,
cuantización y medición de latencia.

![demo](docs/demo.gif) <!-- reemplazar -->

## Resultados

### Precisión (SAGES 2024 CVS Challenge)

| Modelo | Supervisión extra | macro mAP |
|---|---|---|
| LG-CVS (baseline publicado) | cajas | 57.6 ± 0.4 |
| **PercEVA-CVS** | ninguna | **65.1 ± 1.3** |

Promediado sobre tres semillas (42, 1, 50).

### Latencia

<!-- Generado con /bench — no editar a mano -->
| Backend | Precisión | p50 (ms) | p95 (ms) | Throughput | Tamaño | Δ exactitud |
|---|---|---|---|---|---|---|
| _pendiente_ | | | | | | |

## Instalación

```bash
uv sync
uv run pytest
```

## Uso

```bash
uv run uvicorn cvs_serve.api:app --reload
curl -X POST localhost:8000/predict -F "file=@frame.jpg"
```

## Atribución y licencia

Basado en **PercEVA-CVS**: S. Cañar, J. S. Vera, I. S. Tovar, P. Arbeláez,
"Annotation-Efficient Critical View of Safety Assessment with Vision Foundation
Models", SafeSurg Workshop, MICCAI 2026.
[Código original](https://github.com/BCV-Uniandes/PercEVA-CVS) — CC BY-NC-SA 4.0.

Este repositorio hereda la licencia **CC BY-NC-SA 4.0**.
