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
| PyTorch | fp32 | 930.4 | 1098.6 | 1.07 frames/s | 1264.8 MB (pesos en memoria) | referencia |
| ONNX Runtime | fp32 | 740.4 | 789.1 | 1.35 frames/s | 1267.9 MB | paridad con PyTorch: dif. máx. 4.8e-6 |
| ONNX Runtime | int8 (dinámica) | 275.5 | 311.9 | 3.63 frames/s | 341.7 MB | **mAP no medida** (sin set de validación) |

Paso completo por frame con batch 1 (encoder EVA-02 Large a 448×448, ventana
temporal y Perceiver), que es lo que hace la API en cada petición; el encoder
es más del 95% del tiempo. Throughput = 1000 / p50, frames procesados en serie.
CPU AMD Ryzen 7 9800X3D (8 núcleos / 16 hilos), sin GPU, Windows 11, 16 hilos
en todos los backends; PyTorch 2.14.0, ONNX Runtime 1.30.0 (spinning de hilos
desactivado, como en el servicio), Python 3.14.3. 120 iteraciones tras 15 de
calentamiento, dos corridas por etapa (variación de p50 < 2%), cada backend en
un proceso propio. Script: `benchmarks/bench_latency.py`; datos crudos en
`benchmarks/results.json`.

Sobre int8: con 15 frames sintéticos, la probabilidad difiere de fp32 como
máximo en 0.021 y la decisión por criterio coincide en 44 de 45 casos. Eso no
reemplaza medir la mAP en SAGES; hasta entonces el servicio usa fp32 por defecto.

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
