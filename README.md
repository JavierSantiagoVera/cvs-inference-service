# CVS Inference Service

Servicio de inferencia en tiempo real para evaluar la **Critical View of Safety
(CVS)** en video de colecistectomía laparoscópica. Toma el modelo de
investigación **PercEVA-CVS** (SafeSurg, MICCAI 2026) y lo convierte en una API
desplegable: export a ONNX verificado numéricamente, cuantización INT8,
benchmarks de latencia reproducibles e imagen Docker sin PyTorch.

> **Herramienta de investigación.** No es un dispositivo médico y no debe
> usarse para tomar decisiones clínicas.

## Qué hace

En una colecistectomía, antes de cortar el conducto y la arteria císticos, el
cirujano debe confirmar tres criterios visuales (la CVS). El servicio recibe el
video a 1 frame por segundo y, para cada frame, estima si se cumple cada
criterio usando los últimos 15 segundos de contexto:

```
frame (JPG/PNG)
   │  preprocess          resize 448×448, normalización ImageNet
   ▼
encoder.onnx              EVA-02 Large → embedding de 1024 (una vez por frame)
   │
   ▼
ventana de la sesión      últimos 15 embeddings, en memoria
   │
   ▼
perceiver.onnx            Perceiver temporal con compuertas → C1, C2, C3
```

El encoder corre una sola vez por frame y su resultado queda en la ventana de
la sesión, así cada petición cuesta un paso del encoder y no quince.

## Resultados

### Precisión (SAGES 2024 CVS Challenge)

| Modelo | Supervisión extra | macro mAP |
|---|---|---|
| LG-CVS (baseline publicado) | cajas | 57.6 ± 0.4 |
| **PercEVA-CVS** | ninguna | **65.1 ± 1.3** |

Promediado sobre tres semillas (42, 1, 50). Son cifras del paper, no
re-medidas aquí. El servicio carga los pesos publicados con `strict=True`; el
Perceiver adaptado da una diferencia de 0.0 frente al código original, y los
modelos ONNX fp32 difieren de PyTorch en 4.8e-6 como máximo.

### Latencia

<!-- bench:start — generado por benchmarks/bench_latency.py, no editar a mano -->
| Backend | Precisión | p50 (ms) | p95 (ms) | Throughput | Tamaño | Δ exactitud |
|---|---|---|---|---|---|---|
| PyTorch | fp32 | 930.4 | 1098.6 | 1.07 frames/s | 1264.8 MB (pesos en memoria) | referencia |
| ONNX Runtime | fp32 | 740.4 | 789.1 | 1.35 frames/s | 1267.9 MB | paridad con PyTorch: dif. máx. 4.8e-6 |
| ONNX Runtime | int8 (dinámica) | 275.5 | 311.9 | 3.63 frames/s | 341.7 MB | +0.3 pts de mAP (IC 95%: -0.4 a +0.9) |

Paso completo por frame con batch 1 (encoder EVA-02 Large a 448×448, ventana
temporal y Perceiver), que es lo que hace la API en cada petición; el encoder
es más del 95% del tiempo. Throughput = 1000 / p50, frames procesados en serie.
CPU AMD Ryzen 7 9800X3D 8-Core Processor, sin GPU, Windows-11-10.0.26200-SP0,
16 hilos en todos los backends; PyTorch 2.14.0+cpu, ONNX Runtime 1.30.0
(spinning de hilos desactivado, como en el servicio), Python 3.14.3. 120
iteraciones tras 15 de calentamiento, 2 corridas por etapa (variación de p50 <
2%), cada backend en un proceso propio. Script: `benchmarks/bench_latency.py`;
datos crudos en `benchmarks/results.json`.

Sobre int8: en 60 de los 300 videos de test de SAGES 2024 (elegidos al azar con
semilla 0; 1080 fotogramas clave, etiqueta por voto mayoritario), la macro mAP
pasa de 63.8 en fp32 a 64.2 en int8 (+0.3 puntos; IC 95% por bootstrap de
videos: -0.4 a +0.9). La decisión con umbral 0.5 coincide en el 98.3% de los
casos. Es un subconjunto del test, así que su mAP absoluta no es comparable con
la del paper; lo que mide es la diferencia entre precisiones sobre los mismos
frames. Script: `benchmarks/eval_map.py`.
<!-- bench:end -->

## Inicio rápido

### 1. Pesos y export a ONNX

Los pesos los publican los autores en el
[repositorio original](https://github.com/BCV-Uniandes/PercEVA-CVS#pre-trained-weights)
(`perceva_cvs_weights_v1.zip`). Este repositorio no los redistribuye.

```bash
uv sync                                   # dependencias, incluido torch para exportar
unzip perceva_cvs_weights_v1.zip          # crea perceva_cvs_weights/
uv run python -m cvs_serve.onnx_export --weights perceva_cvs_weights/sages2024
```

El export verifica la paridad con PyTorch (batch 1 y 3) antes de cuantizar y
deja en `models/` `encoder.onnx`, `perceiver.onnx` y sus versiones `.int8.onnx`.
Si la paridad falla, se detiene sin cuantizar.

### 2. Levantar la API

Con Docker (imagen sin torch; los modelos se montan, no se copian):

```bash
docker build -t cvs-serve .
docker run -p 8000:8000 -v "$(pwd)/models:/models:ro" cvs-serve
```

O en local:

```bash
uv run uvicorn cvs_serve.api:app
```

La documentación interactiva queda en `http://localhost:8000/docs`.

## Uso de la API

Una sesión por video. Los frames se envían a 1 fps, en orden, desde el inicio
del procedimiento.

```bash
# Abrir una sesión
curl -X POST localhost:8000/sessions
# {"session_id": "b4cd...", "window_size": 15, "expected_fps": 1.0}

# Enviar un frame (repetir cada segundo)
curl -X POST localhost:8000/sessions/b4cd.../frames -F "file=@frame.jpg"
# {"frame_index": 0,
#  "probabilities": {"c1": 0.60, "c2": 0.55, "c3": 0.62},
#  "criteria": {"c1": true, "c2": true, "c3": true},
#  "cvs_achieved": true, "window_filled": 1, "window_size": 15, ...}

# Cerrar la sesión
curl -X DELETE localhost:8000/sessions/b4cd...
```

`cvs_achieved` es verdadero solo si se cumplen los tres criterios (umbral 0.5).
`window_filled` indica cuántos de los 15 segundos de contexto hay; durante los
primeros 14 frames el modelo ve menos historia.

| Endpoint | Respuesta |
|---|---|
| `GET /health` | estado, precisión cargada y sesiones activas |
| `POST /sessions` | `201` con `session_id`; `503` si se alcanzó el límite |
| `POST /sessions/{id}/frames` | predicción; `404` sesión inexistente o expirada, `400` imagen inválida, `413` más de 10 MB |
| `DELETE /sessions/{id}` | `204`; `404` si no existe |

### Configuración

| Variable | Por defecto | |
|---|---|---|
| `CVS_MODELS_DIR` | `models` (`/models` en Docker) | directorio con los `.onnx` |
| `CVS_PRECISION` | `fp32` | `fp32` o `int8` |
| `CVS_MAX_SESSIONS` | `32` | sesiones simultáneas |
| `CVS_SESSION_TTL_S` | `300` | segundos de inactividad antes de expirar una sesión |
| `CVS_NUM_THREADS` | `0` | hilos de ONNX Runtime; `0` = automático |

## Limitaciones

- **No detecta entradas fuera de dominio.** A cualquier imagen, incluso ruido,
  le asigna probabilidades; una imagen que no es cirugía puede dar
  `cvs_achieved: true`.
- **Supone 1 fps desde el inicio del video.** La posición temporal de cada
  frame es su número de orden, como en el entrenamiento. Saltarse frames o
  empezar a mitad del procedimiento le da al modelo un contexto distinto al
  que vio.
- **Sesiones de más de 34 minutos.** La tabla de posiciones aprendidas cubre
  2048 s; a partir de ahí la posición queda fija en 2047, un caso que el
  modelo no vio en entrenamiento.
- **Estado en memoria.** Las sesiones viven en el proceso: un solo worker por
  contenedor y, con varias réplicas, afinidad de sesión en el balanceador.
- **int8 validado en un subconjunto.** La comparación con fp32 usa 60 de los
  300 videos de test de
  [SAGES 2024](https://huggingface.co/datasets/CAMMA-public/SAGES_CVS_Challenge_2024);
  el test completo daría un intervalo más estrecho.

## Desarrollo

```bash
uv sync                                    # dependencias de desarrollo y torch
uv run pytest                              # tests (modelos pequeños, sin datos clínicos)
uv run ruff check . && uv run ruff format --check .
uv run python benchmarks/bench_latency.py --readme README.md   # latencia y tabla del README

# mAP de fp32 e int8 en SAGES 2024; descarga los videos fuera del repo
uv run --group eval python benchmarks/eval_map.py --data ../datasets/SAGES_2024 --n-videos 60
```

```
src/cvs_serve/
├── config.py        constantes del modelo y configuración del servicio
├── model.py         preprocesamiento, ventana temporal, carga de pesos, predictor ONNX
├── perceiver.py     Perceiver temporal (adaptado del repositorio original)
├── onnx_export.py   export, verificación de paridad y cuantización
└── api.py           endpoints FastAPI y sesiones
```

## Atribución y licencia

Basado en **PercEVA-CVS**: S. Cañar, J. S. Vera, I. S. Tovar, P. Arbeláez,
"Annotation-Efficient Critical View of Safety Assessment with Vision Foundation
Models", SafeSurg Workshop, MICCAI 2026.
[Código original](https://github.com/BCV-Uniandes/PercEVA-CVS) — CC BY-NC-SA 4.0.

Este repositorio hereda la licencia **CC BY-NC-SA 4.0**.
