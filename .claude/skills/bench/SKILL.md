---
name: bench
description: Mide la latencia de PyTorch, ONNX fp32 y ONNX int8 y regenera la tabla de latencia del README desde benchmarks/results.json. Úsalo cuando cambie el modelo, el export, la configuración de ONNX Runtime o haya que actualizar las cifras de rendimiento.
when_to_use: El usuario pide medir latencia, correr el benchmark o actualizar la tabla de rendimiento del README
argument-hint: [--render-only]
allowed-tools: Read, Bash(uv run *), Bash(git diff *)
---

## Estado actual

Modelos: !`ls models/*.onnx 2>/dev/null || echo "ninguno: correr primero /export-onnx"`
Última medición: !`uv run python -c "import json; e=json.load(open('benchmarks/results.json', encoding='utf-8'))['latency']['environment']; print(e['cpu'], '|', e['onnxruntime'])" 2>/dev/null || echo "sin mediciones"`

## Procedimiento

1. **Máquina en reposo.** No correr tests, builds ni otros benchmarks en
   paralelo: el encoder usa todos los hilos y cualquier carga altera el p50.

2. **Medir.** Tarda ~15 minutos en CPU:
   ```bash
   uv run python benchmarks/bench_latency.py --readme README.md
   ```
   El script verifica primero la paridad PyTorch vs ONNX fp32 y se detiene si
   falla. Cada backend corre en un proceso propio (con torch cargado, int8 sale
   ~37% más lento) y ONNX Runtime usa `make_onnx_session`, la misma
   configuración del servicio.

   Con `--render-only` no mide: solo regenera el README desde el JSON actual.

3. **Revisar antes de aceptar las cifras.**
   - Si el log muestra `WARNING ... varió X% entre corridas`, no publicar: repetir
     con la máquina en reposo. El README lo marca en negrita si queda así.
   - El paso `full` debe ser ≈ `encoder` + `perceiver`. Si sobra tiempo, hay
     contención de hilos: investigar antes de publicar.
   - `git diff README.md benchmarks/results.json`: solo debe cambiar lo que hay
     entre `<!-- bench:start` y `<!-- bench:end -->`.

4. **Regenerar la gráfica** desde el mismo JSON:
   ```bash
   uv run python scripts/make_latency_chart.py
   ```
   Si cambió la latencia de forma notable, regenerar también el demo
   (`uv run python scripts/make_demo.py`) para que el GIF no contradiga la tabla.

5. **Nunca editar a mano** la tabla ni las notas entre los marcadores.

## Prohibido

- Publicar la caída de exactitud de int8 como medida mientras
  `int8_map_drop` sea `null` en `results.json`.
- Comparar cifras de máquinas distintas en la misma tabla.

$ARGUMENTS
