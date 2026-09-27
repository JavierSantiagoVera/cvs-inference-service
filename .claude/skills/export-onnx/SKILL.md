---
name: export-onnx
description: Exporta un modelo PyTorch a ONNX y verifica la paridad numérica con el modelo original. Úsalo cuando haya que exportar, cuantizar o validar un modelo ONNX.
when_to_use: El usuario pide exportar a ONNX, cuantizar un modelo, o verificar que el modelo exportado da los mismos resultados
argument-hint: [ruta-del-checkpoint]
allowed-tools: Read, Write, Edit, Bash(uv run *), Bash(python *)
---

## Estado actual

Modelos exportados: !`ls -la models/*.onnx 2>/dev/null || echo "ninguno"`

## Procedimiento

1. **Exportar** con `torch.onnx.export`, con ejes dinámicos para el batch y
   `opset_version=17` como mínimo.

2. **Verificar paridad antes de cualquier otra cosa.** Corre el mismo tensor de
   entrada por PyTorch y por ONNX Runtime y compara:
   ```python
   np.testing.assert_allclose(torch_out, onnx_out, rtol=1e-3, atol=1e-5)
   ```
   Si falla, **detente y reporta**. No sigas a cuantización con un export roto.

3. **Cuantizar** solo después de que la paridad pase. Usa
   `onnxruntime.quantization`. Para INT8 estático hace falta un dataset de
   calibración representativo; si no lo hay, usa dinámica y dilo.

4. **Medir la caída de exactitud** sobre el set de validación, no solo la
   latencia. Una cuantización que acelera 3x pero pierde 5 puntos de mAP no
   sirve, y hay que decirlo.

5. **Guardar** el `.onnx` en `models/` (que está en `.gitignore`) y registrar
   las cifras en `benchmarks/results.json`.

## Después

Invoca al subagente `perf-bencher` para la tabla comparativa de latencia.

$ARGUMENTS
