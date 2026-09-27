# Cómo usar este andamiaje

## 1. Crear el repo

```bash
gh repo create cvs-inference-service --private --clone
cd cvs-inference-service
# copiar aquí el contenido de este andamiaje
git add . && git commit -m "Add project scaffold with Claude Code configuration"
```

Empieza **privado**. Lo vuelves público cuando `/ship-check` diga LISTO.

## 2. Abrir Claude Code

```bash
claude
```

Verifica que cargó todo:
- `/context` → debe listar `CLAUDE.md` bajo *Memory files*
- `/agents` → debe mostrar `release-auditor` y `perf-bencher`
- Escribe `/` → deben aparecer `/export-onnx` y `/ship-check`

## 3. Qué hay aquí y por qué

| Archivo | Para qué sirve |
|---|---|
| `CLAUDE.md` | Contexto permanente: comandos, estructura, reglas. Se carga en cada sesión. |
| `.claude/rules/python-style.md` | Reglas que solo cargan cuando se tocan archivos `.py`. |
| `.claude/agents/release-auditor.md` | Subagente: audita antes de publicar. |
| `.claude/agents/perf-bencher.md` | Subagente: mide latencia de forma reproducible. |
| `.claude/skills/export-onnx/` | Skill: procedimiento de export y verificación ONNX. |
| `.claude/skills/ship-check/` | Skill: checklist previo a publicar. |
| `.claude/settings.json` | Permisos preaprobados para no interrumpir el flujo. |

## 4. Flujo típico

```
> /init                          # si quieres que Claude enriquezca CLAUDE.md
> implementa el export a ONNX    # trabajo normal
> /export-onnx models/perceva.pt # usa el procedimiento definido
> usa perf-bencher para la tabla de latencia
> /ship-check                    # antes de publicar
```

## 5. Regla de decisión: ¿dónde va cada cosa?

- **Un hecho que aplica siempre** (comando de build, convención, prohibición)
  → `CLAUDE.md`.
- **Un hecho que solo aplica a ciertos archivos** → `.claude/rules/` con `paths`.
- **Un procedimiento de varios pasos que se repite** → una skill.
- **Una tarea que requiere leer mucho y devolver poco**, o que debe correr
  aislada → un subagente.

Si dudas, empieza en `CLAUDE.md`. Cuando crezca más de 200 líneas, mueve lo
específico a rules y lo procedimental a skills.
