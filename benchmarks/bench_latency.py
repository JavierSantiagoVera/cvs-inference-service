"""Mide la latencia de inferencia del servicio CVS: PyTorch fp32, ONNX fp32 y ONNX int8.

Compara, con la misma entrada sintética (batch=1, el caso real del servicio a 1
fps), el encoder solo, el Perceiver solo y el paso completo por frame (encoder +
TemporalWindow + Perceiver), que es lo que la API hace por petición.

Cada backend se mide en un proceso propio: con PyTorch cargado en el mismo
proceso, ONNX int8 salía un 37% más lento, y el servicio corre sin torch.

Uso:
    uv run python benchmarks/bench_latency.py
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cvs_serve.config import (
    CRITERIA,
    EMBED_DIM,
    IMG_SIZE,
    MODEL_FILES,
    PARITY_ATOL,
    PARITY_RTOL,
    WINDOW_SIZE,
)
from cvs_serve.model import TemporalWindow, make_onnx_session

logger = logging.getLogger(__name__)

SEED = 0
WARMUP = 15
ITERATIONS = 120
REPEATS = 2
BACKENDS = ("pytorch_fp32", "onnx_fp32", "onnx_int8")
STAGES = ("encoder", "perceiver", "full")


@dataclass
class Timings:
    """Latencias crudas de una etapa, en milisegundos."""

    samples_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, float]:
        arr = np.array(self.samples_ms)
        p50 = float(np.percentile(arr, 50))
        return {
            "p50_ms": round(p50, 3),
            "p95_ms": round(float(np.percentile(arr, 95)), 3),
            "mean_ms": round(float(arr.mean()), 3),
            "min_ms": round(float(arr.min()), 3),
            "max_ms": round(float(arr.max()), 3),
            "throughput_img_s": round(1000.0 / p50, 2),
            "n": len(arr),
        }


def _time_calls(fn: Callable[[], None], warmup: int, iterations: int) -> Timings:
    for _ in range(warmup):
        fn()
    timings = Timings()
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.samples_ms.append((time.perf_counter() - start) * 1000.0)
    return timings


def run_stage(
    name: str, fn: Callable[[], None], warmup: int, iterations: int, repeats: int
) -> dict:
    """Corre una etapa `repeats` veces y avisa si el p50 varía más del 10% entre corridas."""
    runs = []
    for r in range(repeats):
        summary = _time_calls(fn, warmup, iterations).summary()
        logger.info(
            "%s run %d/%d: p50=%.3fms p95=%.3fms n=%d",
            name,
            r + 1,
            repeats,
            summary["p50_ms"],
            summary["p95_ms"],
            summary["n"],
        )
        runs.append(summary)

    p50s = [r["p50_ms"] for r in runs]
    variation = (max(p50s) - min(p50s)) / min(p50s)
    stable = variation <= 0.10
    if not stable:
        logger.warning(
            "%s: el p50 varió %.1f%% entre corridas (%s); no se promedia",
            name,
            variation * 100,
            p50s,
        )
    return {
        "runs": runs,
        "p50_variation_pct": round(variation * 100, 2),
        "stable_across_runs": stable,
        # La última corrida es la representativa; las corridas no se promedian.
        "reported": runs[-1],
    }


def _window_inputs(rng: np.random.Generator) -> dict[str, np.ndarray]:
    return {
        "x": rng.standard_normal((1, WINDOW_SIZE, EMBED_DIM), dtype=np.float32),
        "key_index": np.full((1,), WINDOW_SIZE - 1, dtype=np.int64),
        "positions": np.arange(WINDOW_SIZE, dtype=np.int64)[np.newaxis],
    }


def pytorch_stages(weights: Path, threads: int) -> tuple[dict[str, Callable[[], None]], dict]:
    """Etapas con los modelos PyTorch originales."""
    import torch

    from cvs_serve.model import load_encoder, load_perceiver
    from cvs_serve.onnx_export import EncoderEmbedding

    torch.manual_seed(SEED)
    torch.set_num_threads(threads)
    encoder = EncoderEmbedding(load_encoder(weights / "eva02_encoder/best_eva02_enc_cvs.pt"))
    perceiver = load_perceiver(weights / "perceiver")
    rng = np.random.default_rng(SEED)
    pixels = torch.from_numpy(rng.standard_normal((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32))
    window_inputs = {k: torch.from_numpy(v) for k, v in _window_inputs(rng).items()}
    window = TemporalWindow()

    @torch.inference_mode()
    def full() -> None:
        inputs = window.push(encoder(pixels).numpy()[0])
        perceiver(
            torch.from_numpy(inputs.x),
            torch.from_numpy(inputs.key_index),
            torch.from_numpy(inputs.positions),
        )

    stages = {
        "encoder": torch.inference_mode()(lambda: encoder(pixels)),
        "perceiver": torch.inference_mode()(lambda: perceiver(**window_inputs)),
        "full": full,
    }
    n_bytes = sum(
        p.numel() * p.element_size() for m in (encoder, perceiver) for p in m.parameters()
    )
    info = {
        "size_mb": round(n_bytes / 2**20, 1),
        "size_note": "pesos en memoria (parámetros × bytes fp32), no tamaño en disco",
        "torch_version": torch.__version__,
    }
    return stages, info


def onnx_stages(
    models: Path, precision: str, threads: int
) -> tuple[dict[str, Callable[[], None]], dict]:
    """Etapas con las mismas sesiones de ONNX Runtime que usa el servicio."""
    import onnxruntime as ort

    enc_path, per_path = (models / name for name in MODEL_FILES[precision])
    encoder = make_onnx_session(enc_path, threads)
    perceiver = make_onnx_session(per_path, threads)
    rng = np.random.default_rng(SEED)
    pixels = rng.standard_normal((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    window_inputs = _window_inputs(rng)
    window = TemporalWindow()

    def full() -> None:
        inputs = window.push(encoder.run(None, {"pixels": pixels})[0][0])
        perceiver.run(
            None, {"x": inputs.x, "key_index": inputs.key_index, "positions": inputs.positions}
        )

    stages = {
        "encoder": lambda: encoder.run(None, {"pixels": pixels}),
        "perceiver": lambda: perceiver.run(None, window_inputs),
        "full": full,
    }
    size = enc_path.stat().st_size + per_path.stat().st_size
    info = {"size_mb": round(size / 2**20, 1), "onnxruntime_version": ort.__version__}
    return stages, info


def measure_backend(args: argparse.Namespace) -> dict:
    """Mide un solo backend. Se ejecuta en un subproceso propio."""
    if args.backend == "pytorch_fp32":
        stages, info = pytorch_stages(args.weights, args.threads)
    else:
        stages, info = onnx_stages(args.models, args.backend.removeprefix("onnx_"), args.threads)
    loaded = sorted(m for m in ("torch", "onnxruntime") if m in sys.modules)
    results = {
        stage: run_stage(
            f"{args.backend}/{stage}", stages[stage], args.warmup, args.iterations, args.repeats
        )
        for stage in STAGES
    }
    return {**info, "modules_in_process": loaded, "stages": results}


def check_outputs(args: argparse.Namespace) -> dict:
    """Paridad PyTorch vs ONNX fp32 y deriva int8 vs fp32, antes de medir latencias."""

    from cvs_serve.model import OnnxPredictor, load_encoder, load_perceiver
    from cvs_serve.onnx_export import EncoderEmbedding, check_parity

    rng = np.random.default_rng(SEED)
    encoder = EncoderEmbedding(load_encoder(args.weights / "eva02_encoder/best_eva02_enc_cvs.pt"))
    perceiver = load_perceiver(args.weights / "perceiver")
    enc_fp32, per_fp32 = (args.models / n for n in MODEL_FILES["fp32"])
    pixels = rng.standard_normal((1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    parity = {
        "encoder_max_abs_diff": check_parity(encoder, enc_fp32, {"pixels": pixels}),
        "perceiver_max_abs_diff": check_parity(perceiver, per_fp32, _window_inputs(rng)),
        "rtol": PARITY_RTOL,
        "atol": PARITY_ATOL,
    }
    del encoder, perceiver  # liberar ~1.2 GB antes de cargar los ONNX

    frames = rng.standard_normal((WINDOW_SIZE, 1, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    probs = {}
    for precision in ("fp32", "int8"):
        predictor = OnnxPredictor(*(args.models / n for n in MODEL_FILES[precision]))
        window = TemporalWindow()
        probs[precision] = np.stack(
            [predictor.probabilities(window.push(predictor.embed(f))) for f in frames]
        )
    drift = {
        "max_abs_prob_diff": round(float(np.abs(probs["fp32"] - probs["int8"]).max()), 4),
        "threshold_agreement": float(((probs["fp32"] >= 0.5) == (probs["int8"] >= 0.5)).mean()),
        "n_frames": WINDOW_SIZE,
        "note": "entradas sintéticas, sin set de validación clínico",
    }
    logger.info("Paridad %s; deriva int8 %s", parity, drift)
    return {"pytorch_onnx_parity": parity, "int8_vs_fp32_pipeline": drift}


def _cpu_name() -> str:
    if sys.platform == "win32":
        import winreg

        key = winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
        )
        return winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", type=Path, default=Path("perceva_cvs_weights/sages2024"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=WARMUP)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--readme",
        type=Path,
        nargs="*",
        help="README a actualizar; sin rutas, README.md y README.es.md",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="no mide; solo regenera el README desde --results",
    )
    parser.add_argument("--backend", choices=BACKENDS, help="uso interno: mide un solo backend")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s", stream=sys.stderr)

    if args.backend:
        sys.stdout.write(json.dumps(measure_backend(args)))
        return
    readmes = args.readme or (DEFAULT_READMES if args.readme is not None else ())
    if args.render_only:
        for readme in readmes or DEFAULT_READMES:
            update_readme(readme, _read_results(args.results))
        return

    latency = check_outputs(args)
    passthrough = [
        f"--{k}={getattr(args, k)}"
        for k in ("weights", "models", "threads", "warmup", "iterations", "repeats")
    ]
    backends = {}
    for backend in BACKENDS:
        logger.info("--- %s (proceso propio) ---", backend)
        out = subprocess.run(
            [sys.executable, __file__, f"--backend={backend}", *passthrough],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        )
        backends[backend] = json.loads(out.stdout)

    latency["environment"] = {
        "os": platform.platform(),
        "python": platform.python_version(),
        "cpu": _cpu_name(),
        "gpu": "ninguna (solo CPU)",
        "torch": backends["pytorch_fp32"].pop("torch_version"),
        "onnxruntime": backends["onnx_fp32"]["onnxruntime_version"],
        "onnxruntime_provider": "CPUExecutionProvider",
        "onnxruntime_allow_spinning": False,
        "threads": args.threads,
        "seed": SEED,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "repeats_per_stage": args.repeats,
        "process_isolation": "un proceso por backend",
    }
    latency["backends"] = backends

    results = _read_results(args.results)
    results["latency"] = latency
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", "utf-8")
    logger.info("Resultados en %s", args.results)
    for readme in readmes:
        update_readme(readme, results)


def _read_results(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


DEFAULT_READMES = (Path("README.md"), Path("README.es.md"))
README_START = "<!-- bench:start"
README_END = "<!-- bench:end -->"
BACKEND_ROWS = (
    ("pytorch_fp32", "PyTorch", "fp32"),
    ("onnx_fp32", "ONNX Runtime", "fp32"),
    ("onnx_int8", "ONNX Runtime", "int8"),
)
TEXT = {
    "en": {
        "header": "| Backend | Precision | p50 (ms) | p95 (ms) | Throughput | Size | Δ accuracy |",
        "dynamic": "dynamic",
        "in_memory": "weights in memory",
        "reference": "reference",
        "parity": "parity with PyTorch: max diff {x}",
        "int8_measured": "{d} mAP pts (95% CI: {lo} to {hi})",
        "int8_unmeasured": "**mAP not measured** (no validation set)",
        "stable": "p50 variation < {v}%",
        "unstable": "**p50 varied up to {v}% across runs**",
        "no_gpu": "no GPU",
        "method": (
            "Full per-frame step at batch 1 (EVA-02 Large encoder at 448×448, temporal window "
            "and Perceiver), which is what the API does on every request; the encoder is over "
            "{share}% of the time. Throughput = 1000 / p50, frames processed sequentially. CPU "
            "{cpu}, {gpu}, {os}, {threads} threads for every backend; PyTorch {torch}, ONNX "
            "Runtime {ort} (thread spinning disabled, as in the service), Python {python}. "
            "{iterations} iterations after {warmup} warm-up, {repeats} runs per stage "
            "({stability}), each backend in its own process. Script: "
            "`benchmarks/bench_latency.py`; raw data in `benchmarks/results.json`."
        ),
        "int8_eval": (
            "For int8: across {n_videos} of the {of} SAGES 2024 test videos (randomly chosen with "
            "seed {seed}; {keyframes} keyframes, majority-vote labels), macro mAP goes from "
            "{fp32} with fp32 to {int8} with int8 ({d} points; 95% video-bootstrap CI: {lo} to "
            "{hi}). The decision at the 0.5 threshold agrees in {agree}% of cases. It is a "
            "subset of the test split, so its absolute mAP is not comparable with the paper's; "
            "what it measures is the difference between precisions on the same frames. "
            "Script: `benchmarks/eval_map.py`."
        ),
        "int8_synthetic": (
            "For int8: on {n} synthetic frames, the probability differs from fp32 by at most "
            "{x} and the per-criterion decision agrees in {agree} of {total} cases. That does "
            "not replace measuring mAP on SAGES; until then the service defaults to fp32."
        ),
    },
    "es": {
        "header": (
            "| Backend | Precisión | p50 (ms) | p95 (ms) | Throughput | Tamaño | Δ exactitud |"
        ),
        "dynamic": "dinámica",
        "in_memory": "pesos en memoria",
        "reference": "referencia",
        "parity": "paridad con PyTorch: dif. máx. {x}",
        "int8_measured": "{d} pts de mAP (IC 95%: {lo} a {hi})",
        "int8_unmeasured": "**mAP no medida** (sin set de validación)",
        "stable": "variación de p50 < {v}%",
        "unstable": "**variación de p50 de hasta {v}% entre corridas**",
        "no_gpu": "sin GPU",
        "method": (
            "Paso completo por frame con batch 1 (encoder EVA-02 Large a 448×448, ventana "
            "temporal y Perceiver), que es lo que hace la API en cada petición; el encoder es "
            "más del {share}% del tiempo. Throughput = 1000 / p50, frames procesados en serie. "
            "CPU {cpu}, {gpu}, {os}, {threads} hilos en todos los backends; PyTorch {torch}, "
            "ONNX Runtime {ort} (spinning de hilos desactivado, como en el servicio), Python "
            "{python}. {iterations} iteraciones tras {warmup} de calentamiento, {repeats} "
            "corridas por etapa ({stability}), cada backend en un proceso propio. Script: "
            "`benchmarks/bench_latency.py`; datos crudos en `benchmarks/results.json`."
        ),
        "int8_eval": (
            "Sobre int8: en {n_videos} de los {of} videos de test de SAGES 2024 (elegidos al "
            "azar con semilla {seed}; {keyframes} fotogramas clave, etiqueta por voto "
            "mayoritario), la macro mAP pasa de {fp32} en fp32 a {int8} en int8 ({d} puntos; "
            "IC 95% por bootstrap de videos: {lo} a {hi}). La decisión con umbral 0.5 coincide "
            "en el {agree}% de los casos. Es un subconjunto del test, así que su mAP absoluta "
            "no es comparable con la del paper; lo que mide es la diferencia entre precisiones "
            "sobre los mismos frames. Script: `benchmarks/eval_map.py`."
        ),
        "int8_synthetic": (
            "Sobre int8: con {n} frames sintéticos, la probabilidad difiere de fp32 como máximo "
            "en {x} y la decisión por criterio coincide en {agree} de {total} casos. Eso no "
            "reemplaza medir la mAP en SAGES; hasta entonces el servicio usa fp32 por defecto."
        ),
    },
}


def readme_language(readme: Path) -> str:
    """README.es.md → "es"; cualquier otro README → "en"."""
    return "es" if readme.name.endswith(".es.md") else "en"


def render_latency_section(latency: dict, int8_eval: dict | None = None, lang: str = "en") -> str:
    """Tabla y notas de latencia del README, solo con cifras de results.json."""
    text = TEXT[lang]
    env, backends = latency["environment"], latency["backends"]
    parity = latency["pytorch_onnx_parity"]
    drift = latency["int8_vs_fp32_pipeline"]
    max_parity = max(parity["encoder_max_abs_diff"], parity["perceiver_max_abs_diff"])
    if int8_eval:
        # drop = mAP(fp32) - mAP(int8); se muestra el cambio de int8 respecto a fp32.
        change = -int8_eval["int8_drop_points"]
        low, high = (-v for v in reversed(int8_eval["int8_drop_ci95_points"]))
        int8_accuracy = text["int8_measured"].format(
            d=f"{change:+.1f}", lo=f"{low:+.1f}", hi=f"{high:+.1f}"
        )
    else:
        int8_accuracy = text["int8_unmeasured"]
    accuracy = {
        "pytorch_fp32": text["reference"],
        "onnx_fp32": text["parity"].format(x=f"{max_parity:.1e}".replace("e-0", "e-")),
        "onnx_int8": int8_accuracy,
    }
    lines = [text["header"], "|---|---|---|---|---|---|---|"]
    for key, name, precision in BACKEND_ROWS:
        backend = backends[key]
        full = backend["stages"]["full"]["reported"]
        if key == "onnx_int8":
            precision = f"{precision} ({text['dynamic']})"
        size = f"{backend['size_mb']} MB"
        if key.startswith("pytorch"):
            size += f" ({text['in_memory']})"
        lines.append(
            f"| {name} | {precision} | {full['p50_ms']:.1f} | {full['p95_ms']:.1f} "
            f"| {full['throughput_img_s']:.2f} frames/s | {size} | {accuracy[key]} |"
        )

    stages = [s for b in backends.values() for s in b["stages"].values()]
    max_variation = max(s["p50_variation_pct"] for s in stages)
    if all(s["stable_across_runs"] for s in stages):
        stability = text["stable"].format(v=max(1, math.ceil(max_variation)))
    else:
        stability = text["unstable"].format(v=f"{max_variation:.0f}")
    encoder_share = min(
        b["stages"]["encoder"]["reported"]["p50_ms"] / b["stages"]["full"]["reported"]["p50_ms"]
        for b in backends.values()
    )
    method = text["method"].format(
        share=math.floor(encoder_share * 20) * 5,
        cpu=env["cpu"],
        gpu=text["no_gpu"] if env["gpu"].startswith("ninguna") else env["gpu"],
        os=env["os"],
        threads=env["threads"],
        torch=env["torch"],
        ort=env["onnxruntime"],
        python=env["python"],
        iterations=env["measured_iterations"],
        warmup=env["warmup_iterations"],
        repeats=env["repeats_per_stage"],
        stability=stability,
    )
    if int8_eval:
        subset, maps = int8_eval["subset"], int8_eval["map"]
        int8 = text["int8_eval"].format(
            n_videos=subset["n_videos"],
            of=subset["of"],
            seed=subset["seed"],
            keyframes=int8_eval["n_keyframes"],
            fp32=f"{maps['fp32']['macro']:.1f}",
            int8=f"{maps['int8']['macro']:.1f}",
            d=f"{change:+.1f}",
            lo=f"{low:+.1f}",
            hi=f"{high:+.1f}",
            agree=f"{int8_eval['threshold_agreement'] * 100:.1f}",
        )
    else:
        total = drift["n_frames"] * len(CRITERIA)
        int8 = text["int8_synthetic"].format(
            n=drift["n_frames"],
            x=f"{drift['max_abs_prob_diff']:.3f}",
            agree=round(drift["threshold_agreement"] * total),
            total=total,
        )
    wrap = textwrap.TextWrapper(width=79, break_on_hyphens=False)
    return "\n".join([*lines, "", wrap.fill(method), "", wrap.fill(int8)])


def update_readme(readme: Path, results: dict) -> None:
    """Reemplaza lo que hay entre la línea <!-- bench:start ... --> y <!-- bench:end -->."""
    text = readme.read_text(encoding="utf-8")
    start, end = text.find(README_START), text.find(README_END)
    if start < 0 or end < start:
        raise ValueError(f"{readme} no tiene los marcadores {README_START!r} y {README_END!r}")
    marker_end = text.index("\n", start)
    body = render_latency_section(
        results["latency"], results.get("int8_eval"), readme_language(readme)
    )
    new = text[: marker_end + 1] + body + "\n" + text[end:]
    readme.write_text(new, encoding="utf-8")
    logger.info("Sección de latencia actualizada en %s", readme)


if __name__ == "__main__":
    main()
