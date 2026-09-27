"""Mide la macro mAP de ONNX fp32 e int8 en SAGES 2024 (test) y la caída por cuantizar.

Replica el protocolo de PercEVA-CVS: frames a 1 fps extraídos como la herramienta
oficial de SAGES (intervalo int(fps), JPEG calidad 95), ventana causal de 15 s,
evaluación en los fotogramas clave etiquetados (cada 150 frames de origen) con
etiqueta por voto mayoritario de los tres evaluadores, y AP por criterio.

Los datos se guardan fuera del repositorio (--data). Uso:
    uv run --group eval python benchmarks/eval_map.py --data ../datasets/SAGES_2024 --n-videos 60
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score

from cvs_serve.config import CRITERIA, MODEL_FILES
from cvs_serve.model import OnnxPredictor, TemporalWindow, preprocess

logger = logging.getLogger(__name__)

HF_BASE = "https://huggingface.co/datasets/CAMMA-public/SAGES_CVS_Challenge_2024/resolve/main"
KEYFRAME_INTERVAL = 150  # etiquetas cada 5 s a 30 fps
JPEG_QUALITY = 95  # --jpeg-quality por defecto de tools/preprocess_videos.py
PRECISIONS = ("fp32", "int8")


def _download(url: str, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
        while chunk := response.read(2**20):
            out.write(chunk)
    tmp.replace(dest)


def select_videos(data: Path, n_videos: int, seed: int) -> list[str]:
    """Elige n videos del test con semilla fija, a partir de metadata.csv."""
    metadata = data / "test/labels/metadata.csv"
    _download(f"{HF_BASE}/test/labels/metadata.csv", metadata)
    with metadata.open(encoding="utf-8") as f:
        names = sorted(row["video_name"] for row in csv.DictReader(f))
    if n_videos >= len(names):
        return names
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(names, size=n_videos, replace=False).tolist())


def load_keyframe_labels(path: Path) -> dict[int, list[int]]:
    """Etiqueta por voto mayoritario (label_type='hard' en la evaluación original)."""
    labels = {}
    with path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            votes = [[int(row[f"{c}_rater{r}"]) for r in (1, 2, 3)] for c in CRITERIA]
            labels[int(row["frame_id"])] = [int(sum(v) >= 2) for v in votes]
    return labels


def iter_frames_1fps(video: Path) -> tuple[float, list[tuple[int, Image.Image]]]:
    """Frames a 1 fps con su id de origen, pasando por JPEG q95 como la herramienta oficial."""
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise OSError(f"No se pudo abrir {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    interval = int(fps)
    if interval <= 0:
        raise ValueError(f"FPS inválido ({fps}) en {video}")
    frames = []
    index = 0
    while True:
        ok, bgr = capture.read()
        if not ok:
            break
        if index % interval == 0:
            ok, jpeg = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if not ok:
                raise RuntimeError(f"Fallo al codificar el frame {index} de {video}")
            frames.append((index, Image.open(io.BytesIO(jpeg.tobytes())).convert("RGB")))
        index += 1
    capture.release()
    return fps, frames


def evaluate_video(name: str, data: Path, predictors: dict[str, OnnxPredictor]) -> list[dict]:
    """Probabilidades de cada precisión en los fotogramas clave etiquetados de un video."""
    labels = load_keyframe_labels(data / f"test/labels/{name}/frame.csv")
    fps, frames = iter_frames_1fps(data / f"test/videos/{name}.mp4")
    windows = {p: TemporalWindow() for p in predictors}
    records = []
    for frame_id, image in frames:
        pixels = preprocess(image)
        is_key = frame_id % KEYFRAME_INTERVAL == 0 and frame_id in labels
        probs = {}
        for precision, predictor in predictors.items():
            inputs = windows[precision].push(predictor.embed(pixels))
            if is_key:
                probs[precision] = predictor.probabilities(inputs).tolist()
        if is_key:
            records.append(
                {"video": name, "frame_id": frame_id, "fps": fps, "label": labels[frame_id]}
                | {f"probs_{p}": v for p, v in probs.items()}
            )
    missing = sorted(set(labels) - {r["frame_id"] for r in records})
    if missing:
        logger.warning("%s: %d fotogramas etiquetados sin frame extraído", name, len(missing))
    return records


def macro_map(labels: np.ndarray, probs: np.ndarray) -> tuple[float, list[float]]:
    """AP por criterio y su media; NaN si un criterio no tiene positivos."""
    per_class = [
        float(average_precision_score(labels[:, c], probs[:, c])) if labels[:, c].any() else np.nan
        for c in range(labels.shape[1])
    ]
    return float(np.nanmean(per_class)), per_class


def bootstrap_drop(records: list[dict], n: int, seed: int) -> list[float]:
    """IC 95% de mAP(fp32) - mAP(int8), remuestreando videos con reemplazo."""
    rng = np.random.default_rng(seed)
    by_video: dict[str, list[dict]] = {}
    for r in records:
        by_video.setdefault(r["video"], []).append(r)
    videos = sorted(by_video)
    drops = []
    for _ in range(n):
        sample = [r for v in rng.choice(videos, size=len(videos)) for r in by_video[v]]
        labels = np.array([r["label"] for r in sample])
        if not labels.any(axis=0).all():
            continue
        fp32 = macro_map(labels, np.array([r["probs_fp32"] for r in sample]))[0]
        int8 = macro_map(labels, np.array([r["probs_int8"] for r in sample]))[0]
        drops.append(fp32 - int8)
    if not drops:
        logger.warning("Bootstrap sin muestras válidas: algún criterio sin positivos")
        return [float("nan"), float("nan")]
    return [float(np.percentile(drops, 2.5)), float(np.percentile(drops, 97.5))]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True, help="directorio fuera del repo")
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--n-videos", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap", type=int, default=2000)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results.json"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.data.resolve().is_relative_to(Path.cwd().resolve()):
        raise SystemExit("--data debe estar fuera del repositorio: son videos quirúrgicos")

    videos = select_videos(args.data, args.n_videos, args.seed)
    logger.info("%d videos seleccionados (semilla %d)", len(videos), args.seed)
    predictors = {
        p: OnnxPredictor(*(args.models / f for f in MODEL_FILES[p]), num_threads=args.threads)
        for p in PRECISIONS
    }

    # Progreso por video en un .jsonl fuera del repo: se puede reanudar.
    progress = args.data / f"eval_progress_n{args.n_videos}_s{args.seed}.jsonl"
    done = {}
    if progress.exists():
        for line in progress.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            done[entry["video"]] = entry["records"]
    start = time.monotonic()
    for i, name in enumerate(videos, 1):
        if name in done:
            continue
        _download(
            f"{HF_BASE}/test/labels/{name}/frame.csv", args.data / f"test/labels/{name}/frame.csv"
        )
        _download(f"{HF_BASE}/test/videos/{name}.mp4", args.data / f"test/videos/{name}.mp4")
        records = evaluate_video(name, args.data, predictors)
        done[name] = records
        with progress.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"video": name, "records": records}) + "\n")
        logger.info(
            "[%d/%d] %s: %d fotogramas clave (%.0f min)",
            i,
            len(videos),
            name,
            len(records),
            (time.monotonic() - start) / 60,
        )

    records = [r for name in videos for r in done[name]]
    labels = np.array([r["label"] for r in records])
    maps = {p: macro_map(labels, np.array([r[f"probs_{p}"] for r in records])) for p in PRECISIONS}
    drop = maps["fp32"][0] - maps["int8"][0]
    ci = bootstrap_drop(records, args.bootstrap, args.seed)
    agreement = float(
        np.mean(
            (np.array([r["probs_fp32"] for r in records]) >= 0.5)
            == (np.array([r["probs_int8"] for r in records]) >= 0.5)
        )
    )
    fps_values = sorted({r["fps"] for r in records})
    summary = {
        "dataset": "SAGES 2024 CVS Challenge, split test",
        "subset": {"n_videos": len(videos), "of": 300, "seed": args.seed},
        "n_keyframes": len(records),
        "positives_per_criterion": dict(zip(CRITERIA, labels.sum(axis=0).tolist(), strict=True)),
        "source_fps": fps_values,
        "label": "voto mayoritario de 3 evaluadores (como la evaluación original)",
        "map": {
            p: {
                "macro": round(m[0] * 100, 2),
                **{c: round(v * 100, 2) for c, v in zip(CRITERIA, m[1], strict=True)},
            }
            for p, m in maps.items()
        },
        "int8_drop_points": round(drop * 100, 2),
        "int8_drop_ci95_points": [round(v * 100, 2) for v in ci],
        "threshold_agreement": round(agreement, 4),
    }
    logger.info("Resultado: %s", json.dumps(summary, ensure_ascii=False))

    results = json.loads(args.results.read_text(encoding="utf-8")) if args.results.exists() else {}
    results["int8_eval"] = summary
    results.setdefault("export", {})["int8_map_drop"] = summary["int8_drop_points"]
    args.results.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", "utf-8")
    logger.info("Resultados en %s", args.results)


if __name__ == "__main__":
    sys.exit(main())
