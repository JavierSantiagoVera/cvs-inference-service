"""Genera docs/demo.gif: una sesión real contra la API, con frames sintéticos.

Levanta el servicio con los modelos de models/, ejecuta curl de verdad y dibuja
la terminal con las mismas condiciones que benchmarks/bench_latency.py (int8,
16 hilos). Los comandos mostrados son los ejecutados; cada respuesta se
resume en una línea (código, latencia medida por curl, ventana y criterios).
Los frames son degradados sintéticos: no hay imágenes quirúrgicas.

Uso:
    uv run python scripts/make_demo.py
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

PORT = 8000
# 127.0.0.1 y no localhost: en Windows, curl intenta primero ::1 y pierde ~200 ms
# por conexión, porque uvicorn solo escucha en IPv4.
BASE = f"http://127.0.0.1:{PORT}"
N_FRAMES = 5
FONT_CANDIDATES = (
    "C:/Windows/Fonts/consola.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/System/Library/Fonts/Menlo.ttc",
)
COLORS = {
    "bg": (26, 26, 25),
    "text": (235, 235, 230),
    "muted": (137, 135, 129),
    "prompt": (57, 135, 229),
    "ok": (27, 175, 122),
    "accent": (217, 89, 38),
}
WIDTH, LINE_H, PAD = 980, 24, 22


def _font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_CANDIDATES:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def _synthetic_frames(folder: Path) -> list[Path]:
    """Degradados grises de 854×480 con brillo distinto por frame."""
    paths = []
    ramp = np.linspace(0, 1, 854, dtype=np.float32)[np.newaxis, :, np.newaxis]
    for i in range(N_FRAMES):
        level = 60 + 25 * i
        pixels = (level + 80 * ramp) * np.ones((480, 1, 3), dtype=np.float32)
        path = folder / f"frame_{i:03d}.jpg"
        Image.fromarray(pixels.clip(0, 255).astype(np.uint8)).save(path, quality=90)
        paths.append(path)
    return paths


def _curl(args: list[str], cwd: Path) -> tuple[int, float, str]:
    out = subprocess.run(
        ["curl", "-s", "-w", "\n%{http_code} %{time_total}", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    body, _, status = out.rpartition("\n")
    code, seconds = status.split()
    return int(code), float(seconds) * 1000, body


def _wait_for_health(timeout_s: float = 120) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError("La API no respondió a /health")


def record_session(workdir: Path) -> list[tuple[str, list[tuple[str, str]]]]:
    """Ejecuta la sesión y devuelve pasos (comando, líneas de salida con su color)."""
    frames = _synthetic_frames(workdir)
    steps = []

    code, ms, body = _curl(["-X", "POST", f"{BASE}/sessions"], workdir)
    session_id = json.loads(body)["session_id"]
    steps.append(
        (
            f"curl -X POST {BASE}/sessions",
            [(f"← {code} · session_id {session_id[:8]}… · ventana de 15 s a 1 fps", "ok")],
        )
    )
    for frame in frames:
        code, ms, body = _curl(
            ["-X", "POST", f"{BASE}/sessions/{session_id}/frames", "-F", f"file=@{frame.name}"],
            workdir,
        )
        r = json.loads(body)
        probs = "  ".join(f"{k} {v:.2f}" for k, v in r["probabilities"].items())
        steps.append(
            (
                f"curl -X POST {BASE}/sessions/$SID/frames -F file=@{frame.name}",
                [
                    (
                        f"← {code} · {ms:.0f} ms · frame {r['frame_index']} · "
                        f"ventana {r['window_filled']}/{r['window_size']}",
                        "ok",
                    ),
                    (f"  {probs}  ·  cvs_achieved {str(r['cvs_achieved']).lower()}", "muted"),
                ],
            )
        )
    code, ms, _ = _curl(["-X", "DELETE", f"{BASE}/sessions/{session_id}"], workdir)
    steps.append((f"curl -X DELETE {BASE}/sessions/$SID", [(f"← {code} · sesión cerrada", "ok")]))
    return steps


def render_gif(steps: list[tuple[str, list[tuple[str, str]]]], out: Path) -> None:
    font, small = _font(16), _font(14)
    header = [
        ("# CVS Inference Service · una sesión de video, un frame por segundo", "muted"),
        ("# frames sintéticos: muestra el flujo de la API, no predicciones clínicas", "muted"),
        ("", "text"),
    ]
    total_lines = len(header) + sum(1 + len(out_lines) for _, out_lines in steps)
    height = PAD * 2 + LINE_H * total_lines

    def draw(lines: list[tuple[str, str]]) -> Image.Image:
        image = Image.new("RGB", (WIDTH, height), COLORS["bg"])
        canvas = ImageDraw.Draw(image)
        for i, (text, color) in enumerate(lines):
            y = PAD + i * LINE_H
            if text.startswith("$ "):
                canvas.text((PAD, y), "$", font=font, fill=COLORS["prompt"])
                canvas.text((PAD + 18, y), text[2:], font=font, fill=COLORS["text"])
            else:
                canvas.text(
                    (PAD, y), text, font=small if color == "muted" else font, fill=COLORS[color]
                )
        return image

    lines = list(header)
    frames, durations = [draw(lines)], [900]
    for command, out_lines in steps:
        lines.append((f"$ {command}", "text"))
        frames.append(draw(lines))
        durations.append(700)
        lines.extend(out_lines)
        frames.append(draw(lines))
        durations.append(900)
    durations[-1] = 4000
    out.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True
    )
    logger.info("GIF en %s (%d fotogramas, %.0f KB)", out, len(frames), out.stat().st_size / 1024)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--precision", default="int8")
    parser.add_argument("--threads", type=int, default=16, help="igual que el benchmark")
    parser.add_argument("--out", type=Path, default=Path("docs/demo.gif"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    env = {
        **os.environ,
        "CVS_MODELS_DIR": str(args.models.resolve()),
        "CVS_PRECISION": args.precision,
        "CVS_NUM_THREADS": str(args.threads),
    }
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "cvs_serve.api:app", "--port", str(PORT)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        _wait_for_health()
        with tempfile.TemporaryDirectory() as tmp:
            steps = record_session(Path(tmp))
    finally:
        server.terminate()
        server.wait(timeout=30)
    render_gif(steps, args.out)


if __name__ == "__main__":
    main()
