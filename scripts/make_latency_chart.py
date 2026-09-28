"""Gráfica de latencia por frame (p50 y p95) para el README, desde benchmarks/results.json.

Genera docs/latency-{light,dark}.svg (inglés) y docs/latency-{light,dark}.es.svg
(español). Paleta de referencia validada para ambos modos (azul p50, naranja p95).

Uso:
    uv run python scripts/make_latency_chart.py
"""

from __future__ import annotations

import argparse
import json
import logging
from html import escape
from pathlib import Path

logger = logging.getLogger(__name__)

ROWS = (
    ("pytorch_fp32", "PyTorch fp32"),
    ("onnx_fp32", "ONNX fp32"),
    ("onnx_int8", "ONNX int8"),
)
THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "secondary": "#52514e",
        "muted": "#898781",
        "grid": "#e1e0d9",
        "axis": "#c3c2b7",
        "p50": "#2a78d6",
        "p95": "#eb6834",
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "secondary": "#c3c2b7",
        "muted": "#898781",
        "grid": "#2c2c2a",
        "axis": "#383835",
        "p50": "#3987e5",
        "p95": "#d95926",
    },
}
# Inglés con el nombre de archivo base; español con sufijo .es (latency-light.es.svg).
LANGUAGES = {
    "en": {
        "suffix": "",
        "title": "Latency per frame",
        "subtitle": "Full step, batch 1, CPU {cpu} · lower is better",
        "aria": "Per-frame latency of PyTorch fp32, ONNX fp32 and ONNX int8",
    },
    "es": {
        "suffix": ".es",
        "title": "Latencia por frame",
        "subtitle": "Paso completo, batch 1, CPU {cpu} · menos es mejor",
        "aria": "Latencia por frame de PyTorch fp32, ONNX fp32 y ONNX int8",
    },
}
WIDTH, LEFT, RIGHT = 720, 132, 150
TOP, ROW_H, BAR_H = 92, 44, 22
FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"


def _bar(x0: float, x1: float, y: float, h: float, r: float = 4) -> str:
    """Barra horizontal: cuadrada en la base, extremo de datos redondeado 4px."""
    r = min(r, (x1 - x0) / 2, h / 2)
    return (
        f"M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} "
        f"V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z"
    )


def render(latency: dict, theme: dict[str, str], text: dict[str, str]) -> str:
    backends = latency["backends"]
    stats = [(label, backends[key]["stages"]["full"]["reported"]) for key, label in ROWS]
    domain = 250 * -(-max(s["p95_ms"] for _, s in stats) // 250)  # redondeo hacia arriba
    plot_w = WIDTH - LEFT - RIGHT
    height = TOP + ROW_H * len(stats) + 40

    def sx(ms: float) -> float:
        return LEFT + ms / domain * plot_w

    cpu = latency["environment"]["cpu"].replace(" 8-Core Processor", "")
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
        f'viewBox="0 0 {WIDTH} {height}" role="img" font-family="{FONT}" '
        f'aria-label="{escape(text["aria"])}">',
        f'<rect width="{WIDTH}" height="{height}" rx="8" fill="{theme["surface"]}"/>',
        f'<text x="20" y="32" font-size="16" font-weight="600" fill="{theme["text"]}">'
        f"{escape(text['title'])}</text>",
        f'<text x="20" y="52" font-size="12" fill="{theme["secondary"]}">'
        f"{escape(text['subtitle'].format(cpu=cpu))}</text>",
    ]
    # Leyenda
    lx = WIDTH - 190
    parts += [
        f'<rect x="{lx}" y="25" width="12" height="12" rx="2" fill="{theme["p50"]}"/>',
        f'<text x="{lx + 18}" y="35" font-size="12" fill="{theme["secondary"]}">p50</text>',
        f'<rect x="{lx + 60}" y="23" width="3" height="16" rx="1" fill="{theme["p95"]}"/>',
        f'<text x="{lx + 70}" y="35" font-size="12" fill="{theme["secondary"]}">p95</text>',
    ]
    # Cuadrícula y eje
    bottom = TOP + ROW_H * len(stats) - 8
    for tick in range(0, int(domain) + 1, 250):
        x = sx(tick)
        color = theme["axis"] if tick == 0 else theme["grid"]
        parts += [
            f'<line x1="{x:.1f}" y1="{TOP - 12}" x2="{x:.1f}" y2="{bottom}" '
            f'stroke="{color}" stroke-width="1"/>',
            f'<text x="{x:.1f}" y="{bottom + 18}" font-size="11" text-anchor="middle" '
            f'fill="{theme["muted"]}">{tick} ms</text>',
        ]
    # Barras
    for i, (label, s) in enumerate(stats):
        y = TOP + i * ROW_H
        x50, x95 = sx(s["p50_ms"]), sx(s["p95_ms"])
        mid = y + BAR_H / 2
        parts += [
            f'<text x="{LEFT - 12}" y="{mid + 4:.1f}" font-size="13" text-anchor="end" '
            f'fill="{theme["text"]}">{escape(label)}</text>',
            f"<g><title>{escape(label)}: p50 {s['p50_ms']:.1f} ms, p95 {s['p95_ms']:.1f} ms"
            "</title>",
            f'<path d="{_bar(sx(0), x50, y, BAR_H)}" fill="{theme["p50"]}"/>',
            f'<line x1="{x50 + 2:.1f}" y1="{mid:.1f}" x2="{x95:.1f}" y2="{mid:.1f}" '
            f'stroke="{theme["p95"]}" stroke-width="2"/>',
            f'<rect x="{x95 - 1.5:.1f}" y="{y + 3}" width="3" height="{BAR_H - 6}" rx="1" '
            f'fill="{theme["p95"]}"/></g>',
            f'<text x="{x95 + 10:.1f}" y="{mid + 4:.1f}" font-size="13" font-weight="600" '
            f'fill="{theme["text"]}">{s["p50_ms"]:.0f} ms'
            f'<tspan font-weight="400" fill="{theme["muted"]}"> · p95 {s["p95_ms"]:.0f}</tspan>'
            "</text>",
        ]
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--results", type=Path, default=Path("benchmarks/results.json"))
    parser.add_argument("--out", type=Path, default=Path("docs"))
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    latency = json.loads(args.results.read_text(encoding="utf-8"))["latency"]
    args.out.mkdir(parents=True, exist_ok=True)
    for text in LANGUAGES.values():
        for name, theme in THEMES.items():
            path = args.out / f"latency-{name}{text['suffix']}.svg"
            path.write_text(render(latency, theme, text), encoding="utf-8")
            logger.info("Gráfica en %s", path)


if __name__ == "__main__":
    main()
