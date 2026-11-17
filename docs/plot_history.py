"""Draw docs/assets/curriculum.svg from the training record.

    python docs/plot_history.py
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
HISTORY = HERE / "data" / "history.jsonl"
OUT = HERE / "assets" / "curriculum.svg"
COLORS = {"r0-most": "#f2b897", "r1-pair": "#e8d27c", "r2-panel": "#a9c1ee",
          "r3-units": "#98cfe0", "r4-constraints": "#bfaeea",
          "r5-real": "#5f9e80"}
W, H, LEFT, RIGHT, TOP, BOTTOM = 1100, 360, 60, 24, 44, 34
FLOOR = 0.2


def main() -> None:
    rows = [json.loads(line) for line in HISTORY.read_text().splitlines()
            if line.strip()]
    hours, elapsed = [], 0.0
    for row in rows:
        elapsed += row["seconds"] / 3600
        hours.append(elapsed)
    total = hours[-1]
    x = lambda h: LEFT + (W - LEFT - RIGHT) * h / total
    y = lambda a: TOP + (H - TOP - BOTTOM) * (1 - (a - FLOOR) / (1 - FLOOR))
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
             'font-family="-apple-system, Segoe UI, Inter, sans-serif">',
             f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    for level in (0.25, 0.5, 0.75, 0.9, 1.0):
        parts.append(f'<line x1="{LEFT}" x2="{W - RIGHT}" y1="{y(level):.1f}" '
                     f'y2="{y(level):.1f}" stroke="#e6e8ef"/>')
        parts.append(f'<text x="{LEFT - 8}" y="{y(level) + 4:.1f}" font-size="11" '
                     f'text-anchor="end" fill="#6b7486">{int(level * 100)}%</text>')
    start = 0
    for index in range(1, len(rows) + 1):
        if index == len(rows) or rows[index]["rung"] != rows[start]["rung"]:
            rung = rows[start]["rung"]
            points = " ".join(f"{x(hours[i]):.1f},{y(rows[i]['accuracy']):.1f}"
                              for i in range(start, index))
            parts.append(f'<polyline points="{points}" fill="none" '
                         f'stroke="{COLORS[rung]}" stroke-width="1.6"/>')
            start = index
    for day in range(0, int(total // 24) + 1, 4):
        parts.append(f'<text x="{x(day * 24):.1f}" y="{H - 10}" font-size="11" '
                     f'text-anchor="middle" fill="#6b7486">{day} d</text>')
    for slot, (rung, color) in enumerate(COLORS.items()):
        left = W - RIGHT - 6 * 118 + slot * 118
        parts.append(f'<rect x="{left}" y="24" width="14" height="4" rx="2" '
                     f'fill="{color}"/>')
        parts.append(f'<text x="{left + 20}" y="30" font-size="11" '
                     f'fill="#2f3747">{rung}</text>')
    first = datetime.fromisoformat(rows[0]["at"].replace("Z", "+00:00"))
    parts.append(f'<text x="{LEFT}" y="18" font-size="12" fill="#2f3747">'
                 f'Exam accuracy per cycle, by rung, against training hours '
                 f'({len(rows)} cycles from {first:%d %b %Y}, '
                 f'{total:.0f} h of training)</text>')
    parts.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
