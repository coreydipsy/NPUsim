#!/usr/bin/env python3

"""Plot prefill and decode execution time against batch size.

Expected input layout:
    <results_dir>/bs1/prefill.json
    <results_dir>/bs1/decode.json
    <results_dir>/bs4/prefill.json
    <results_dir>/bs4/decode.json
    ...

The script extracts `total_execution_time_chip_ns` from each JSON, writes a
summary CSV, and saves either a line chart or grouped bar chart.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import tempfile
from pathlib import Path


_BATCH_DIR_RE = re.compile(r"bs(\d+)$")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot prefill and decode execution time versus batch size."
    )
    parser.add_argument(
        "results_dir",
        type=Path,
        help="Directory containing bs*/prefill.json and bs*/decode.json subdirectories.",
    )
    parser.add_argument(
        "--chart-type",
        choices=("line", "bar"),
        default="line",
        help="Chart style for the output figure.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output image path. Defaults to <results_dir>/batch_size_execution_times.png.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <results_dir>/batch_size_execution_times.csv.",
    )
    return parser.parse_args()


def _load_rows(results_dir: Path) -> list[dict[str, float | int]]:
    rows: list[dict[str, float | int]] = []

    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue

        match = _BATCH_DIR_RE.fullmatch(child.name)
        if not match:
            continue

        batch_size = int(match.group(1))
        prefill_path = child / "prefill.json"
        decode_path = child / "decode.json"
        if not prefill_path.exists() or not decode_path.exists():
            continue

        with open(prefill_path) as f:
            prefill = json.load(f)
        with open(decode_path) as f:
            decode = json.load(f)

        prefill_time_ns = float(prefill["total_execution_time_chip_ns"])
        decode_time_ns = float(decode["total_execution_time_chip_ns"])

        rows.append(
            {
                "batch_size": batch_size,
                "prefill_execution_time_ms": prefill_time_ns / 1e6,
                "decode_execution_time_ms": decode_time_ns / 1e6,
                "prefill_execution_time_sec": prefill_time_ns / 1e9,
                "decode_execution_time_sec": decode_time_ns / 1e9,
                "TTFT_sec": float(prefill.get("TTFT_sec", prefill_time_ns / 1e9)),
                "TPOT_ms_request": float(decode.get("TPOT_ms_request", 0.0)),
            }
        )

    rows.sort(key=lambda row: int(row["batch_size"]))
    return rows


def _write_csv(rows: list[dict[str, float | int]], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _format_tick(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}"
    if value >= 1000:
        return f"{value:.0f}"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


def _nice_tick_step(max_value: float, target_intervals: int = 5) -> float:
    if max_value <= 0:
        return 1.0

    raw_step = max_value / max(target_intervals, 1)
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10 ** exponent)

    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10

    return nice_fraction * (10 ** exponent)


def _plot_svg(
    rows: list[dict[str, float | int]],
    output_path: Path,
    chart_type: str,
) -> Path:
    actual_output = output_path if output_path.suffix.lower() == ".svg" else output_path.with_suffix(".svg")

    width = 920
    height = 560
    margin_left = 95
    margin_right = 35
    margin_top = 65
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    plot_x0 = margin_left
    plot_y0 = margin_top
    plot_x1 = plot_x0 + plot_width
    plot_y1 = plot_y0 + plot_height

    batch_sizes = [int(row["batch_size"]) for row in rows]
    prefill_ms = [float(row["prefill_execution_time_ms"]) for row in rows]
    decode_ms = [float(row["decode_execution_time_ms"]) for row in rows]
    data_max = max(prefill_ms + decode_ms)
    tick_step = _nice_tick_step(data_max)
    y_max = max(tick_step * math.ceil(data_max / tick_step), tick_step)
    tick_values = [idx * tick_step for idx in range(int(round(y_max / tick_step)) + 1)]

    if len(batch_sizes) == 1:
        centers = [plot_x0 + plot_width / 2]
    else:
        step = plot_width / (len(batch_sizes) - 1)
        centers = [plot_x0 + idx * step for idx in range(len(batch_sizes))]

    def scale_y(value: float) -> float:
        return plot_y1 - (value / y_max) * plot_height

    parts: list[str] = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="white" />',
        (
            f'<text x="{width / 2:.1f}" y="34" text-anchor="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="22" font-weight="bold">'
            'Execution Time vs Batch Size'
            "</text>"
        ),
    ]

    for tick_value in tick_values:
        y = scale_y(tick_value)
        parts.append(
            f'<line x1="{plot_x0:.2f}" y1="{y:.2f}" x2="{plot_x1:.2f}" y2="{y:.2f}" '
            'stroke="#d9d9d9" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{plot_x0 - 12:.2f}" y="{y + 5:.2f}" text-anchor="end" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#333333">'
            f"{_format_tick(tick_value)}"
            "</text>"
        )

    parts.append(
        f'<line x1="{plot_x0:.2f}" y1="{plot_y1:.2f}" x2="{plot_x1:.2f}" y2="{plot_y1:.2f}" '
        'stroke="#333333" stroke-width="1.5" />'
    )
    parts.append(
        f'<line x1="{plot_x0:.2f}" y1="{plot_y0:.2f}" x2="{plot_x0:.2f}" y2="{plot_y1:.2f}" '
        'stroke="#333333" stroke-width="1.5" />'
    )

    for x, batch_size in zip(centers, batch_sizes):
        parts.append(
            f'<line x1="{x:.2f}" y1="{plot_y1:.2f}" x2="{x:.2f}" y2="{plot_y1 + 6:.2f}" '
            'stroke="#333333" stroke-width="1.2" />'
        )
        parts.append(
            f'<text x="{x:.2f}" y="{plot_y1 + 26:.2f}" text-anchor="middle" '
            'font-family="Arial, Helvetica, sans-serif" font-size="12" fill="#333333">'
            f"{batch_size}"
            "</text>"
        )

    prefill_color = "#1f77b4"
    decode_color = "#d62728"

    if chart_type == "bar":
        if len(batch_sizes) == 1:
            group_width = plot_width * 0.25
        else:
            group_width = min((centers[1] - centers[0]) * 0.65, 90.0)
        bar_width = group_width / 2.4

        for x, prefill_value, decode_value in zip(centers, prefill_ms, decode_ms):
            prefill_left = x - bar_width - 4
            decode_left = x + 4
            parts.append(
                f'<rect x="{prefill_left:.2f}" y="{scale_y(prefill_value):.2f}" '
                f'width="{bar_width:.2f}" height="{plot_y1 - scale_y(prefill_value):.2f}" '
                f'fill="{prefill_color}" opacity="0.85" />'
            )
            parts.append(
                f'<rect x="{decode_left:.2f}" y="{scale_y(decode_value):.2f}" '
                f'width="{bar_width:.2f}" height="{plot_y1 - scale_y(decode_value):.2f}" '
                f'fill="{decode_color}" opacity="0.85" />'
            )
    else:
        prefill_points = " ".join(
            f"{x:.2f},{scale_y(value):.2f}" for x, value in zip(centers, prefill_ms)
        )
        decode_points = " ".join(
            f"{x:.2f},{scale_y(value):.2f}" for x, value in zip(centers, decode_ms)
        )
        parts.append(
            f'<polyline fill="none" stroke="{prefill_color}" stroke-width="3" '
            f'points="{prefill_points}" />'
        )
        parts.append(
            f'<polyline fill="none" stroke="{decode_color}" stroke-width="3" '
            f'stroke-dasharray="9 6" points="{decode_points}" />'
        )
        for x, value in zip(centers, prefill_ms):
            parts.append(
                f'<circle cx="{x:.2f}" cy="{scale_y(value):.2f}" r="5" fill="{prefill_color}" />'
            )
        for x, value in zip(centers, decode_ms):
            y = scale_y(value)
            size = 5.5
            parts.append(
                f'<rect x="{x - size:.2f}" y="{y - size:.2f}" width="{2 * size:.2f}" '
                f'height="{2 * size:.2f}" fill="{decode_color}" />'
            )

    legend_x = plot_x1 - 160
    legend_y = plot_y0 + 12
    parts.append(
        f'<line x1="{legend_x:.2f}" y1="{legend_y:.2f}" x2="{legend_x + 28:.2f}" y2="{legend_y:.2f}" '
        f'stroke="{prefill_color}" stroke-width="3" />'
    )
    parts.append(
        f'<circle cx="{legend_x + 14:.2f}" cy="{legend_y:.2f}" r="5" fill="{prefill_color}" />'
    )
    parts.append(
        f'<text x="{legend_x + 38:.2f}" y="{legend_y + 4:.2f}" '
        'font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#333333">'
        "Prefill"
        "</text>"
    )
    parts.append(
        f'<line x1="{legend_x:.2f}" y1="{legend_y + 24:.2f}" '
        f'x2="{legend_x + 28:.2f}" y2="{legend_y + 24:.2f}" '
        f'stroke="{decode_color}" stroke-width="3" stroke-dasharray="9 6" />'
    )
    parts.append(
        f'<rect x="{legend_x + 8.5:.2f}" y="{legend_y + 18.5:.2f}" width="11" height="11" '
        f'fill="{decode_color}" />'
    )
    parts.append(
        f'<text x="{legend_x + 38:.2f}" y="{legend_y + 28:.2f}" '
        'font-family="Arial, Helvetica, sans-serif" font-size="13" fill="#333333">'
        "Decode"
        "</text>"
    )

    parts.append(
        f'<text x="{width / 2:.1f}" y="{height - 22:.1f}" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#333333">'
        "Batch Size"
        "</text>"
    )
    parts.append(
        f'<text x="24" y="{height / 2:.1f}" text-anchor="middle" '
        'font-family="Arial, Helvetica, sans-serif" font-size="14" fill="#333333" '
        f'transform="rotate(-90 24 {height / 2:.1f})">'
        "Execution Time (ms)"
        "</text>"
    )
    parts.append("</svg>")

    actual_output.write_text("\n".join(parts))
    return actual_output


def _plot(rows: list[dict[str, float | int]], output_path: Path, chart_type: str) -> tuple[Path, str]:
    mpl_config_dir = Path(tempfile.gettempdir()) / "neusim-mpl-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    try:
        import os
        os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return _plot_svg(rows, output_path, chart_type), "svg"

    batch_sizes = [int(row["batch_size"]) for row in rows]
    x_positions = list(range(len(batch_sizes)))
    prefill_ms = [float(row["prefill_execution_time_ms"]) for row in rows]
    decode_ms = [float(row["decode_execution_time_ms"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8.5, 5))

    if chart_type == "bar":
        width = 0.36
        ax.bar(
            [x - width / 2 for x in x_positions],
            prefill_ms,
            width=width,
            label="Prefill",
        )
        ax.bar(
            [x + width / 2 for x in x_positions],
            decode_ms,
            width=width,
            label="Decode",
        )
    else:
        ax.plot(x_positions, prefill_ms, "o-", linewidth=2, label="Prefill")
        ax.plot(x_positions, decode_ms, "s--", linewidth=2, label="Decode")

    ax.set_title("Execution Time vs Batch Size")
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Execution Time (ms)")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([str(bs) for bs in batch_sizes])
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path, "matplotlib"


def main() -> None:
    args = _parse_args()
    results_dir = args.results_dir.resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    rows = _load_rows(results_dir)
    if not rows:
        raise FileNotFoundError(
            f"No batch-size results found under {results_dir}. "
            "Expected bs*/prefill.json and bs*/decode.json."
        )

    output_path = args.output or results_dir / "batch_size_execution_times.png"
    summary_csv = args.summary_csv or results_dir / "batch_size_execution_times.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_csv.parent.mkdir(parents=True, exist_ok=True)

    _write_csv(rows, summary_csv)
    actual_output_path, backend = _plot(rows, output_path, args.chart_type)

    print(f"Wrote summary CSV: {summary_csv}")
    if backend == "svg" and actual_output_path != output_path:
        print(
            "matplotlib not available; wrote SVG plot instead: "
            f"{actual_output_path}"
        )
    else:
        print(f"Wrote plot: {actual_output_path}")


if __name__ == "__main__":
    main()
