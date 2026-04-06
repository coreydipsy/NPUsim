#!/usr/bin/env python3

"""Compare full attention vs sliding-window attention for GPT-OSS.

This script implements MP4 Part II §2.2 item 2 using NeuSim's existing
operator builders and single-op simulator helper. For each sequence length, it:

1. Builds a single full-attention layer in prefill mode.
2. Builds a single sliding-window attention layer in prefill mode.
3. Simulates each operator in the layer on the chosen chip config.
4. Aggregates FLOPs and memory traffic across the layer.
5. Writes a CSV summary and saves a two-panel comparison plot.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile

NEUSIM_ROOT = Path(__file__).resolve().parent.parent.parent
if str(NEUSIM_ROOT) not in sys.path:
    sys.path.insert(0, str(NEUSIM_ROOT))

import neusim.npusim.frontend.llm_ops_lib as ops_lib
from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.models.LLMConfig import GptOssConfig
from neusim.npusim.frontend.Operator import Operator
from neusim.npusim.frontend.run_single_op import run_sim_single_op

DEFAULT_MODEL = NEUSIM_ROOT / "configs/models/gpt-oss-20b.json"
DEFAULT_CHIP = NEUSIM_ROOT / "configs/chips/tpuv5p.json"
DEFAULT_OUTPUT_DIR = NEUSIM_ROOT / "results/attn_compare"
DEFAULT_SEQ_LENS = [512, 1024, 2048, 4096, 8192]


def _load_model_config(path: Path) -> GptOssConfig:
    with open(path) as f:
        return GptOssConfig.model_validate(json.load(f))


def _load_chip_config(path: Path) -> ChipConfig:
    with open(path) as f:
        return ChipConfig.model_validate(json.load(f))


def _build_full_attention_ops(config: GptOssConfig, seq_len: int) -> list[Operator]:
    return ops_lib.create_multi_head_attention(
        batch_size=config.global_batch_size,
        input_seqlen=seq_len,
        output_seqlen=config.output_seqlen,
        decode_width=config.decode_width,
        num_heads=config.num_heads,
        d_model=config.d_model,
        d_head=config.d_head,
        num_layers=1,
        config=config,
        is_decode=False,
        use_flash_attention=config.use_flash_attention,
        description_prefix=f"full-attn-seq{seq_len}",
        num_kv_heads=config.num_kv_heads,
    )


def _build_sliding_attention_ops(
    config: GptOssConfig,
    seq_len: int,
    sliding_window_size: int,
) -> list[Operator]:
    return ops_lib.create_sliding_window_attention(
        batch_size=config.global_batch_size,
        q_seqlen=seq_len,
        sliding_window_size=sliding_window_size,
        num_heads=config.num_heads,
        num_kv_heads=config.num_kv_heads,
        d_model=config.d_model,
        d_head=config.d_head,
        num_layers=1,
        config=config,
        is_decode=False,
        description_prefix=f"sliding-attn-seq{seq_len}",
        use_flash_attention=config.use_flash_attention,
    )


def _simulate_ops(ops: list[Operator], chip_config: ChipConfig) -> list[Operator]:
    with contextlib.redirect_stdout(io.StringIO()):
        return [run_sim_single_op(op, chip_config) for op in ops]


def _sum_flops(ops: list[Operator]) -> int:
    return sum(op.stats.flop_count * op.stats.count for op in ops)


def _sum_bytes(ops: list[Operator]) -> int:
    return sum(op.stats.memory_traffic_bytes * op.stats.count for op in ops)


def _write_csv(rows: list[dict[str, float | int]], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_with_matplotlib(rows: list[dict[str, float | int]], output_path: Path) -> None:
    mpl_config_dir = Path(tempfile.gettempdir()) / "neusim-mpl-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib.pyplot as plt

    seq_lens = [row["seq_len"] for row in rows]
    x_positions = list(range(len(seq_lens)))
    full_gflops = [row["full_gflops"] for row in rows]
    sliding_gflops = [row["sliding_gflops"] for row in rows]
    full_gb = [row["full_gb"] for row in rows]
    sliding_gb = [row["sliding_gb"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(x_positions, full_gflops, marker="o", linewidth=2, label="Full attention")
    axes[0].plot(x_positions, sliding_gflops, marker="o", linewidth=2, label="Sliding window")
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels(seq_lens)
    axes[0].set_xlim(-0.2, len(seq_lens) - 1 + 0.2)
    axes[0].set_title("Attention FLOPs")
    axes[0].set_xlabel("Input sequence length")
    axes[0].set_ylabel("GFLOPs per layer")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_positions, full_gb, marker="o", linewidth=2, label="Full attention")
    axes[1].plot(x_positions, sliding_gb, marker="o", linewidth=2, label="Sliding window")
    axes[1].set_xticks(x_positions)
    axes[1].set_xticklabels(seq_lens)
    axes[1].set_xlim(-0.2, len(seq_lens) - 1 + 0.2)
    axes[1].set_title("Attention Memory Traffic")
    axes[1].set_xlabel("Input sequence length")
    axes[1].set_ylabel("GB per layer")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("GPT-OSS Attention Comparison (Prefill, 1 Layer)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_with_svg(rows: list[dict[str, float | int]], output_path: Path) -> None:
    width = 1200
    height = 450
    margin_left = 70
    margin_right = 30
    margin_top = 50
    margin_bottom = 55
    gap = 70
    panel_width = (width - margin_left - margin_right - gap) / 2
    panel_height = height - margin_top - margin_bottom

    seq_lens = [int(row["seq_len"]) for row in rows]
    full_gflops = [float(row["full_gflops"]) for row in rows]
    sliding_gflops = [float(row["sliding_gflops"]) for row in rows]
    full_gb = [float(row["full_gb"]) for row in rows]
    sliding_gb = [float(row["sliding_gb"]) for row in rows]

    def scale_x(idx: int, x0: float) -> float:
        if len(seq_lens) == 1:
            return x0 + panel_width / 2
        return x0 + idx * panel_width / (len(seq_lens) - 1)

    def scale_y(val: float, y_max: float) -> float:
        return margin_top + panel_height - (val * panel_height / y_max)

    def polyline(values: list[float], x0: float, y_max: float) -> str:
        pts = [
            f"{scale_x(idx, x0):.2f},{scale_y(val, y_max):.2f}"
            for idx, val in enumerate(values)
        ]
        return " ".join(pts)

    def circles(values: list[float], x0: float, y_max: float, color: str) -> str:
        out = []
        for idx, val in enumerate(values):
            out.append(
                f'<circle cx="{scale_x(idx, x0):.2f}" cy="{scale_y(val, y_max):.2f}" '
                f'r="4" fill="{color}" />'
            )
        return "\n".join(out)

    def panel(
        title: str,
        y_label: str,
        full_values: list[float],
        sliding_values: list[float],
        x0: float,
    ) -> str:
        y_max = max(full_values + sliding_values) * 1.1
        x1 = x0 + panel_width
        y0 = margin_top
        y1 = margin_top + panel_height
        ticks = 5

        grid = []
        labels = []
        for i in range(ticks + 1):
            frac = i / ticks
            y = y1 - frac * panel_height
            value = frac * y_max
            grid.append(
                f'<line x1="{x0:.2f}" y1="{y:.2f}" x2="{x1:.2f}" y2="{y:.2f}" '
                'stroke="#dddddd" stroke-width="1" />'
            )
            labels.append(
                f'<text x="{x0 - 10:.2f}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-size="12">{value:.1f}</text>'
            )

        x_ticks = []
        for idx, seq in enumerate(seq_lens):
            x = scale_x(idx, x0)
            x_ticks.append(
                f'<text x="{x:.2f}" y="{y1 + 22:.2f}" text-anchor="middle" '
                f'font-size="12">{seq}</text>'
            )

        return f"""
        <g>
          <rect x="{x0:.2f}" y="{y0:.2f}" width="{panel_width:.2f}" height="{panel_height:.2f}"
                fill="white" stroke="#333333" stroke-width="1.5" />
          {''.join(grid)}
          {''.join(labels)}
          {''.join(x_ticks)}
          <text x="{(x0 + x1) / 2:.2f}" y="{y0 - 15:.2f}" text-anchor="middle"
                font-size="16" font-weight="bold">{title}</text>
          <text x="{(x0 + x1) / 2:.2f}" y="{y1 + 42:.2f}" text-anchor="middle"
                font-size="13">Input sequence length</text>
          <text x="{x0 - 52:.2f}" y="{(y0 + y1) / 2:.2f}" text-anchor="middle"
                transform="rotate(-90 {x0 - 52:.2f} {(y0 + y1) / 2:.2f})"
                font-size="13">{y_label}</text>
          <polyline points="{polyline(full_values, x0, y_max)}" fill="none"
                    stroke="#1f77b4" stroke-width="2.5" />
          <polyline points="{polyline(sliding_values, x0, y_max)}" fill="none"
                    stroke="#d62728" stroke-width="2.5" />
          {circles(full_values, x0, y_max, "#1f77b4")}
          {circles(sliding_values, x0, y_max, "#d62728")}
        </g>
        """

    left_x0 = margin_left
    right_x0 = margin_left + panel_width + gap

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">
    <rect width="100%" height="100%" fill="#fafafa" />
    <text x="{width / 2:.2f}" y="28" text-anchor="middle" font-size="20" font-weight="bold">
      GPT-OSS Attention Comparison (Prefill, 1 Layer)
    </text>
    {panel("Attention FLOPs", "GFLOPs per layer", full_gflops, sliding_gflops, left_x0)}
    {panel("Attention Memory Traffic", "GB per layer", full_gb, sliding_gb, right_x0)}
    <g>
      <line x1="{width - 250}" y1="35" x2="{width - 220}" y2="35" stroke="#1f77b4" stroke-width="2.5" />
      <circle cx="{width - 235}" cy="35" r="4" fill="#1f77b4" />
      <text x="{width - 210}" y="39" font-size="12">Full attention</text>
      <line x1="{width - 120}" y1="35" x2="{width - 90}" y2="35" stroke="#d62728" stroke-width="2.5" />
      <circle cx="{width - 105}" cy="35" r="4" fill="#d62728" />
      <text x="{width - 80}" y="39" font-size="12">Sliding window</text>
    </g>
    </svg>
    """
    output_path.write_text(svg)


def _plot(rows: list[dict[str, float | int]], output_dir: Path) -> Path:
    png_path = output_dir / "attention_compare.png"
    try:
        _plot_with_matplotlib(rows, png_path)
        return png_path
    except ModuleNotFoundError:
        svg_path = output_dir / "attention_compare.svg"
        _plot_with_svg(rows, svg_path)
        return svg_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare full attention vs sliding-window attention for GPT-OSS."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to the GPT-OSS model config JSON.",
    )
    parser.add_argument(
        "--chip",
        type=Path,
        default=DEFAULT_CHIP,
        help="Path to the chip config JSON.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write CSV and plot outputs.",
    )
    parser.add_argument(
        "--sliding_window_size",
        type=int,
        default=None,
        help="Override the sliding-window size. Defaults to the model config value.",
    )
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENS,
        help="Input sequence lengths to compare.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = _load_model_config(args.model)
    chip_config = _load_chip_config(args.chip)
    sliding_window_size = args.sliding_window_size or model_config.sliding_window_size

    rows: list[dict[str, float | int]] = []
    for seq_len in args.seq_lens:
        full_ops = _simulate_ops(_build_full_attention_ops(model_config, seq_len), chip_config)
        sliding_ops = _simulate_ops(
            _build_sliding_attention_ops(model_config, seq_len, sliding_window_size),
            chip_config,
        )

        full_flops = _sum_flops(full_ops)
        sliding_flops = _sum_flops(sliding_ops)
        full_bytes = _sum_bytes(full_ops)
        sliding_bytes = _sum_bytes(sliding_ops)

        rows.append(
            {
                "seq_len": seq_len,
                "full_flops": full_flops,
                "sliding_flops": sliding_flops,
                "full_bytes": full_bytes,
                "sliding_bytes": sliding_bytes,
                "full_gflops": full_flops / 1e9,
                "sliding_gflops": sliding_flops / 1e9,
                "full_gb": full_bytes / 1e9,
                "sliding_gb": sliding_bytes / 1e9,
                "flops_reduction_x": full_flops / sliding_flops,
                "bytes_reduction_x": full_bytes / sliding_bytes,
            }
        )

    csv_path = output_dir / "attention_compare.csv"
    _write_csv(rows, csv_path)
    plot_path = _plot(rows, output_dir)

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
