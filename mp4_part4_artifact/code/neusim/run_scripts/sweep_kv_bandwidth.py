#!/usr/bin/env python3

"""Sweep KV-transfer overhead versus DCN bandwidth for MP4 Part IV discussion."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "configs").is_dir() and (candidate / "neusim").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate the NeuSim repo root from {start}.")


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = _find_repo_root(SCRIPT_DIR)
SCRIPT_STEM = Path(__file__).stem
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import sweep_pd_allocation


DEFAULT_MODEL = REPO_ROOT / "configs/models/gpt-oss-120b.json"
DEFAULT_CHIP = REPO_ROOT / "configs/chips/tpuv5p.json"
DEFAULT_PART3_CSV = REPO_ROOT / "MP4 Part 3.2 Throughput/parallelism_sweep.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / f"results/{SCRIPT_STEM}"
DEFAULT_INPUT_SEQLEN = 2048
DEFAULT_OUTPUT_SEQLEN = 128
DEFAULT_BANDWIDTHS_GBPS = [
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    300,
    350,
    400,
]


def _write_csv(rows: list[dict[str, float | int]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _compute_rows(metrics: dict[str, Any], bandwidths_GBps: list[float]) -> list[dict[str, float | int]]:
    kv_cache_bytes = metrics["kv_cache_transfer_bytes"]
    prefill_ttft_sec = metrics["prefill_time_sec_without_kv_transfer"]
    rows: list[dict[str, float | int]] = []

    for bandwidth_GBps in bandwidths_GBps:
        transfer_time_sec = kv_cache_bytes / (bandwidth_GBps * 1024**3)
        total_ttft_sec = prefill_ttft_sec + transfer_time_sec
        rows.append(
            {
                "dcn_bw_GBps": bandwidth_GBps,
                "kv_transfer_time_ms": transfer_time_sec * 1e3,
                "prefill_ttft_ms_without_kv_transfer": prefill_ttft_sec * 1e3,
                "total_ttft_ms_with_kv_transfer": total_ttft_sec * 1e3,
                "kv_fraction_of_prefill_compute": transfer_time_sec / prefill_ttft_sec,
                "kv_fraction_of_total_ttft": transfer_time_sec / total_ttft_sec,
            }
        )

    return rows


def _plot(rows: list[dict[str, float | int]], output_path: Path) -> None:
    mpl_config_dir = Path(tempfile.gettempdir()) / "neusim-mpl-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib.pyplot as plt

    bandwidths = [float(row["dcn_bw_GBps"]) for row in rows]
    kv_ms = [float(row["kv_transfer_time_ms"]) for row in rows]
    prefill_ms = [float(row["prefill_ttft_ms_without_kv_transfer"]) for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

    axes[0].plot(bandwidths, kv_ms, marker="o", linewidth=2.5, label="KV transfer overhead")
    axes[0].plot(
        bandwidths,
        prefill_ms,
        linestyle="--",
        linewidth=2,
        color="black",
        label="Prefill compute TTFT",
    )

    axes[1].plot(bandwidths, kv_ms, marker="o", linewidth=2.5, color="#1f77b4")

    for target_bw in [50, 200]:
        for bw, val in zip(bandwidths, kv_ms):
            if abs(bw - target_bw) < 1e-9:
                axes[0].scatter(bw, val, color="#d62728", s=70, zorder=5)
                axes[1].scatter(bw, val, color="#d62728", s=70, zorder=5)
                axes[1].annotate(
                    f"{int(target_bw)} GB/s\n{val:.3f} ms",
                    (bw, val),
                    textcoords="offset points",
                    xytext=(6, 8),
                    fontsize=9,
                )
                break

    axes[0].set_title("Comparison to Prefill Compute")
    axes[0].set_xlabel("DCN bandwidth (GB/s)")
    axes[0].set_ylabel("Time (ms)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].set_title("KV Transfer Overhead (Zoomed)")
    axes[1].set_xlabel("DCN bandwidth (GB/s)")
    axes[1].set_ylabel("KV transfer time (ms)")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("KV Transfer Overhead vs DCN Bandwidth")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep KV-transfer overhead versus DCN bandwidth."
    )
    parser.add_argument("--model_config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--chip_config", type=Path, default=DEFAULT_CHIP)
    parser.add_argument("--part3_csv", type=Path, default=DEFAULT_PART3_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input_seqlen", type=int, default=DEFAULT_INPUT_SEQLEN)
    parser.add_argument("--output_seqlen", type=int, default=DEFAULT_OUTPUT_SEQLEN)
    parser.add_argument(
        "--bandwidths_GBps",
        type=str,
        default=",".join(str(x) for x in DEFAULT_BANDWIDTHS_GBPS),
        help="Comma-separated DCN bandwidth values in GB/s.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = sweep_pd_allocation._load_json(args.model_config)
    model_name = model_config["model_name"]
    part3_rows = sweep_pd_allocation._load_part3_rows(args.part3_csv, model_name)
    prefill_choice = sweep_pd_allocation._pick_best_config(part3_rows, "sr_pf_tps")
    decode_choice = sweep_pd_allocation._pick_best_config(part3_rows, "sr_dc_tps")

    metrics = sweep_pd_allocation._simulate_selected_configs(
        args.model_config,
        args.chip_config,
        output_dir,
        prefill_choice=prefill_choice,
        decode_choice=decode_choice,
        input_seqlen=args.input_seqlen,
        output_seqlen=args.output_seqlen,
    )

    bandwidths_GBps = [float(x) for x in args.bandwidths_GBps.split(",") if x.strip()]
    rows = _compute_rows(metrics, bandwidths_GBps)

    csv_path = output_dir / f"{SCRIPT_STEM}.csv"
    plot_path = output_dir / f"{SCRIPT_STEM}.png"
    summary_path = output_dir / f"{SCRIPT_STEM}_summary.json"
    _write_csv(rows, csv_path)
    _plot(rows, plot_path)

    by_bandwidth = {str(int(row["dcn_bw_GBps"])): row for row in rows if float(row["dcn_bw_GBps"]).is_integer()}
    summary = {
        "model_name": metrics["model_name"],
        "chip_name": metrics["chip_name"],
        "input_seqlen": args.input_seqlen,
        "output_seqlen": args.output_seqlen,
        "selected_prefill_config_from_part3": prefill_choice,
        "prefill_ttft_ms_without_kv_transfer": metrics["prefill_time_sec_without_kv_transfer"] * 1e3,
        "kv_cache_transfer_bytes": metrics["kv_cache_transfer_bytes"],
        "rows": rows,
        "requested_bandwidth_points": {
            "50_GBps": by_bandwidth.get("50"),
            "200_GBps": by_bandwidth.get("200"),
        },
        "csv_path": str(csv_path),
        "plot_path": str(plot_path),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
