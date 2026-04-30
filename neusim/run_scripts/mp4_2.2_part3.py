#!/usr/bin/env python3

"""Run the GPT-OSS-20B sequence-length sweep for MP4 Part II.

This script reproduces the deliverable:
  - input sequence lengths: 1024, 2048, 4096, 8192
  - output sequence length: 128
  - phases: prefill and decode
  - metrics: total memory footprint, total memory traffic, total FLOPs

It uses the GPT-OSS ops generator directly, simulates the generated operators on
the selected chip configuration, writes a CSV summary, and saves a 3-panel plot.
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


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "configs").is_dir() and (candidate / "neusim").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate NeuSim repo root from {start}.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
SCRIPT_STEM = Path(__file__).stem
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neusim.configs.chips.ChipConfig import ChipConfig
from neusim.configs.models.LLMConfig import GptOssConfig
from neusim.npusim.frontend.Operator import Operator
from neusim.npusim.frontend.llm_ops_generator import GptOssOpsGenerator
from neusim.npusim.frontend.run_single_op import run_sim_single_op
from neusim.npusim.frontend.run_sim_lib import get_pstr_moe


DEFAULT_MODEL = REPO_ROOT / "configs/models/gpt-oss-20b.json"
DEFAULT_CHIP = REPO_ROOT / "configs/chips/tpuv5p.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / f"results/{SCRIPT_STEM}"
DEFAULT_RAW_OUTPUT_DIR = REPO_ROOT / "results/raw"
DEFAULT_SEQ_LENS = [1024, 2048, 4096, 8192]
DEFAULT_OUTPUT_SEQLEN = 128
DEFAULT_BATCH_SIZE = 4


def _load_model_config(path: Path) -> GptOssConfig:
    with open(path) as f:
        return GptOssConfig.model_validate(json.load(f))


def _load_chip_config(path: Path) -> ChipConfig:
    with open(path) as f:
        return ChipConfig.model_validate(json.load(f))


def _simulate_ops(ops: list[Operator], chip_config: ChipConfig) -> list[Operator]:
    with contextlib.redirect_stdout(io.StringIO()):
        return [run_sim_single_op(op, chip_config) for op in ops]


def _sum_flops(ops: list[Operator]) -> int:
    return sum(op.stats.flop_count * op.stats.count for op in ops)


def _sum_bytes(ops: list[Operator]) -> int:
    return sum(op.stats.memory_traffic_bytes * op.stats.count for op in ops)


def _get_raw_run_dir(output_root: Path, seq_len: int, output_seqlen: int, batch_size: int) -> Path:
    pstr = get_pstr_moe(
        dp=1,
        tp=1,
        pp=1,
        ep=1,
        dp_dcn=1,
        tp_dcn=1,
        pp_dcn=1,
        ep_dcn=1,
        global_batch_size=batch_size,
    )
    return output_root / f"gpt-oss-20b_{seq_len}_{output_seqlen}" / pstr


def _load_raw_phase_metrics(run_dir: Path, phase: str) -> dict[str, float]:
    csv_path = run_dir / f"inference-v5p_{phase}.csv"
    json_path = run_dir / f"inference-v5p_{phase}.json"
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    with open(json_path) as f:
        stats = json.load(f)
    return {
        "memory_traffic_gb": sum(float(row["Bytes Accessed"]) * float(row["Count"]) for row in rows) / 1e9,
        "flops_t": sum(float(row["FLOP Count"]) * float(row["Count"]) for row in rows) / 1e12,
        "mem_footprint_gb": stats["mem_footprint_GB"],
        "op_count": len(rows),
    }


def _load_raw_metrics(
    output_root: Path,
    seq_len: int,
    output_seqlen: int,
    batch_size: int,
) -> dict[str, float | int] | None:
    run_dir = _get_raw_run_dir(output_root, seq_len, output_seqlen, batch_size)
    required = [
        run_dir / "inference-v5p_prefill.csv",
        run_dir / "inference-v5p_prefill.json",
        run_dir / "inference-v5p_decode.csv",
        run_dir / "inference-v5p_decode.json",
    ]
    if not all(path.exists() for path in required):
        return None

    prefill = _load_raw_phase_metrics(run_dir, "prefill")
    decode = _load_raw_phase_metrics(run_dir, "decode")
    return {
        "prefill_mem_footprint_gb": prefill["mem_footprint_gb"],
        "decode_mem_footprint_gb": decode["mem_footprint_gb"],
        "prefill_memory_traffic_gb": prefill["memory_traffic_gb"],
        "decode_memory_traffic_gb": decode["memory_traffic_gb"],
        "prefill_flops_t": prefill["flops_t"],
        "decode_flops_t": decode["flops_t"],
        "prefill_ops": int(prefill["op_count"]),
        "decode_ops": int(decode["op_count"]),
    }


def _write_csv(rows: list[dict[str, float | int]], output_path: Path) -> None:
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | int]], output_path: Path) -> None:
    mpl_config_dir = Path(tempfile.gettempdir()) / "neusim-mpl-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib.pyplot as plt

    seq_lens = [int(row["input_seqlen"]) for row in rows]
    prefill_mem = [float(row["prefill_mem_footprint_gb"]) for row in rows]
    decode_mem = [float(row["decode_mem_footprint_gb"]) for row in rows]
    prefill_traffic = [float(row["prefill_memory_traffic_gb"]) for row in rows]
    decode_traffic = [float(row["decode_memory_traffic_gb"]) for row in rows]
    prefill_flops = [float(row["prefill_flops_t"]) for row in rows]
    decode_flops = [float(row["decode_flops_t"]) for row in rows]
    batch_size = int(rows[0]["batch_size"])
    output_seqlen = int(rows[0]["output_seqlen"])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def style_axis(ax, title: str, ylabel: str, log_y: bool = False) -> None:
        ax.set_title(title)
        ax.set_xlabel("Input Sequence Length")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(seq_lens)
        ax.set_xticklabels([str(x) for x in seq_lens])
        if log_y:
            ax.set_yscale("log")
        ax.grid(True, alpha=0.3)
        ax.legend()

    axes[0].plot(seq_lens, prefill_mem, "o-", label="Prefill")
    axes[0].plot(seq_lens, decode_mem, "s--", label="Decode")
    style_axis(axes[0], "Memory Footprint", "HBM Footprint (GB)")

    axes[1].plot(seq_lens, prefill_traffic, "o-", label="Prefill")
    axes[1].plot(seq_lens, decode_traffic, "s--", label="Decode")
    style_axis(axes[1], "Memory Traffic", "Bytes Accessed (GB)")

    axes[2].plot(seq_lens, prefill_flops, "o-", label="Prefill")
    axes[2].plot(seq_lens, decode_flops, "s--", label="Decode")
    style_axis(axes[2], "Compute (FLOPs)", "TFLOPs", log_y=True)

    fig.suptitle(
        f"GPT-OSS-20b on TPUv5p (batch={batch_size}, output_seqlen={output_seqlen})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_generator(
    base_model_config: GptOssConfig,
    chip_config: ChipConfig,
    seq_len: int,
    output_seqlen: int,
    output_dir: Path,
    batch_size: int | None,
) -> GptOssOpsGenerator:
    merged = {
        **chip_config.model_dump(),
        **base_model_config.model_dump(),
        "num_chips": 1,
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": 1,
        "pipeline_parallelism_degree": 1,
        "expert_parallelism_degree": 1,
        "data_parallel_degree_dcn": 1,
        "tensor_parallel_degree_dcn": 1,
        "pipeline_parallel_degree_dcn": 1,
        "expert_parallel_degree_dcn": 1,
        "input_seqlen": seq_len,
        "output_seqlen": output_seqlen,
        "output_file_path": str(output_dir / f"{SCRIPT_STEM}_seq{seq_len}.csv"),
    }
    if batch_size is not None:
        merged.update(
            {
                "global_batch_size": batch_size,
                "microbatch_size_ici": batch_size,
                "microbatch_size_dcn": batch_size,
            }
        )
    return GptOssOpsGenerator(merged)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep GPT-OSS-20B over sequence length for prefill and decode."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--chip", type=Path, default=DEFAULT_CHIP)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--raw_output_dir",
        type=Path,
        default=DEFAULT_RAW_OUTPUT_DIR,
        help="Directory containing raw run_sim.py outputs. If present, these totals are used directly.",
    )
    parser.add_argument(
        "--seq_lens",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_LENS,
        help="Input sequence lengths to evaluate.",
    )
    parser.add_argument(
        "--output_seqlen",
        type=int,
        default=DEFAULT_OUTPUT_SEQLEN,
        help="Output sequence length for decode and memory-footprint calculations.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Single-chip batch size override. Defaults to 4 to match deliverable.py.",
    )
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = _load_model_config(args.model)
    chip_config = _load_chip_config(args.chip)

    rows: list[dict[str, float | int]] = []
    for seq_len in args.seq_lens:
        raw_metrics = _load_raw_metrics(
            args.raw_output_dir,
            seq_len,
            args.output_seqlen,
            args.batch_size,
        )
        if raw_metrics is not None:
            rows.append(
                {
                    "input_seqlen": seq_len,
                    "output_seqlen": args.output_seqlen,
                    "batch_size": args.batch_size,
                    **raw_metrics,
                }
            )
            continue

        ops_gen = _build_generator(
            model_config,
            chip_config,
            seq_len,
            args.output_seqlen,
            output_dir,
            args.batch_size,
        )

        prefill_ops = _simulate_ops(ops_gen.generate_prefill_ops(), chip_config)
        decode_ops = _simulate_ops(ops_gen.generate_decode_ops(), chip_config)

        prefill_flops = _sum_flops(prefill_ops)
        decode_flops = _sum_flops(decode_ops)
        prefill_bytes = _sum_bytes(prefill_ops)
        decode_bytes = _sum_bytes(decode_ops)
        prefill_mem = ops_gen.compute_memory_footprint_bytes("prefill")
        decode_mem = ops_gen.compute_memory_footprint_bytes("decode")

        rows.append(
            {
                "input_seqlen": seq_len,
                "output_seqlen": args.output_seqlen,
                "batch_size": ops_gen.batch_size,
                "prefill_mem_footprint_gb": prefill_mem / (1024**3),
                "decode_mem_footprint_gb": decode_mem / (1024**3),
                "prefill_memory_traffic_gb": prefill_bytes / 1e9,
                "decode_memory_traffic_gb": decode_bytes / 1e9,
                "prefill_flops_t": prefill_flops / 1e12,
                "decode_flops_t": decode_flops / 1e12,
                "prefill_ops": len(prefill_ops),
                "decode_ops": len(decode_ops),
            }
        )

    csv_path = output_dir / f"{SCRIPT_STEM}.csv"
    plot_path = output_dir / f"{SCRIPT_STEM}.png"
    _write_csv(rows, csv_path)
    _plot(rows, plot_path)

    print(f"Wrote {csv_path}")
    print(f"Wrote {plot_path}")


if __name__ == "__main__":
    main()
