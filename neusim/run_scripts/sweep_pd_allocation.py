#!/usr/bin/env python3

"""Run the MP4 4.2.2 chip-allocation sweep for PD disaggregation."""

from __future__ import annotations

import argparse
import csv
import json
import os
from math import isclose
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

import run_pd_disagg


DEFAULT_MODEL = REPO_ROOT / "configs/models/gpt-oss-120b.json"
DEFAULT_CHIP = REPO_ROOT / "configs/chips/tpuv5p.json"
DEFAULT_PART3_CSV = REPO_ROOT / "MP4 Part 3.2 Throughput/parallelism_sweep.csv"
DEFAULT_OUTPUT_DIR = REPO_ROOT / f"results/{SCRIPT_STEM}"
DEFAULT_TOTAL_CHIPS = 256
DEFAULT_INPUT_SEQLEN = 2048
DEFAULT_OUTPUT_SEQLEN = 128


def _load_json(path: Path) -> dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def _load_part3_rows(path: Path, model_name: str) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    filtered_rows = [row for row in rows if row["model"] == model_name]
    if not filtered_rows:
        raise ValueError(f"No Part III throughput rows found for model {model_name}.")
    return filtered_rows


def _pick_best_config(
    rows: list[dict[str, str]],
    metric: str,
) -> dict[str, Any]:
    best_row = max(rows, key=lambda row: float(row[metric]))
    # The saved Part III artifact uses `microbatch_size` as the experimental batch-size field.
    return {
        "tp": int(best_row["tp"]),
        "pp": int(best_row["pp"]),
        "batch_size": int(best_row["microbatch_size"]),
        "selection_metric": metric,
        "selection_metric_value": float(best_row[metric]),
        "part3_row": {
            "model": best_row["model"],
            "model_display_name": best_row.get("model_display_name", best_row["model"]),
            "microbatch_size": int(best_row["microbatch_size"]),
            "tp": int(best_row["tp"]),
            "pp": int(best_row["pp"]),
            "pf_stage_ms": float(best_row["pf_stage_ms"]),
            "dc_stage_ms": float(best_row["dc_stage_ms"]),
            "ttft_ms": float(best_row["ttft_ms"]),
            "tpot_ms": float(best_row["tpot_ms"]),
            "pf_ici_bytes": float(best_row["pf_ici_bytes"]),
            "dc_ici_bytes": float(best_row["dc_ici_bytes"]),
            "ss_pf_tps": float(best_row["ss_pf_tps"]),
            "ss_dc_tps": float(best_row["ss_dc_tps"]),
            "sr_pf_tps": float(best_row["sr_pf_tps"]),
            "sr_dc_tps": float(best_row["sr_dc_tps"]),
        },
    }


def _simulate_selected_configs(
    model_config_path: Path,
    chip_config_path: Path,
    output_dir: Path,
    *,
    prefill_choice: dict[str, Any],
    decode_choice: dict[str, Any],
    input_seqlen: int,
    output_seqlen: int,
) -> dict[str, Any]:
    base_model_config = _load_json(model_config_path)
    base_chip_config = _load_json(chip_config_path)
    base_system_config = _load_json(REPO_ROOT / "configs/systems/system_config.json")
    base_config = {**base_model_config, **base_chip_config, **base_system_config}
    base_config["input_seqlen"] = input_seqlen
    base_config["output_seqlen"] = output_seqlen

    model_name = base_config["model_name"]
    chip_name = base_chip_config["name"]
    gen_cls, is_moe = run_pd_disagg._get_generator_class(model_name)

    shared_output_file = str(output_dir / f"{SCRIPT_STEM}.csv")
    prefill_csv = output_dir / f"{SCRIPT_STEM}_prefill.csv"
    decode_csv = output_dir / f"{SCRIPT_STEM}_decode.csv"
    prefill_json = output_dir / f"{SCRIPT_STEM}_prefill.json"
    decode_json = output_dir / f"{SCRIPT_STEM}_decode.json"

    prefill_config = run_pd_disagg._build_phase_config(
        base_config,
        model_name,
        chip_name,
        tp=prefill_choice["tp"],
        pp=prefill_choice["pp"],
        batch_size=prefill_choice["batch_size"],
        output_file=shared_output_file,
        is_moe=is_moe,
    )
    decode_config = run_pd_disagg._build_phase_config(
        base_config,
        model_name,
        chip_name,
        tp=decode_choice["tp"],
        pp=decode_choice["pp"],
        batch_size=decode_choice["batch_size"],
        output_file=shared_output_file,
        is_moe=is_moe,
    )

    run_pd_disagg._validate_phase_config("prefill", model_name, chip_name, prefill_config)
    run_pd_disagg._validate_phase_config("decode", model_name, chip_name, decode_config)

    _, prefill_stats = run_pd_disagg._run_phase(
        "prefill",
        gen_cls,
        prefill_config,
        str(prefill_csv),
    )
    _, decode_stats = run_pd_disagg._run_phase(
        "decode",
        gen_cls,
        decode_config,
        str(decode_csv),
    )

    run_pd_disagg._populate_prefill_metrics(prefill_config, prefill_stats)
    run_pd_disagg._populate_decode_metrics(decode_config, decode_stats)

    run_pd_disagg._write_phase_stats_json(str(prefill_json), prefill_config, prefill_stats)
    run_pd_disagg._write_phase_stats_json(str(decode_json), decode_config, decode_stats)

    kv_cache_transfer_bytes = run_pd_disagg._get_unsharded_kv_cache_bytes(prefill_config)
    kv_cache_transfer_time_ms = run_pd_disagg.compute_kv_cache_transfer_time_ms(
        prefill_config,
        prefill_config["dcn_bw_GBps"],
    )
    kv_cache_transfer_time_sec = kv_cache_transfer_time_ms / 1e3
    prefill_time_sec_without_kv_transfer = prefill_stats["TTFT_sec"]
    prefill_time_sec_with_kv_transfer = prefill_time_sec_without_kv_transfer + kv_cache_transfer_time_sec
    decode_time_sec = decode_stats["TPOT_ms_request"] * decode_config["output_seqlen"] / 1e3

    return {
        "model_name": model_name,
        "chip_name": chip_name,
        "prefill_config": prefill_config,
        "decode_config": decode_config,
        "prefill_stats": prefill_stats,
        "decode_stats": decode_stats,
        "kv_cache_transfer_bytes": kv_cache_transfer_bytes,
        "kv_cache_transfer_time_ms": kv_cache_transfer_time_ms,
        "kv_cache_transfer_time_sec": kv_cache_transfer_time_sec,
        "prefill_time_sec_without_kv_transfer": prefill_time_sec_without_kv_transfer,
        "prefill_time_sec_with_kv_transfer": prefill_time_sec_with_kv_transfer,
        "decode_time_sec": decode_time_sec,
        "prefill_request_qps_without_kv_transfer": (
            prefill_config["global_batch_size"] / prefill_time_sec_without_kv_transfer
        ),
        "prefill_request_qps_with_kv_transfer": (
            prefill_config["global_batch_size"] / prefill_time_sec_with_kv_transfer
        ),
        "decode_request_qps": decode_config["global_batch_size"] / decode_time_sec,
    }


def _enumerate_valid_allocations(
    total_chips: int,
    prefill_instance_chips: int,
    decode_instance_chips: int,
) -> list[tuple[int, int]]:
    allocations: list[tuple[int, int]] = []
    for prefill_pool_chips in range(total_chips + 1):
        decode_pool_chips = total_chips - prefill_pool_chips
        if prefill_pool_chips % prefill_instance_chips != 0:
            continue
        if decode_pool_chips % decode_instance_chips != 0:
            continue
        allocations.append((prefill_pool_chips, decode_pool_chips))
    if not allocations:
        raise ValueError(
            "No valid allocations found for total_chips="
            f"{total_chips}, prefill_instance_chips={prefill_instance_chips}, "
            f"decode_instance_chips={decode_instance_chips}."
        )
    return allocations


def _bottleneck_name(left: float, right: float) -> str:
    if isclose(left, right, rel_tol=1e-9, abs_tol=1e-12):
        return "balanced"
    return "prefill" if left < right else "decode"


def _build_sweep_rows(
    metrics: dict[str, Any],
    total_chips: int,
) -> list[dict[str, float | int | str]]:
    prefill_instance_chips = metrics["prefill_config"]["num_chips"]
    decode_instance_chips = metrics["decode_config"]["num_chips"]
    rows: list[dict[str, float | int | str]] = []

    for prefill_pool_chips, decode_pool_chips in _enumerate_valid_allocations(
        total_chips,
        prefill_instance_chips,
        decode_instance_chips,
    ):
        num_prefill_instances = prefill_pool_chips // prefill_instance_chips
        num_decode_instances = decode_pool_chips // decode_instance_chips

        prefill_capacity_without_kv = (
            num_prefill_instances * metrics["prefill_request_qps_without_kv_transfer"]
        )
        prefill_capacity_with_kv = (
            num_prefill_instances * metrics["prefill_request_qps_with_kv_transfer"]
        )
        decode_capacity = num_decode_instances * metrics["decode_request_qps"]
        system_throughput_without_kv = min(prefill_capacity_without_kv, decode_capacity)
        system_throughput_with_kv = min(prefill_capacity_with_kv, decode_capacity)

        rows.append(
            {
                "prefill_pool_chips": prefill_pool_chips,
                "decode_pool_chips": decode_pool_chips,
                "num_prefill_instances": num_prefill_instances,
                "num_decode_instances": num_decode_instances,
                "prefill_instance_chips": prefill_instance_chips,
                "decode_instance_chips": decode_instance_chips,
                "prefill_capacity_qps_without_kv_transfer": round(prefill_capacity_without_kv, 6),
                "prefill_capacity_qps_with_kv_transfer": round(prefill_capacity_with_kv, 6),
                "decode_capacity_qps": round(decode_capacity, 6),
                "system_throughput_qps_without_kv_transfer": round(
                    system_throughput_without_kv,
                    6,
                ),
                "system_throughput_qps_with_kv_transfer": round(
                    system_throughput_with_kv,
                    6,
                ),
                "bottleneck_without_kv_transfer": _bottleneck_name(
                    prefill_capacity_without_kv,
                    decode_capacity,
                ),
                "bottleneck_with_kv_transfer": _bottleneck_name(
                    prefill_capacity_with_kv,
                    decode_capacity,
                ),
            }
        )

    return rows


def _write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows: list[dict[str, float | int | str]], output_path: Path) -> None:
    mpl_config_dir = Path(tempfile.gettempdir()) / "neusim-mpl-cache"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    import matplotlib.pyplot as plt

    prefill_pool_chips = [int(row["prefill_pool_chips"]) for row in rows]
    prefill_capacity = [float(row["prefill_capacity_qps_with_kv_transfer"]) for row in rows]
    decode_capacity = [float(row["decode_capacity_qps"]) for row in rows]
    system_throughput = [float(row["system_throughput_qps_with_kv_transfer"]) for row in rows]
    best_throughput = max(system_throughput)
    best_indices = [
        idx for idx, value in enumerate(system_throughput)
        if isclose(value, best_throughput, rel_tol=1e-9, abs_tol=1e-12)
    ]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(prefill_pool_chips, prefill_capacity, marker="o", linewidth=2, label="Prefill capacity")
    ax.plot(prefill_pool_chips, decode_capacity, marker="s", linewidth=2, label="Decode capacity")
    ax.plot(
        prefill_pool_chips,
        system_throughput,
        marker="D",
        linewidth=2.5,
        color="black",
        label="System throughput",
    )

    for idx in best_indices:
        ax.scatter(
            prefill_pool_chips[idx],
            system_throughput[idx],
            color="#d62728",
            s=70,
            zorder=5,
        )

    ax.set_title("PD Allocation Sweep for GPT-OSS-120B (including KV transfer)")
    ax.set_xlabel("Prefill Pool Chips (N_P)")
    ax.set_ylabel("Request Throughput (queries/s)")
    ax.set_xticks(prefill_pool_chips)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _pick_best_allocations(
    rows: list[dict[str, float | int | str]],
    throughput_key: str,
) -> list[dict[str, float | int | str]]:
    best_value = max(float(row[throughput_key]) for row in rows)
    return [
        row for row in rows
        if isclose(float(row[throughput_key]), best_value, rel_tol=1e-9, abs_tol=1e-12)
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MP4 4.2.2 chip-allocation sweep for PD disaggregation."
    )
    parser.add_argument("--model_config", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--chip_config", type=Path, default=DEFAULT_CHIP)
    parser.add_argument("--part3_csv", type=Path, default=DEFAULT_PART3_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--total_chips", type=int, default=DEFAULT_TOTAL_CHIPS)
    parser.add_argument("--input_seqlen", type=int, default=DEFAULT_INPUT_SEQLEN)
    parser.add_argument("--output_seqlen", type=int, default=DEFAULT_OUTPUT_SEQLEN)
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = _load_json(args.model_config)
    model_name = model_config["model_name"]
    part3_rows = _load_part3_rows(args.part3_csv, model_name)
    prefill_choice = _pick_best_config(part3_rows, "sr_pf_tps")
    decode_choice = _pick_best_config(part3_rows, "sr_dc_tps")

    metrics = _simulate_selected_configs(
        args.model_config,
        args.chip_config,
        output_dir,
        prefill_choice=prefill_choice,
        decode_choice=decode_choice,
        input_seqlen=args.input_seqlen,
        output_seqlen=args.output_seqlen,
    )
    rows = _build_sweep_rows(metrics, args.total_chips)

    csv_path = output_dir / f"{SCRIPT_STEM}.csv"
    plot_path = output_dir / f"{SCRIPT_STEM}.png"
    summary_path = output_dir / f"{SCRIPT_STEM}_summary.json"
    _write_csv(rows, csv_path)
    _plot(rows, plot_path)

    best_with_kv = _pick_best_allocations(rows, "system_throughput_qps_with_kv_transfer")
    best_without_kv = _pick_best_allocations(rows, "system_throughput_qps_without_kv_transfer")
    summary = {
        "model_name": metrics["model_name"],
        "chip_name": metrics["chip_name"],
        "input_seqlen": args.input_seqlen,
        "output_seqlen": args.output_seqlen,
        "total_chips": args.total_chips,
        "selected_prefill_config_from_part3": {
            **prefill_choice,
            "derived_request_qps_from_part3": (
                prefill_choice["part3_row"]["sr_pf_tps"] / args.input_seqlen
            ),
        },
        "selected_decode_config_from_part3": {
            **decode_choice,
            "derived_request_qps_from_part3": (
                decode_choice["part3_row"]["sr_dc_tps"] / args.output_seqlen
            ),
        },
        "prefill_instance_metrics": {
            "tp": metrics["prefill_config"]["tensor_parallelism_degree"],
            "pp": metrics["prefill_config"]["pipeline_parallelism_degree"],
            "batch_size": metrics["prefill_config"]["global_batch_size"],
            "chips_per_instance": metrics["prefill_config"]["num_chips"],
            "ttft_sec_without_kv_transfer": metrics["prefill_time_sec_without_kv_transfer"],
            "ttft_sec_with_kv_transfer": metrics["prefill_time_sec_with_kv_transfer"],
            "kv_cache_transfer_bytes": metrics["kv_cache_transfer_bytes"],
            "kv_cache_transfer_time_ms": metrics["kv_cache_transfer_time_ms"],
            "request_qps_without_kv_transfer": metrics["prefill_request_qps_without_kv_transfer"],
            "request_qps_with_kv_transfer": metrics["prefill_request_qps_with_kv_transfer"],
        },
        "decode_instance_metrics": {
            "tp": metrics["decode_config"]["tensor_parallelism_degree"],
            "pp": metrics["decode_config"]["pipeline_parallelism_degree"],
            "batch_size": metrics["decode_config"]["global_batch_size"],
            "chips_per_instance": metrics["decode_config"]["num_chips"],
            "tpot_ms_request": metrics["decode_stats"]["TPOT_ms_request"],
            "decode_time_sec": metrics["decode_time_sec"],
            "request_qps": metrics["decode_request_qps"],
        },
        "best_allocations_with_kv_transfer": best_with_kv,
        "best_allocations_without_kv_transfer": best_without_kv,
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
