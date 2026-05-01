#!/usr/bin/env python3

"""Run MP4 4.1 prefill/decode disaggregation experiments."""

from __future__ import annotations

import contextlib
import csv
from copy import deepcopy
import io
import json
import os
import sys
from math import ceil
from typing import Any, Sequence

from absl import app, flags, logging

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, ROOT)

import neusim.npusim.frontend.Operator as Operator
import neusim.npusim.frontend.memory_footprint_analysis_lib as mem_footprint_lib
import neusim.npusim.frontend.op_analysis_lib as analysis_lib
import neusim.npusim.frontend.run_sim_lib as run_sim_lib
from neusim.npusim.frontend.llm_ops_generator import (
    DeepSeekOpsGenerator,
    GptOssOpsGenerator,
    LLMOpsGeneratorInference,
)

__MODEL_CONFIG = flags.DEFINE_string(
    "model_config",
    "configs/models/llama3-8b.json",
    "Path to the model config JSON.",
)
__CHIP_CONFIG = flags.DEFINE_string(
    "chip_config",
    "configs/chips/tpuv5p.json",
    "Path to the chip config JSON.",
)
__PREFILL_TP = flags.DEFINE_integer(
    "prefill_tp",
    1,
    "Prefill tensor parallelism degree.",
)
__PREFILL_PP = flags.DEFINE_integer(
    "prefill_pp",
    1,
    "Prefill pipeline parallelism degree.",
)
__PREFILL_BATCH_SIZE = flags.DEFINE_integer(
    "prefill_batch_size",
    1,
    "Prefill batch size.",
)
__DECODE_TP = flags.DEFINE_integer(
    "decode_tp",
    1,
    "Decode tensor parallelism degree.",
)
__DECODE_PP = flags.DEFINE_integer(
    "decode_pp",
    1,
    "Decode pipeline parallelism degree.",
)
__DECODE_BATCH_SIZE = flags.DEFINE_integer(
    "decode_batch_size",
    1,
    "Decode batch size.",
)
__TOTAL_CHIPS = flags.DEFINE_integer(
    "total_chips",
    16,
    "Total number of chips to split between prefill and decode.",
)
__PREFILL_POOL_CHIPS = flags.DEFINE_integer(
    "prefill_pool_chips",
    -1,
    "Chips allocated to the prefill pool. Defaults to half of total_chips.",
)
__DECODE_POOL_CHIPS = flags.DEFINE_integer(
    "decode_pool_chips",
    -1,
    "Chips allocated to the decode pool. Defaults to the remainder after prefill allocation.",
)
__OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "results/pd_disagg",
    "Output directory.",
)

def _resolve_path(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _load_json(path: str) -> dict[str, Any]:
    with open(_resolve_path(path), "r") as f:
        return json.load(f)


def _resolve_output_dir(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(ROOT, path)


def _get_generator_class(model_name: str) -> tuple[type, bool]:
    model_name = model_name.lower()
    if "llama" in model_name:
        return LLMOpsGeneratorInference, False
    if "deepseek" in model_name:
        return DeepSeekOpsGenerator, True
    if "gpt-oss" in model_name:
        return GptOssOpsGenerator, True
    raise ValueError(f"Invalid model: {model_name}")


def _build_phase_config(
    base_config: dict[str, Any],
    model_name: str,
    chip_name: str,
    *,
    tp: int,
    pp: int,
    batch_size: int,
    output_file: str,
    is_moe: bool,
) -> dict[str, Any]:
    config = dict(base_config)
    config["num_chips"] = tp * pp
    config["data_parallelism_degree"] = 1
    config["tensor_parallelism_degree"] = tp
    config["pipeline_parallelism_degree"] = pp
    config["data_parallel_degree_dcn"] = 1
    config["tensor_parallel_degree_dcn"] = 1
    config["pipeline_parallel_degree_dcn"] = 1
    config["global_batch_size"] = batch_size
    config["microbatch_size_ici"] = ceil(batch_size / pp)
    config["microbatch_size_dcn"] = batch_size
    config["output_file_path"] = output_file

    parallelism_config = {
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": tp,
        "pipeline_parallelism_degree": pp,
    }
    if is_moe:
        config["expert_parallelism_degree"] = 1
        config["expert_parallel_degree_dcn"] = 1
        parallelism_config["expert_parallelism_degree"] = 1

    axes = run_sim_lib.map_parallelism_to_ici_axes(model_name, chip_name, parallelism_config)
    if is_moe:
        dp_axes, tp_axes, pp_axes, ep_axes = axes
        config["num_expert_parallel_axes"] = ep_axes
    else:
        dp_axes, tp_axes, pp_axes = axes
    config["num_data_parallel_axes"] = dp_axes
    config["num_tensor_parallel_axes"] = tp_axes
    config["num_pipeline_parallel_axes"] = pp_axes
    return config


def _validate_phase_config(
    phase_name: str,
    model_name: str,
    chip_name: str,
    config: dict[str, Any],
) -> None:
    is_valid = run_sim_lib.validate_parallelism_config(
        model_name,
        chip_name,
        config,
        workload="inference",
        allow_oom=1.0,
        prefill_or_decode=phase_name,
    )
    if not is_valid:
        raise ValueError(
            f"Invalid {phase_name} configuration for {model_name} on TPUv{chip_name}: "
            f"tp={config['tensor_parallelism_degree']}, "
            f"pp={config['pipeline_parallelism_degree']}, "
            f"batch_size={config['global_batch_size']}."
        )


def _write_ops_csv(csv_path: str, ops: list[Operator.Operator]) -> None:
    ops_dict = [Operator.to_csv_dict(op) for op in ops]
    if not ops_dict:
        raise ValueError(f"No operators were generated for {csv_path}.")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ops_dict[0].keys())
        writer.writeheader()
        writer.writerows(ops_dict)


def _run_phase(
    phase_name: str,
    gen_cls: type,
    config: dict[str, Any],
    csv_path: str,
) -> tuple[Any, dict[str, Any]]:
    logging.info("Generating %s ops", phase_name)
    generator = gen_cls(config)
    if phase_name == "prefill":
        ops = generator.generate_prefill_ops()
    elif phase_name == "decode":
        ops = generator.generate_decode_ops()
    else:
        raise ValueError(f"Invalid phase: {phase_name}")

    # The analysis helper prints a per-op trace; suppress it to keep runs readable.
    with contextlib.redirect_stdout(io.StringIO()):
        ops = analysis_lib.fill_operators_execution_info(
            ops,
            generator.config,
            analyze_energy=False,
        )

    _write_ops_csv(csv_path, ops)
    stats = run_sim_lib.get_statistics_from_trace_file(csv_path)
    return generator, stats


def _annotate_phase_config(config: dict[str, Any]) -> None:
    num_pods = config["data_parallel_degree_dcn"] * config["pipeline_parallel_degree_dcn"]
    config["num_pods"] = num_pods
    config["batch_size_per_pod"] = ceil(config["global_batch_size"] / num_pods)
    total_pp = config["pipeline_parallelism_degree"] * config["pipeline_parallel_degree_dcn"]
    config["layers_per_pp_stage"] = ceil(config["num_layers"] / total_pp)


def _populate_prefill_metrics(config: dict[str, Any], stats: dict[str, Any]) -> None:
    _annotate_phase_config(config)
    prefill_pp_stage_time_ns = ceil(stats["total_execution_time_chip_ns"])
    if prefill_pp_stage_time_ns <= 0:
        raise ValueError("Prefill execution time must be positive.")

    microbatch_size_ici = config["microbatch_size_ici"]
    input_seqlen = config["input_seqlen"]
    pp = config["pipeline_parallelism_degree"]
    pp_dcn = config["pipeline_parallel_degree_dcn"]

    stats["throughput_tokens_per_sec"] = (
        microbatch_size_ici * input_seqlen * 1e9 / prefill_pp_stage_time_ns
    )
    prefill_pod_latency_ns = (
        stats["total_execution_time_non_pp_ns"] + stats["total_pp_ici_time_ns"]
    ) * pp
    prefill_tot_latency_ns = (
        prefill_pod_latency_ns + stats["total_pp_dcn_time_ns"]
    ) * pp_dcn
    stats["TTFT_sec"] = prefill_tot_latency_ns / 1e9
    stats["throughput_queries_per_sec"] = (
        stats["throughput_tokens_per_sec"] / input_seqlen
    )


def _populate_decode_metrics(config: dict[str, Any], stats: dict[str, Any]) -> None:
    _annotate_phase_config(config)
    output_seqlen = config["output_seqlen"]
    decode_pp_stage_time_ns = ceil(stats["total_execution_time_chip_ns"] / output_seqlen)
    if decode_pp_stage_time_ns <= 0:
        raise ValueError("Decode execution time must be positive.")

    microbatch_size_ici = config["microbatch_size_ici"]
    pp = config["pipeline_parallelism_degree"]
    pp_dcn = config["pipeline_parallel_degree_dcn"]

    decode_pod_latency_ns = (
        stats["total_execution_time_non_pp_ns"] + stats["total_pp_ici_time_ns"]
    ) * pp / output_seqlen
    decode_tot_latency_ns = (
        decode_pod_latency_ns + stats["total_pp_dcn_time_ns"] / output_seqlen
    ) * pp_dcn

    stats["TPOT_ms_request"] = decode_tot_latency_ns / 1e6
    stats["throughput_tokens_per_sec"] = microbatch_size_ici * 1e9 / decode_pp_stage_time_ns
    stats["throughput_tokens_per_sec_request"] = 1e3 / stats["TPOT_ms_request"]
    stats["throughput_queries_per_sec"] = (
        stats["throughput_tokens_per_sec"] / output_seqlen
    )


def _get_unsharded_kv_cache_bytes(config: dict[str, Any]) -> int:
    kv_config = deepcopy(config)
    kv_config["tensor_parallelism_degree"] = 1
    kv_config["tensor_parallel_degree_dcn"] = 1
    return mem_footprint_lib.get_llm_inference_kv_cache_mem_requirement(
        kv_config,
        "prefill",
    )


def compute_kv_cache_transfer_time_ms(config: dict[str, Any], dcn_bw_GBps: float) -> float:
    if dcn_bw_GBps <= 0:
        raise ValueError(f"DCN bandwidth must be positive, got {dcn_bw_GBps}.")
    kv_cache_bytes = _get_unsharded_kv_cache_bytes(config)
    dcn_bw_bytes_per_sec = dcn_bw_GBps * 1024 * 1024 * 1024
    return kv_cache_bytes * 1e3 / dcn_bw_bytes_per_sec


def _write_phase_stats_json(path: str, config: dict[str, Any], stats: dict[str, Any]) -> None:
    payload = dict(stats)
    payload["sim_config"] = config
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def _resolve_pool_chip_budgets(
    total_chips: int,
    requested_prefill_chips: int,
    requested_decode_chips: int,
) -> tuple[int, int]:
    if total_chips <= 0:
        raise ValueError(f"total_chips must be positive, got {total_chips}.")

    if requested_prefill_chips < 0 and requested_decode_chips < 0:
        prefill_chips = total_chips // 2
        decode_chips = total_chips - prefill_chips
    elif requested_prefill_chips >= 0 and requested_decode_chips < 0:
        prefill_chips = requested_prefill_chips
        decode_chips = total_chips - prefill_chips
    elif requested_prefill_chips < 0 and requested_decode_chips >= 0:
        decode_chips = requested_decode_chips
        prefill_chips = total_chips - decode_chips
    else:
        prefill_chips = requested_prefill_chips
        decode_chips = requested_decode_chips

    if prefill_chips < 0 or decode_chips < 0:
        raise ValueError(
            f"Invalid pool allocation: prefill={prefill_chips}, decode={decode_chips}, "
            f"total={total_chips}."
        )
    if prefill_chips + decode_chips != total_chips:
        raise ValueError(
            f"Pool chips must sum to total_chips: prefill={prefill_chips}, "
            f"decode={decode_chips}, total={total_chips}."
        )

    return prefill_chips, decode_chips


def main(argv: Sequence[str]) -> None:
    del argv

    total_chips = __TOTAL_CHIPS.value
    prefill_chip_budget, decode_chip_budget = _resolve_pool_chip_budgets(
        total_chips,
        __PREFILL_POOL_CHIPS.value,
        __DECODE_POOL_CHIPS.value,
    )

    output_dir = _resolve_output_dir(__OUTPUT_DIR.value)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pd_disagg.csv")
    prefill_csv = output_file.replace(".csv", "_prefill.csv")
    decode_csv = output_file.replace(".csv", "_decode.csv")
    prefill_stats_json = output_file.replace(".csv", "_prefill.json")
    decode_stats_json = output_file.replace(".csv", "_decode.json")
    summary_json = os.path.join(output_dir, "pd_disagg_summary.json")

    base_model_config = _load_json(__MODEL_CONFIG.value)
    base_npu_config = _load_json(__CHIP_CONFIG.value)
    base_sys_config = _load_json("configs/systems/system_config.json")
    base_config = {**base_model_config, **base_npu_config, **base_sys_config}

    model_name = base_config["model_name"]
    chip_name = base_npu_config["name"]
    gen_cls, is_moe = _get_generator_class(model_name)

    prefill_cfg = _build_phase_config(
        base_config,
        model_name,
        chip_name,
        tp=__PREFILL_TP.value,
        pp=__PREFILL_PP.value,
        batch_size=__PREFILL_BATCH_SIZE.value,
        output_file=output_file,
        is_moe=is_moe,
    )
    decode_cfg = _build_phase_config(
        base_config,
        model_name,
        chip_name,
        tp=__DECODE_TP.value,
        pp=__DECODE_PP.value,
        batch_size=__DECODE_BATCH_SIZE.value,
        output_file=output_file,
        is_moe=is_moe,
    )

    _validate_phase_config("prefill", model_name, chip_name, prefill_cfg)
    _validate_phase_config("decode", model_name, chip_name, decode_cfg)

    _, prefill_stats = _run_phase("prefill", gen_cls, prefill_cfg, prefill_csv)
    _, decode_stats = _run_phase("decode", gen_cls, decode_cfg, decode_csv)

    _populate_prefill_metrics(prefill_cfg, prefill_stats)
    _populate_decode_metrics(decode_cfg, decode_stats)

    kv_cache_bytes = _get_unsharded_kv_cache_bytes(prefill_cfg)
    kv_transfer_time_ms = compute_kv_cache_transfer_time_ms(
        prefill_cfg,
        prefill_cfg["dcn_bw_GBps"],
    )
    kv_transfer_time_sec = kv_transfer_time_ms / 1e3

    num_prefill_instances = prefill_chip_budget // prefill_cfg["num_chips"]
    num_decode_instances = decode_chip_budget // decode_cfg["num_chips"]
    system_throughput = min(
        num_prefill_instances * prefill_stats["throughput_queries_per_sec"],
        num_decode_instances * decode_stats["throughput_queries_per_sec"],
    )

    _write_phase_stats_json(prefill_stats_json, prefill_cfg, prefill_stats)
    _write_phase_stats_json(decode_stats_json, decode_cfg, decode_stats)

    summary = {
        "model_name": model_name,
        "chip_name": chip_name,
        "prefill_chip_budget": prefill_chip_budget,
        "decode_chip_budget": decode_chip_budget,
        "ttft_sec_without_kv_transfer": prefill_stats["TTFT_sec"],
        "kv_cache_transfer_bytes": kv_cache_bytes,
        "kv_cache_transfer_time_ms": kv_transfer_time_ms,
        "kv_cache_transfer_time_sec": kv_transfer_time_sec,
        "ttft_sec_with_kv_transfer": prefill_stats["TTFT_sec"] + kv_transfer_time_sec,
        "tpot_ms_request": decode_stats["TPOT_ms_request"],
        "prefill_throughput_tokens_per_sec_per_instance": prefill_stats["throughput_tokens_per_sec"],
        "decode_throughput_tokens_per_sec_per_instance": decode_stats["throughput_tokens_per_sec"],
        "prefill_throughput_queries_per_sec_per_instance": prefill_stats["throughput_queries_per_sec"],
        "decode_throughput_queries_per_sec_per_instance": decode_stats["throughput_queries_per_sec"],
        "num_prefill_instances": num_prefill_instances,
        "num_decode_instances": num_decode_instances,
        "system_throughput_queries_per_sec": system_throughput,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    app.run(main)
