#!/usr/bin/env python3
import csv
import json
import os
import sys
from math import ceil
from typing import Sequence

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
__OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "results/pd_disagg",
    "Output directory.",
)

def main(argv: Sequence[str]):
    del argv  # Unused.

    output_dir = os.path.join(ROOT, __OUTPUT_DIR.value)
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "pd_disagg.csv")
    prefill_csv = output_file.replace(".csv", "_prefill.csv")
    decode_csv = output_file.replace(".csv", "_decode.csv")
    summary_json = os.path.join(output_dir, "pd_disagg_summary.json")

    base_model_config = json.load(open(os.path.join(ROOT, __MODEL_CONFIG.value), "r"))
    base_npu_config = json.load(open(os.path.join(ROOT, __CHIP_CONFIG.value), "r"))
    base_sys_config = json.load(open(os.path.join(ROOT, "configs/systems/system_config.json"), "r"))

    base_config = {**base_model_config, **base_npu_config, **base_sys_config}
    model_name = base_config["model_name"]
    v = base_npu_config["name"]

    if "llama" in model_name.lower():
        gen_cls = LLMOpsGeneratorInference
        is_moe = False
    elif "deepseek" in model_name.lower():
        gen_cls = DeepSeekOpsGenerator
        is_moe = True
    elif "gpt-oss" in model_name.lower():
        gen_cls = GptOssOpsGenerator
        is_moe = True
    else:
        raise ValueError(f"Invalid model: {model_name}")

    prefill_cfg = {**base_config}
    prefill_cfg["num_chips"] = __PREFILL_TP.value * __PREFILL_PP.value
    prefill_cfg["data_parallelism_degree"] = 1
    prefill_cfg["tensor_parallelism_degree"] = __PREFILL_TP.value
    prefill_cfg["pipeline_parallelism_degree"] = __PREFILL_PP.value
    prefill_cfg["data_parallel_degree_dcn"] = 1
    prefill_cfg["tensor_parallel_degree_dcn"] = 1
    prefill_cfg["pipeline_parallel_degree_dcn"] = 1
    global_batch_size = __PREFILL_BATCH_SIZE.value
    dp_dcn = prefill_cfg["data_parallel_degree_dcn"]
    pp = prefill_cfg["pipeline_parallelism_degree"]
    pp_dcn = prefill_cfg["pipeline_parallel_degree_dcn"]
    microbatch_size_ici = ceil(global_batch_size / dp_dcn / pp / pp_dcn)
    microbatch_size_dcn = ceil(global_batch_size / pp_dcn)
    prefill_cfg["global_batch_size"] = global_batch_size
    prefill_cfg["microbatch_size_ici"] = microbatch_size_ici
    prefill_cfg["microbatch_size_dcn"] = microbatch_size_dcn
    if is_moe:
        prefill_cfg["expert_parallelism_degree"] = 1
        prefill_cfg["expert_parallel_degree_dcn"] = 1
    prefill_par_cfg = {
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": __PREFILL_TP.value,
        "pipeline_parallelism_degree": __PREFILL_PP.value,
        "expert_parallelism_degree": 1 if is_moe else None,
    }
    prefill_axes = run_sim_lib.map_parallelism_to_ici_axes(model_name, v, prefill_par_cfg)
    if is_moe:
        dp_axes, tp_axes, pp_axes, ep_axes = prefill_axes
        prefill_cfg["num_expert_parallel_axes"] = ep_axes
    else:
        dp_axes, tp_axes, pp_axes = prefill_axes
    prefill_cfg["num_data_parallel_axes"] = dp_axes
    prefill_cfg["num_tensor_parallel_axes"] = tp_axes
    prefill_cfg["num_pipeline_parallel_axes"] = pp_axes
    prefill_cfg["output_file_path"] = output_file

    decode_cfg = {**base_config}
    decode_cfg["num_chips"] = __DECODE_TP.value * __DECODE_PP.value
    decode_cfg["data_parallelism_degree"] = 1
    decode_cfg["tensor_parallelism_degree"] = __DECODE_TP.value
    decode_cfg["pipeline_parallelism_degree"] = __DECODE_PP.value
    decode_cfg["data_parallel_degree_dcn"] = 1
    decode_cfg["tensor_parallel_degree_dcn"] = 1
    decode_cfg["pipeline_parallel_degree_dcn"] = 1
    global_batch_size = __DECODE_BATCH_SIZE.value
    dp_dcn = decode_cfg["data_parallel_degree_dcn"]
    pp = decode_cfg["pipeline_parallelism_degree"]
    pp_dcn = decode_cfg["pipeline_parallel_degree_dcn"]
    microbatch_size_ici = ceil(global_batch_size / dp_dcn / pp / pp_dcn)
    microbatch_size_dcn = ceil(global_batch_size / pp_dcn)
    decode_cfg["global_batch_size"] = global_batch_size
    decode_cfg["microbatch_size_ici"] = microbatch_size_ici
    decode_cfg["microbatch_size_dcn"] = microbatch_size_dcn
    if is_moe:
        decode_cfg["expert_parallelism_degree"] = 1
        decode_cfg["expert_parallel_degree_dcn"] = 1
    decode_par_cfg = {
        "data_parallelism_degree": 1,
        "tensor_parallelism_degree": __DECODE_TP.value,
        "pipeline_parallelism_degree": __DECODE_PP.value,
        "expert_parallelism_degree": 1 if is_moe else None,
    }
    decode_axes = run_sim_lib.map_parallelism_to_ici_axes(model_name, v, decode_par_cfg)
    if is_moe:
        dp_axes, tp_axes, pp_axes, ep_axes = decode_axes
        decode_cfg["num_expert_parallel_axes"] = ep_axes
    else:
        dp_axes, tp_axes, pp_axes = decode_axes
    decode_cfg["num_data_parallel_axes"] = dp_axes
    decode_cfg["num_tensor_parallel_axes"] = tp_axes
    decode_cfg["num_pipeline_parallel_axes"] = pp_axes
    decode_cfg["output_file_path"] = output_file

    logging.info("Generating prefill ops")
    prefill_gen = gen_cls(prefill_cfg)
    prefill_ops = prefill_gen.generate_prefill_ops()
    prefill_ops = analysis_lib.fill_operators_execution_info(
        prefill_ops, prefill_gen.config, analyze_energy=False
    )
    prefill_ops_dict = [Operator.to_csv_dict(op) for op in prefill_ops]
    with open(prefill_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=prefill_ops_dict[0].keys())
        writer.writeheader()
        writer.writerows(prefill_ops_dict)
    prefill_stats = run_sim_lib.get_statistics_from_trace_file(prefill_csv)

    logging.info("Generating decode ops")
    decode_gen = gen_cls(decode_cfg)
    decode_ops = decode_gen.generate_decode_ops()
    decode_ops = analysis_lib.fill_operators_execution_info(
        decode_ops, decode_gen.config, analyze_energy=False
    )
    decode_ops_dict = [Operator.to_csv_dict(op) for op in decode_ops]
    with open(decode_csv, "w") as f:
        writer = csv.DictWriter(f, fieldnames=decode_ops_dict[0].keys())
        writer.writeheader()
        writer.writerows(decode_ops_dict)
    decode_stats = run_sim_lib.get_statistics_from_trace_file(decode_csv)

    global_batch_size = prefill_cfg["global_batch_size"]
    microbatch_size_ici = prefill_cfg["microbatch_size_ici"]
    pp_dcn = prefill_cfg["pipeline_parallel_degree_dcn"]
    dp_dcn = prefill_cfg["data_parallel_degree_dcn"]
    pp = prefill_cfg["pipeline_parallelism_degree"]
    num_pods = dp_dcn * pp_dcn
    batch_size_per_pod = ceil(global_batch_size / num_pods)
    prefill_cfg["num_pods"] = num_pods
    prefill_cfg["batch_size_per_pod"] = batch_size_per_pod
    total_pp = pp * pp_dcn
    layers_per_pp_stage = ceil(prefill_cfg["num_layers"] / total_pp)
    prefill_cfg["layers_per_pp_stage"] = layers_per_pp_stage
    input_seqlen = prefill_cfg["input_seqlen"]
    prefill_pp_stage_time_ns = ceil(prefill_stats["total_execution_time_chip_ns"])
    assert prefill_pp_stage_time_ns > 0, \
        f"csv files: {output_file.replace('.csv', '_prefill.csv')}, {output_file.replace('.csv', '_decode.csv')}"
    prefill_stats["throughput_tokens_per_sec"] = (
        microbatch_size_ici * input_seqlen * 1e9 / prefill_pp_stage_time_ns
    )
    prefill_pod_latency_ns = (
        prefill_stats["total_execution_time_non_pp_ns"] + prefill_stats["total_pp_ici_time_ns"]
    ) * pp
    prefill_tot_latency_ns = (prefill_pod_latency_ns + prefill_stats["total_pp_dcn_time_ns"]) * pp_dcn
    prefill_stats["TTFT_sec"] = prefill_tot_latency_ns / 1e9

    global_batch_size = decode_cfg["global_batch_size"]
    microbatch_size_ici = decode_cfg["microbatch_size_ici"]
    pp_dcn = decode_cfg["pipeline_parallel_degree_dcn"]
    dp_dcn = decode_cfg["data_parallel_degree_dcn"]
    pp = decode_cfg["pipeline_parallelism_degree"]
    num_pods = dp_dcn * pp_dcn
    batch_size_per_pod = ceil(global_batch_size / num_pods)
    decode_cfg["num_pods"] = num_pods
    decode_cfg["batch_size_per_pod"] = batch_size_per_pod
    total_pp = pp * pp_dcn
    layers_per_pp_stage = ceil(decode_cfg["num_layers"] / total_pp)
    decode_cfg["layers_per_pp_stage"] = layers_per_pp_stage
    output_seqlen = decode_cfg["output_seqlen"]
    decode_pod_latency_ns = (
        decode_stats["total_execution_time_non_pp_ns"] + decode_stats["total_pp_ici_time_ns"]
    ) * pp / output_seqlen
    decode_tot_latency_ns = (
        decode_pod_latency_ns + decode_stats["total_pp_dcn_time_ns"] / output_seqlen
    ) * pp_dcn
    decode_pp_stage_time_ns = ceil(decode_stats["total_execution_time_chip_ns"] / output_seqlen)
    assert decode_stats["total_execution_time_chip_ns"] > 0, \
        f"csv files: {output_file.replace('.csv', '_prefill.csv')}, {output_file.replace('.csv', '_decode.csv')}, stats: {decode_stats}"
    decode_stats["TPOT_ms_request"] = decode_tot_latency_ns / 1e6
    decode_stats["throughput_tokens_per_sec"] = microbatch_size_ici * 1e9 / decode_pp_stage_time_ns
    decode_stats["throughput_tokens_per_sec_request"] = 1e3 / decode_stats["TPOT_ms_request"]

    kv_cache_bytes = mem_footprint_lib.get_llm_inference_kv_cache_mem_requirement(
        prefill_gen.config, "prefill"
    )
    transfer_size = kv_cache_bytes
    kv_transfer_time_ns = max(
        ceil(transfer_size * 1e9 / 1024 / 1024 / 1024 / prefill_gen.config.pcie_bw_GBps),
        prefill_gen.config.pcie_latency_ns,
    )
    kv_transfer_time_sec = kv_transfer_time_ns / 1e9

    prefill_chip_budget = __TOTAL_CHIPS.value // 2
    decode_chip_budget = __TOTAL_CHIPS.value - prefill_chip_budget
    num_prefill_instances = prefill_chip_budget // prefill_cfg["num_chips"]
    num_decode_instances = decode_chip_budget // decode_cfg["num_chips"]
    prefill_qps = prefill_stats["throughput_tokens_per_sec"] / input_seqlen
    decode_qps = decode_stats["throughput_tokens_per_sec"] / output_seqlen
    system_throughput = min(
        num_prefill_instances * prefill_qps,
        num_decode_instances * decode_qps,
    )

    summary = {
        "model_name": model_name,
        "chip_name": v,
        "ttft_sec_without_kv_transfer": prefill_stats["TTFT_sec"],
        "kv_cache_transfer_time_sec": kv_transfer_time_sec,
        "ttft_sec_with_kv_transfer": prefill_stats["TTFT_sec"] + kv_transfer_time_sec,
        "tpot_ms_request": decode_stats["TPOT_ms_request"],
        "prefill_throughput_tokens_per_sec_per_instance": prefill_stats["throughput_tokens_per_sec"],
        "decode_throughput_tokens_per_sec_per_instance": decode_stats["throughput_tokens_per_sec"],
        "num_prefill_instances": num_prefill_instances,
        "num_decode_instances": num_decode_instances,
        "system_throughput_queries_per_sec": system_throughput,
    }

    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=4)


if __name__ == "__main__":
    app.run(main)
