#!/usr/bin/env python3
"""Standalone regression runner for NeuSim simulations.

Runs a set of simulation experiments without absl flags, accepting configuration
via --config JSON argument. Designed to be invoked both in-process and via
subprocess (for baseline comparisons in a separate venv/worktree).
"""

import argparse
import json
import os
from math import ceil
from typing import Any


def _load_configs(configs_path: str, model: str, chip_version: str) -> dict[str, Any]:
    """Load and merge model, chip, and system configs from JSON files."""
    model_config_path = os.path.join(configs_path, f"models/{model}.json")
    chip_config_path = os.path.join(configs_path, f"chips/tpuv{chip_version}.json")
    sys_config_path = os.path.join(configs_path, "systems/system_config.json")

    with open(model_config_path) as f:
        model_config = json.load(f)
    with open(chip_config_path) as f:
        chip_config = json.load(f)
    with open(sys_config_path) as f:
        sys_config = json.load(f)

    return {**model_config, **chip_config, **sys_config}


def _run_single_experiment(
    experiment: dict[str, Any],
    configs_path: str,
    output_dir: str,
) -> list[str]:
    """Run a single simulation experiment and return list of output file paths."""
    # Lazy imports so this module can be loaded without neusim installed
    import neusim.npusim.frontend.run_sim_lib as run_sim_lib
    from neusim.npusim.frontend.dit_ops_generator import DiTOpsGenerator
    from neusim.npusim.frontend.dlrm_ops_generator import DLRMOpsGenerator
    from neusim.npusim.frontend.llm_ops_generator import (
        DeepSeekOpsGenerator,
        LLMOpsGeneratorInference,
        LLMOpsGeneratorTraining,
    )

    name = experiment["name"]
    model = experiment["model"]
    chip_version = experiment["chip_version"]
    workload = experiment["workload"]
    global_batch_size = experiment["global_batch_size"]
    parallelism_config = experiment["parallelism_config"]

    # Load and merge configs
    base_config = _load_configs(configs_path, model, chip_version)
    base_config.update(parallelism_config)

    pp = parallelism_config["pipeline_parallelism_degree"]
    dp_dcn = parallelism_config.get("data_parallel_degree_dcn", 1)
    pp_dcn = parallelism_config.get("pipeline_parallel_degree_dcn", 1)

    # Compute microbatch sizes
    microbatch_size_ici = ceil(global_batch_size / dp_dcn / pp / pp_dcn)
    microbatch_size_dcn = ceil(global_batch_size / pp_dcn)
    base_config["global_batch_size"] = global_batch_size
    base_config["microbatch_size_ici"] = microbatch_size_ici
    base_config["microbatch_size_dcn"] = microbatch_size_dcn

    # Build output file path
    exp_output_dir = os.path.join(output_dir, name)
    os.makedirs(exp_output_dir, exist_ok=True)
    output_file = os.path.join(exp_output_dir, f"{workload}-v{chip_version}.csv")
    base_config["output_file_path"] = output_file

    # Map parallelism to ICI axes
    axes_mappings = run_sim_lib.map_parallelism_to_ici_axes(
        model, chip_version, parallelism_config
    )
    if len(axes_mappings) == 4:
        dp_axes, tp_axes, pp_axes, ep_axes = axes_mappings
        base_config["num_expert_parallel_axes"] = ep_axes
    else:
        dp_axes, tp_axes, pp_axes = axes_mappings
    base_config["num_data_parallel_axes"] = dp_axes
    base_config["num_tensor_parallel_axes"] = tp_axes
    base_config["num_pipeline_parallel_axes"] = pp_axes

    # Create and run the appropriate ops generator
    model_lower = model.lower()
    if "llama" in model_lower:
        if workload == "training":
            base_config["use_flash_attention"] = False
            ops_generator = LLMOpsGeneratorTraining(base_config)
        else:
            ops_generator = LLMOpsGeneratorInference(base_config)
    elif "deepseek" in model_lower:
        ops_generator = DeepSeekOpsGenerator(base_config)
    elif "dlrm" in model_lower:
        ops_generator = DLRMOpsGenerator(base_config)
    elif "dit" in model_lower:
        ops_generator = DiTOpsGenerator(base_config)
    else:
        raise ValueError(f"Unsupported model: {model}")

    ops_generator.generate(dump_to_file=True, analyze_energy=False)

    # Dump stats (replicates run_sim.py dump_stats logic)
    config_dict = ops_generator.config.model_dump(mode="json")
    output_files = _dump_stats(
        model, chip_version, config_dict, ops_generator, workload
    )

    return output_files


def _dump_stats(
    model: str,
    v: str,
    base_config: dict[str, Any],
    ops_gen: Any,
    workload: str,
) -> list[str]:
    """Dump statistics and return list of output file paths produced."""
    model_lower = model.lower()
    if "llama" in model_lower or "deepseek" in model_lower:
        return _dump_stats_llm(v, base_config, ops_gen, workload)
    elif "dlrm" in model_lower:
        return _dump_stats_dlrm(v, base_config, ops_gen, workload)
    elif "dit" in model_lower:
        return _dump_stats_stable_diffusion(v, base_config, ops_gen, workload)
    else:
        raise ValueError(f"Unsupported model for stats: {model}")


def _dump_stats_llm(
    v: str,
    base_config: dict[str, Any],
    ops_gen: Any,
    workload: str,
) -> list[str]:
    """Dump LLM stats. Returns list of output file paths."""
    import neusim.npusim.frontend.run_sim_lib as run_sim_lib

    output_file: str = base_config["output_file_path"]
    global_batch_size: int = base_config["global_batch_size"]
    microbatch_size_ici: int = base_config["microbatch_size_ici"]
    pp_dcn: int = base_config["pipeline_parallel_degree_dcn"]
    dp_dcn: int = base_config["data_parallel_degree_dcn"]
    pp: int = base_config["pipeline_parallelism_degree"]

    num_pods = dp_dcn * pp_dcn
    batch_size_per_pod = ceil(global_batch_size / num_pods)
    base_config["num_pods"] = num_pods
    base_config["batch_size_per_pod"] = batch_size_per_pod

    total_pp = pp * pp_dcn
    layers_per_pp_stage = ceil(base_config["num_layers"] / total_pp)
    base_config["layers_per_pp_stage"] = layers_per_pp_stage

    output_files = [output_file]

    if workload == "inference":
        prefill_csv = output_file.replace(".csv", "_prefill.csv")
        prefill_json = output_file.replace(".csv", "_prefill.json")
        decode_csv = output_file.replace(".csv", "_decode.csv")
        decode_json = output_file.replace(".csv", "_decode.json")

        prefill_stats = run_sim_lib.get_statistics_from_trace_file(prefill_csv)
        decode_stats = run_sim_lib.get_statistics_from_trace_file(decode_csv)

        input_seqlen = base_config["input_seqlen"]
        output_seqlen = base_config["output_seqlen"]

        # Prefill throughput and TTFT
        prefill_pp_stage_time_ns = ceil(prefill_stats["total_execution_time_chip_ns"])
        prefill_stats["throughput_tokens_per_sec"] = (
            microbatch_size_ici * input_seqlen * 1e9 / prefill_pp_stage_time_ns
        )
        prefill_pod_latency_ns = (
            prefill_stats["total_execution_time_non_pp_ns"]
            + prefill_stats["total_pp_ici_time_ns"]
        ) * pp
        prefill_tot_latency_ns = (
            prefill_pod_latency_ns + prefill_stats["total_pp_dcn_time_ns"]
        ) * pp_dcn
        prefill_stats["TTFT_sec"] = prefill_tot_latency_ns / 1e9

        # Decode throughput and TPOT
        decode_pod_latency_ns = (
            (
                decode_stats["total_execution_time_non_pp_ns"]
                + decode_stats["total_pp_ici_time_ns"]
            )
            * pp
            / output_seqlen
        )
        decode_tot_latency_ns = (
            decode_pod_latency_ns + decode_stats["total_pp_dcn_time_ns"] / output_seqlen
        ) * pp_dcn
        decode_pp_stage_time_ns = ceil(
            decode_stats["total_execution_time_chip_ns"] / output_seqlen
        )
        decode_stats["TPOT_ms_request"] = decode_tot_latency_ns / 1e6
        decode_stats["throughput_tokens_per_sec"] = (
            microbatch_size_ici * 1e9 / decode_pp_stage_time_ns
        )
        decode_stats["throughput_tokens_per_sec_request"] = (
            1e3 / decode_stats["TPOT_ms_request"]
        )

        prefill_stats["mem_footprint_GB"] = (
            ops_gen.compute_memory_footprint_bytes("prefill") / 1024**3
        )
        prefill_stats["out_of_memory"] = (
            base_config["hbm_size_GB"] < prefill_stats["mem_footprint_GB"]
        )
        decode_stats["mem_footprint_GB"] = (
            ops_gen.compute_memory_footprint_bytes("decode") / 1024**3
        )
        decode_stats["out_of_memory"] = (
            base_config["hbm_size_GB"] < decode_stats["mem_footprint_GB"]
        )

        prefill_stats["sim_config"] = base_config
        decode_stats["sim_config"] = base_config

        with open(prefill_json, "w") as f:
            json.dump(prefill_stats, f, indent=4)
        with open(decode_json, "w") as f:
            json.dump(decode_stats, f, indent=4)

        output_files.extend([prefill_csv, prefill_json, decode_csv, decode_json])

    elif workload == "training":
        stats = run_sim_lib.get_statistics_from_trace_file(output_file)

        pp_time_multiplier = pp
        stats["total_execution_time_pod_ns"] = (
            stats["total_execution_time_chip_ns"] * pp_time_multiplier
        )
        stats["compute_only_time_pod_ns"] = (
            stats["compute_only_time_chip_ns"] * pp_time_multiplier
        )
        stats["memory_only_time_pod_ns"] = (
            stats["memory_only_time_chip_ns"] * pp_time_multiplier
        )
        stats["ici_bound_time_pod_ns"] = (
            stats["ici_bound_time_chip_ns"] * pp_time_multiplier
        )

        total_pp_dcn_time = stats["total_pp_dcn_time_ns"]
        total_pod_time = stats["total_execution_time_pod_ns"]
        if total_pp_dcn_time > total_pod_time:
            stats["bounded_by_pp_dcn"] = True
            stats["total_execution_time_ns"] = total_pp_dcn_time
        else:
            stats["bounded_by_pp_dcn"] = False
            stats["total_execution_time_ns"] = total_pod_time

        pp_dcn_time_multiplier = pp_dcn
        stats["total_execution_time_ns"] *= pp_dcn_time_multiplier

        stats["mem_footprint_GB"] = ops_gen.compute_memory_footprint_bytes() / 1024**3
        stats["out_of_memory"] = (
            base_config["num_chips"] * base_config["hbm_size_GB"]
            < stats["mem_footprint_GB"]
        )

        stats["sim_config"] = base_config
        stats_json = output_file.replace(".csv", ".json")
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=4)

        output_files.append(stats_json)
    else:
        raise ValueError(f"Invalid workload: {workload}")

    return output_files


def _dump_stats_dlrm(
    v: str,
    base_config: dict[str, Any],
    ops_gen: Any,
    workload: str,
) -> list[str]:
    """Dump DLRM stats. Returns list of output file paths."""
    import neusim.npusim.frontend.run_sim_lib as run_sim_lib

    output_file: str = base_config["output_file_path"]
    global_batch_size: int = base_config["global_batch_size"]
    pp_dcn: int = base_config["pipeline_parallel_degree_dcn"]
    dp_dcn: int = base_config["data_parallel_degree_dcn"]
    num_chips: int = base_config["num_chips"]

    num_pods = dp_dcn * pp_dcn
    batch_size_per_pod = ceil(global_batch_size / num_pods)
    base_config["num_pods"] = num_pods
    base_config["batch_size_per_pod"] = batch_size_per_pod

    output_files = [output_file]

    if workload == "inference":
        # Per-chip stats
        for chip_id in range(num_chips):
            chip_csv = output_file.replace(".csv", f"_chip{chip_id}.csv")
            stats = run_sim_lib.get_statistics_from_trace_file(chip_csv)
            stats["throughput_requests_per_sec"] = (
                global_batch_size / stats["total_execution_time_chip_ns"] * 1e9
            )
            stats["latency_ns"] = stats["total_execution_time_chip_ns"]
            stats["sim_config"] = base_config
            chip_json = chip_csv.replace(".csv", ".json")
            with open(chip_json, "w") as f:
                json.dump(stats, f, indent=4)
            output_files.extend([chip_csv, chip_json])

        # Overall stats (straggler chip)
        stats = run_sim_lib.get_statistics_from_trace_file(output_file)
        stats["throughput_requests_per_sec"] = (
            global_batch_size / stats["total_execution_time_chip_ns"] * 1e9
        )
        stats["latency_ns"] = stats["total_execution_time_chip_ns"]
        stats["mem_footprint_GB"] = ops_gen.compute_memory_footprint_bytes() / 1024**3
        stats["out_of_memory"] = (
            base_config["num_chips"] * base_config["hbm_size_GB"]
            < stats["mem_footprint_GB"]
        )
        stats["sim_config"] = base_config
        stats_json = output_file.replace(".csv", ".json")
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=4)
        output_files.append(stats_json)
    else:
        raise ValueError(f"DLRM only supports inference workload, got: {workload}")

    return output_files


def _dump_stats_stable_diffusion(
    v: str,
    base_config: dict[str, Any],
    ops_gen: Any,
    workload: str,
) -> list[str]:
    """Dump DiT/stable diffusion stats. Returns list of output file paths."""
    import neusim.npusim.frontend.run_sim_lib as run_sim_lib

    output_file: str = base_config["output_file_path"]
    global_batch_size: int = base_config["global_batch_size"]
    num_diffusion_steps: int = base_config["num_diffusion_steps"]

    output_files = [output_file]

    if workload == "inference":
        stats = run_sim_lib.get_statistics_from_trace_file(output_file)
        if base_config["model_type"] == "dit":
            total_num_steps = 1
        else:
            raise ValueError(f"Unsupported model type: {base_config['model_type']}")

        stats["throughput_requests_per_sec"] = global_batch_size / (
            stats["total_execution_time_chip_ns"] / 1e9 * total_num_steps
        )
        stats["throughput_step_per_sec_per_request"] = num_diffusion_steps / (
            stats["total_execution_time_chip_ns"] / 1e9
        )
        stats["latency_sec"] = (
            stats["total_execution_time_chip_ns"] / 1e9 * total_num_steps
        )
        stats["latency_step_sec"] = (
            stats["total_execution_time_chip_ns"]
            / num_diffusion_steps
            / 1e9
            / total_num_steps
        )
        stats["mem_footprint_GB"] = ops_gen.compute_memory_footprint_bytes() / 1024**3
        stats["out_of_memory"] = (
            base_config["num_chips"] * base_config["hbm_size_GB"]
            < stats["mem_footprint_GB"]
        )
        stats["sim_config"] = base_config
        stats_json = output_file.replace(".csv", ".json")
        with open(stats_json, "w") as f:
            json.dump(stats, f, indent=4)
        output_files.append(stats_json)
    else:
        raise ValueError(f"DiT only supports inference workload, got: {workload}")

    return output_files


def run_experiments(
    experiments: list[dict[str, Any]],
    configs_path: str,
    output_dir: str,
) -> dict[str, list[str]]:
    """Run all experiments and return {experiment_name: [output_file_paths]}.

    This is the main entry point for both in-process and subprocess invocation.
    """
    results: dict[str, list[str]] = {}
    for experiment in experiments:
        name = experiment["name"]
        output_files = _run_single_experiment(experiment, configs_path, output_dir)
        results[name] = output_files
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuSim regression runner")
    parser.add_argument(
        "--config",
        required=True,
        help="JSON string with experiments config",
    )
    args = parser.parse_args()

    config = json.loads(args.config)
    experiments = config["experiments"]
    configs_path = config["configs_path"]
    output_dir = config["output_dir"]

    results = run_experiments(experiments, configs_path, output_dir)

    # Write manifest
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({"status": "ok", "manifest": manifest_path}))


if __name__ == "__main__":
    main()
