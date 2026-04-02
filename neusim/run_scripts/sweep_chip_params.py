#!/usr/bin/env python3

"""Sweep individual NPU chip parameters for single-chip design space exploration.

This script loads the base TPUv5p + Llama-3-8B configuration, then sweeps one
chip parameter at a time across a set of values and batch sizes. For each
combination it runs the NeuSim simulation and collects end-to-end latency and
throughput metrics for both prefill and decode phases.

The baseline TPUv5p configuration values are:
    num_sa=8, sa_dim=128, num_vu=6, hbm_bw_GBps=2765, vmem_size_MB=128

Each sweep varies ONE parameter while keeping all others at baseline. This
lets you study the sensitivity of inference performance to each hardware
component in isolation.

Output:
    results/sweep/<param>_<value>/bs<batch_size>/
        inference-v5p.csv            -- combined operator trace
        inference-v5p_prefill.csv    -- prefill operator trace
        inference-v5p_decode.csv     -- decode operator trace
        prefill.json                 -- prefill end-to-end stats
        decode.json                  -- decode end-to-end stats
    results/sweep/sweep_summary.csv  -- one-row-per-run summary of all sweeps

Usage:
    # Run all default sweeps (5 params x 5 values x 5 batch sizes = 125 runs)
    python sweep_chip_params.py

    # Sweep only specific parameters
    python sweep_chip_params.py --params num_sa hbm_bw_GBps

    # Override batch sizes (comma-separated)
    python sweep_chip_params.py --batch_sizes 1,16,256

    # Change the model config (default: llama3-8b)
    python sweep_chip_params.py --model configs/models/llama3-8b.json

    # Change the chip config baseline (default: tpuv5p)
    python sweep_chip_params.py --chip configs/chips/tpuv5p.json

    # Change output directory
    python sweep_chip_params.py --output_dir results/my_sweep

    # Override sequence lengths
    python sweep_chip_params.py --batch_sizes 4 --input_seqlen 4096 --output_seqlen 128

    # Combine flags
    python sweep_chip_params.py --params num_sa sa_dim --batch_sizes 1,4 --output_dir results/quick
"""

import json
import argparse
import csv
from pathlib import Path

from neusim.npusim.frontend.llm_ops_generator import LLMOpsGeneratorInference
from neusim.npusim.frontend import run_sim_lib


def _get_ops_generator(config: dict):
    """Return the appropriate OpsGenerator based on the model name in config."""
    model_name = config.get("model_name", "").lower()
    if "gpt-oss" in model_name:
        from neusim.npusim.frontend.llm_ops_generator import GptOssOpsGenerator
        return GptOssOpsGenerator(config)
    elif "deepseek" in model_name:
        from neusim.npusim.frontend.llm_ops_generator import DeepSeekOpsGenerator
        return DeepSeekOpsGenerator(config)
    else:
        return LLMOpsGeneratorInference(config)


NEUSIM_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------
# Each entry maps a parameter name to:
#   (list_of_values_to_sweep, extra_overrides_or_None)
#
# extra_overrides is a dict mapping {target_param: source_param} for cases
# where changing one parameter requires a coupled change to another.
# For example, num_vu_ports must always equal num_vu.

DEFAULT_SWEEPS: dict[str, tuple[list, dict[str, str] | None]] = {
    "num_sa":       ([1, 2, 4, 8, 16],             None),
    "sa_dim":       ([32, 64, 128, 256, 512],       None),
    "num_vu":       ([1, 2, 4, 6, 12],              {"num_vu_ports": "num_vu"}),
    "hbm_bw_GBps":  ([500, 1000, 2000, 2765, 4000], None),
    "vmem_size_MB": ([16, 32, 64, 128, 256],        None),
}

DEFAULT_BATCH_SIZES = [1, 4, 16, 64, 256]


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_base_config(
    model_path: str | None = None,
    chip_path: str | None = None,
) -> dict:
    """Load and merge the base model, chip, and system configs.

    Args:
        model_path: Path to model JSON config. Defaults to llama3-8b.
        chip_path: Path to chip JSON config. Defaults to tpuv5p.
    """
    model_path = model_path or str(NEUSIM_ROOT / "configs/models/llama3-8b.json")
    chip_path = chip_path or str(NEUSIM_ROOT / "configs/chips/tpuv5p.json")

    with open(model_path) as f:
        model_cfg = json.load(f)
    with open(chip_path) as f:
        npu_cfg = json.load(f)
    with open(NEUSIM_ROOT / "configs/systems/system_config.json") as f:
        sys_cfg = json.load(f)
    return {**sys_cfg, **npu_cfg, **model_cfg}


# ---------------------------------------------------------------------------
# Single simulation run
# ---------------------------------------------------------------------------

def run_single(
    base_config: dict,
    param_name: str,
    param_value,
    batch_size: int,
    output_root: Path,
    extra_overrides: dict[str, str] | None = None,
    input_seqlen: int | None = None,
    output_seqlen: int | None = None,
) -> dict:
    """Run one simulation with the given parameter override and return metrics.

    Args:
        base_config: Merged base config dict.
        param_name: Name of the chip parameter being swept (e.g. "num_sa").
        param_value: Value to set for this parameter.
        batch_size: Global batch size for this run.
        output_root: Root directory for output files.
        extra_overrides: Optional coupled parameter overrides.
        input_seqlen: Override input sequence length. If None, uses model config default.
        output_seqlen: Override output sequence length. If None, uses model config default.

    Returns:
        Dict with columns: param, value, batch_size, TTFT_sec,
        prefill_throughput_tok_per_sec, TPOT_ms, decode_throughput_tok_per_sec,
        prefill_frac_compute, prefill_frac_memory, decode_frac_compute,
        decode_frac_memory.
    """
    config = dict(base_config)

    # Apply the swept parameter
    config[param_name] = param_value

    # Apply any coupled overrides (e.g., num_vu_ports = num_vu)
    if extra_overrides:
        for target_key, source_key in extra_overrides.items():
            config[target_key] = config[source_key]

    # Override sequence lengths if specified
    if input_seqlen is not None:
        config["input_seqlen"] = input_seqlen
    if output_seqlen is not None:
        config["output_seqlen"] = output_seqlen

    # Single-chip, no parallelism
    config["num_chips"] = 1
    config["global_batch_size"] = batch_size
    config["microbatch_size_ici"] = batch_size
    config["microbatch_size_dcn"] = batch_size
    config["data_parallelism_degree"] = 1
    config["tensor_parallelism_degree"] = 1
    config["pipeline_parallelism_degree"] = 1
    config["data_parallel_degree_dcn"] = 1
    config["pipeline_parallel_degree_dcn"] = 1

    outdir = output_root / f"{param_name}_{param_value}" / f"bs{batch_size}"
    outdir.mkdir(parents=True, exist_ok=True)
    config["output_file_path"] = str(outdir / "inference-v5p.csv")

    # ---- Run simulation ----
    ops_gen = _get_ops_generator(config)
    ops_gen.generate(dump_to_file=True, separate_prefill_decode=True)

    # ---- Collect stats ----
    prefill_stats = run_sim_lib.get_statistics_from_trace_file(
        str(outdir / "inference-v5p_prefill.csv")
    )
    decode_stats = run_sim_lib.get_statistics_from_trace_file(
        str(outdir / "inference-v5p_decode.csv")
    )

    input_seqlen = config["input_seqlen"]
    output_seqlen = config["output_seqlen"]

    # Time to First Token (prefill latency)
    ttft = prefill_stats["total_execution_time_chip_ns"] / 1e9

    # Prefill throughput (tokens/sec across all sequences in the batch)
    prefill_tput = batch_size * input_seqlen * 1e9 / prefill_stats["total_execution_time_chip_ns"]

    # Time Per Output Token per request (decode latency)
    tpot = decode_stats["total_execution_time_chip_ns"] / output_seqlen / 1e6

    # Decode throughput (tokens/sec across all sequences in the batch)
    decode_tput = batch_size * 1e9 / (decode_stats["total_execution_time_chip_ns"] / output_seqlen)

    # Bottleneck breakdown (fraction of total chip time)
    for stats in [prefill_stats, decode_stats]:
        total = stats["total_execution_time_chip_ns"]
        stats["frac_compute"] = stats.get("compute_only_time_chip_ns", 0) / total if total else 0
        stats["frac_memory"] = stats.get("memory_only_time_chip_ns", 0) / total if total else 0

    result = {
        "param": param_name,
        "value": param_value,
        "batch_size": batch_size,
        "TTFT_sec": ttft,
        "prefill_throughput_tok_per_sec": prefill_tput,
        "TPOT_ms": tpot,
        "decode_throughput_tok_per_sec": decode_tput,
        "prefill_frac_compute": prefill_stats["frac_compute"],
        "prefill_frac_memory": prefill_stats["frac_memory"],
        "decode_frac_compute": decode_stats["frac_compute"],
        "decode_frac_memory": decode_stats["frac_memory"],
    }

    # Save per-run JSON
    prefill_stats["TTFT_sec"] = ttft
    prefill_stats["throughput_tokens_per_sec"] = prefill_tput
    decode_stats["TPOT_ms_request"] = tpot
    decode_stats["throughput_tokens_per_sec"] = decode_tput
    with open(str(outdir / "prefill.json"), "w") as f:
        json.dump(prefill_stats, f, indent=4)
    with open(str(outdir / "decode.json"), "w") as f:
        json.dump(decode_stats, f, indent=4)

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Sweep NPU chip parameters and measure LLM inference performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python sweep_chip_params.py                              # all sweeps, default settings
  python sweep_chip_params.py --params num_sa hbm_bw_GBps  # only sweep two params
  python sweep_chip_params.py --batch_sizes 1,16            # fewer batch sizes for speed
  python sweep_chip_params.py --model configs/models/gpt-oss-20b.json  # different model
""",
    )
    parser.add_argument(
        "--params", nargs="*", default=None,
        help="Parameters to sweep (space-separated). "
             "Choices: num_sa, sa_dim, num_vu, hbm_bw_GBps, vmem_size_MB. "
             "Default: all.",
    )
    parser.add_argument(
        "--batch_sizes", type=str, default=None,
        help="Comma-separated list of batch sizes. Default: 1,4,16,64,256.",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model JSON config. Default: configs/models/llama3-8b.json.",
    )
    parser.add_argument(
        "--chip", type=str, default=None,
        help="Path to chip JSON config (baseline). Default: configs/chips/tpuv5p.json.",
    )
    parser.add_argument(
        "--input_seqlen", type=int, default=None,
        help="Override input sequence length. Default: use model config value.",
    )
    parser.add_argument(
        "--output_seqlen", type=int, default=None,
        help="Override output sequence length. Default: use model config value.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=str(NEUSIM_ROOT / "results" / "sweep"),
        help="Output directory for sweep results. Default: results/sweep/.",
    )
    args = parser.parse_args()

    params_to_sweep = args.params if args.params else list(DEFAULT_SWEEPS.keys())
    batch_sizes = (
        [int(x) for x in args.batch_sizes.split(",")]
        if args.batch_sizes
        else DEFAULT_BATCH_SIZES
    )
    output_root = Path(args.output_dir)
    input_seqlen = args.input_seqlen
    output_seqlen = args.output_seqlen

    base_config = load_base_config(model_path=args.model, chip_path=args.chip)

    all_results = []
    total = sum(len(DEFAULT_SWEEPS[p][0]) for p in params_to_sweep if p in DEFAULT_SWEEPS) * len(batch_sizes)
    done = 0

    for param_name in params_to_sweep:
        if param_name not in DEFAULT_SWEEPS:
            print(f"Warning: unknown parameter '{param_name}', skipping.")
            continue
        values, extra_overrides = DEFAULT_SWEEPS[param_name]
        print(f"\n{'='*60}")
        print(f"Sweeping {param_name}: {values}")
        print(f"Batch sizes: {batch_sizes}")
        print(f"{'='*60}")

        for val in values:
            for bs in batch_sizes:
                done += 1
                print(f"  [{done}/{total}] {param_name}={val}, bs={bs} ... ", end="", flush=True)
                result = run_single(base_config, param_name, val, bs, output_root, extra_overrides,
                                    input_seqlen=input_seqlen, output_seqlen=output_seqlen)
                all_results.append(result)
                print(f"TTFT={result['TTFT_sec']:.4f}s, TPOT={result['TPOT_ms']:.2f}ms")

    # Write summary CSV
    if all_results:
        summary_path = output_root / "sweep_summary.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nSummary written to {summary_path}")
        print(f"Total runs: {len(all_results)}")


if __name__ == "__main__":
    main()
