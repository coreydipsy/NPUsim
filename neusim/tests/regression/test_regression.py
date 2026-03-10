"""Regression test comparing simulation outputs against a baseline commit.

This test always passes — it generates a diff report for human review rather
than failing on output differences (which may be intentional, e.g., bug fixes).

Usage:
    # Run the regression test
    pytest neusim/npusim/frontend/tests/test_regression.py --runslow -v

    # Compare against a specific commit
    NEUSIM_REGRESSION_BASELINE=v1.0.0 pytest ... --runslow -v

    # Save report to file
    NEUSIM_REGRESSION_REPORT_PATH=report.txt pytest ... --runslow -v
"""

import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # TP + PP (ICI)
    {
        "name": "llm_inference_llama3_8b_tp2_pp2_v5p",
        "model": "llama3-8b",
        "chip_version": "5p",
        "workload": "inference",
        "global_batch_size": 1,
        "parallelism_config": {
            "num_chips": 4,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 2,
            "pipeline_parallelism_degree": 2,
            "data_parallel_degree_dcn": 1,
            "pipeline_parallel_degree_dcn": 1,
        },
    },
    # DP + TP (ICI)
    {
        "name": "llm_training_llama3_8b_dp2_tp2_v5p",
        "model": "llama3-8b",
        "chip_version": "5p",
        "workload": "training",
        "global_batch_size": 64,
        "parallelism_config": {
            "num_chips": 4,
            "data_parallelism_degree": 2,
            "tensor_parallelism_degree": 2,
            "pipeline_parallelism_degree": 1,
            "data_parallel_degree_dcn": 1,
            "pipeline_parallel_degree_dcn": 1,
        },
    },
    # DP_DCN + PP_DCN (DCN) — tp_dcn is not supported by the simulator
    {
        "name": "llm_inference_llama3_8b_tp2_dpdcn2_ppdcn2_v5p",
        "model": "llama3-8b",
        "chip_version": "5p",
        "workload": "inference",
        "global_batch_size": 4,
        "parallelism_config": {
            "num_chips": 2,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 2,
            "pipeline_parallelism_degree": 1,
            "data_parallel_degree_dcn": 2,
            "pipeline_parallel_degree_dcn": 2,
        },
    },
    # DP (ICI) for DLRM multi-chip
    {
        "name": "dlrm_inference_dlrm_s_dp2_v5p",
        "model": "dlrm-s",
        "chip_version": "5p",
        "workload": "inference",
        "global_batch_size": 4,
        "parallelism_config": {
            "num_chips": 2,
            "data_parallelism_degree": 2,
            "tensor_parallelism_degree": 1,
            "pipeline_parallelism_degree": 1,
            "data_parallel_degree_dcn": 1,
            "pipeline_parallel_degree_dcn": 1,
        },
    },
    # DP (ICI) for DiT
    {
        "name": "dit_inference_dit_xl_dp2_v5p",
        "model": "dit-xl",
        "chip_version": "5p",
        "workload": "inference",
        "global_batch_size": 2,
        "parallelism_config": {
            "num_chips": 2,
            "data_parallelism_degree": 2,
            "tensor_parallelism_degree": 1,
            "pipeline_parallelism_degree": 1,
            "data_parallel_degree_dcn": 1,
            "pipeline_parallel_degree_dcn": 1,
        },
    },
    # TP + EP (ICI) for DeepSeek MoE
    {
        "name": "deepseek_inference_deepseekv2_236b_tp4_ep4_v5p",
        "model": "deepseekv2-236b",
        "chip_version": "5p",
        "workload": "inference",
        "global_batch_size": 1,
        "parallelism_config": {
            "num_chips": 4,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 4,
            "pipeline_parallelism_degree": 1,
            "expert_parallelism_degree": 4,
            "data_parallel_degree_dcn": 1,
            "pipeline_parallel_degree_dcn": 1,
            "expert_parallel_degree_dcn": 1,
        },
    },
    # EP_DCN (DCN) for DeepSeek MoE
    {
        "name": "deepseek_inference_deepseekv2_236b_tp2_ep2_epdcn2_v5p",
        "model": "deepseekv2-236b",
        "chip_version": "5p",
        "workload": "inference",
        "global_batch_size": 1,
        "parallelism_config": {
            "num_chips": 2,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 2,
            "pipeline_parallelism_degree": 1,
            "expert_parallelism_degree": 2,
            "data_parallel_degree_dcn": 1,
            "pipeline_parallel_degree_dcn": 1,
            "expert_parallel_degree_dcn": 2,
        },
    },
]

# Fields to skip when comparing JSON outputs (paths vary between runs)
SKIP_JSON_FIELDS = {"output_file_path"}


# ---------------------------------------------------------------------------
# Fixture: set up worktree + venv for baseline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def regression_env(tmp_path_factory: pytest.TempPathFactory):
    """Set up a git worktree + lightweight venv for the baseline commit."""
    tmp_path = tmp_path_factory.mktemp("regression")
    repo_root = Path(__file__).resolve().parents[3]  # NeuSim repo root
    configs_path_current = repo_root / "configs"

    # Determine baseline commit
    baseline = os.environ.get("NEUSIM_REGRESSION_BASELINE", "HEAD~1")

    # Check that the baseline commit exists
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--verify", baseline],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )
        baseline_sha = result.stdout.strip()
    except subprocess.CalledProcessError:
        pytest.skip(f"Baseline commit '{baseline}' not found (no previous commit?)")

    worktree_path = tmp_path / "worktree"
    venv_path = tmp_path / "venv"
    baseline_output = tmp_path / "baseline_output"
    current_output = tmp_path / "current_output"
    baseline_output.mkdir()
    current_output.mkdir()

    try:
        # Create git worktree at baseline commit
        subprocess.run(
            ["git", "worktree", "add", str(worktree_path), baseline_sha],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=True,
        )

        # Create lightweight venv with system site-packages
        subprocess.run(
            [
                sys.executable,
                "-m",
                "venv",
                str(venv_path),
                "--system-site-packages",
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        venv_python = venv_path / "bin" / "python"

        # Install baseline code into venv (no-deps to avoid touching conda)
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--no-deps",
                "-e",
                str(worktree_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        # Copy regression_runner.py into the worktree test directory
        runner_src = Path(__file__).parent / "regression_runner.py"
        runner_dst_dir = worktree_path / "neusim" / "tests" / "regression"
        runner_dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(runner_src, runner_dst_dir / "regression_runner.py")

        configs_path_baseline = worktree_path / "configs"

        yield {
            "venv_python": str(venv_python),
            "worktree_path": str(worktree_path),
            "configs_path_baseline": str(configs_path_baseline),
            "configs_path_current": str(configs_path_current),
            "baseline_output": str(baseline_output),
            "current_output": str(current_output),
            "baseline_sha": baseline_sha,
            "repo_root": str(repo_root),
        }
    finally:
        # Cleanup worktree
        try:
            subprocess.run(
                ["git", "worktree", "remove", "--force", str(worktree_path)],
                cwd=repo_root,
                capture_output=True,
                text=True,
            )
        except Exception:
            pass
        # Cleanup venv and temp dirs
        if venv_path.exists():
            shutil.rmtree(venv_path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Comparison & reporting utilities
# ---------------------------------------------------------------------------


def _try_numeric(value: str) -> float | str:
    """Try to parse a string as a number."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return value


def _compare_json_recursive(
    baseline: Any,
    current: Any,
    path: str = "",
    diffs: list[dict[str, Any]] | None = None,
    total_fields: list[int] | None = None,
) -> tuple[list[dict[str, Any]], int]:
    """Recursively compare two JSON structures, collecting diffs."""
    if diffs is None:
        diffs = []
    if total_fields is None:
        total_fields = [0]

    if isinstance(baseline, dict) and isinstance(current, dict):
        all_keys = set(baseline.keys()) | set(current.keys())
        for key in sorted(all_keys):
            field_path = f"{path}.{key}" if path else key
            # Skip path fields
            if key in SKIP_JSON_FIELDS:
                continue
            if key not in baseline:
                diffs.append(
                    {
                        "field": field_path,
                        "type": "added",
                        "current": current[key],
                    }
                )
                total_fields[0] += 1
            elif key not in current:
                diffs.append(
                    {
                        "field": field_path,
                        "type": "removed",
                        "baseline": baseline[key],
                    }
                )
                total_fields[0] += 1
            else:
                _compare_json_recursive(
                    baseline[key], current[key], field_path, diffs, total_fields
                )
    elif isinstance(baseline, list) and isinstance(current, list):
        max_len = max(len(baseline), len(current))
        for i in range(max_len):
            field_path = f"{path}[{i}]"
            if i >= len(baseline):
                diffs.append(
                    {
                        "field": field_path,
                        "type": "added",
                        "current": current[i],
                    }
                )
                total_fields[0] += 1
            elif i >= len(current):
                diffs.append(
                    {
                        "field": field_path,
                        "type": "removed",
                        "baseline": baseline[i],
                    }
                )
                total_fields[0] += 1
            else:
                _compare_json_recursive(
                    baseline[i], current[i], field_path, diffs, total_fields
                )
    else:
        total_fields[0] += 1
        if baseline != current:
            diff_entry: dict[str, Any] = {
                "field": path,
                "type": "changed",
                "baseline": baseline,
                "current": current,
            }
            # Compute deltas for numeric values
            if isinstance(baseline, int | float) and isinstance(current, int | float):
                abs_delta = current - baseline
                rel_delta = (
                    (abs_delta / baseline * 100) if baseline != 0 else float("inf")
                )
                diff_entry["abs_delta"] = abs_delta
                diff_entry["rel_delta_pct"] = rel_delta
            diffs.append(diff_entry)

    return diffs, total_fields[0]


def _compare_csv_files(
    baseline_path: str, current_path: str
) -> tuple[list[dict[str, Any]], int]:
    """Compare two CSV files row by row."""
    diffs: list[dict[str, Any]] = []
    total_fields = 0

    with open(baseline_path, newline="") as f:
        baseline_rows = list(csv.DictReader(f))
    with open(current_path, newline="") as f:
        current_rows = list(csv.DictReader(f))

    if len(baseline_rows) != len(current_rows):
        diffs.append(
            {
                "field": "row_count",
                "type": "changed",
                "baseline": len(baseline_rows),
                "current": len(current_rows),
            }
        )

    # Compare headers
    baseline_headers = list(baseline_rows[0].keys()) if baseline_rows else []
    current_headers = list(current_rows[0].keys()) if current_rows else []
    if baseline_headers != current_headers:
        added = set(current_headers) - set(baseline_headers)
        removed = set(baseline_headers) - set(current_headers)
        if added:
            diffs.append(
                {"field": "headers_added", "type": "added", "current": sorted(added)}
            )
        if removed:
            diffs.append(
                {
                    "field": "headers_removed",
                    "type": "removed",
                    "baseline": sorted(removed),
                }
            )

    # Compare cell values
    common_headers = [h for h in baseline_headers if h in current_headers]
    min_rows = min(len(baseline_rows), len(current_rows))
    for row_idx in range(min_rows):
        for header in common_headers:
            total_fields += 1
            b_val = baseline_rows[row_idx].get(header, "")
            c_val = current_rows[row_idx].get(header, "")
            if b_val != c_val:
                diff_entry: dict[str, Any] = {
                    "field": f"row[{row_idx}].{header}",
                    "type": "changed",
                    "baseline": b_val,
                    "current": c_val,
                }
                b_num = _try_numeric(b_val)
                c_num = _try_numeric(c_val)
                if isinstance(b_num, float) and isinstance(c_num, float):
                    abs_delta = c_num - b_num
                    rel_delta = (
                        (abs_delta / b_num * 100) if b_num != 0 else float("inf")
                    )
                    diff_entry["abs_delta"] = abs_delta
                    diff_entry["rel_delta_pct"] = rel_delta
                diffs.append(diff_entry)

    return diffs, total_fields


def _format_report(
    all_results: dict[str, dict[str, tuple[list[dict[str, Any]], int]]],
    baseline_sha: str,
) -> str:
    """Format the comparison results into a human-readable report."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("NeuSim Regression Report")
    lines.append(f"Baseline commit: {baseline_sha}")
    lines.append("=" * 72)

    any_diff = False
    for exp_name, file_results in sorted(all_results.items()):
        lines.append("")
        lines.append(f"--- Experiment: {exp_name} ---")

        exp_has_diff = False
        for filename, (diffs, total_fields) in sorted(file_results.items()):
            if diffs:
                exp_has_diff = True
                any_diff = True
                lines.append(
                    f"  {filename}: {len(diffs)} field(s) changed out of {total_fields}"
                )
                for d in diffs:
                    field = d["field"]
                    dtype = d["type"]
                    if dtype == "changed":
                        base_val = d["baseline"]
                        curr_val = d["current"]
                        delta_str = ""
                        if "abs_delta" in d:
                            abs_d = d["abs_delta"]
                            rel_d = d["rel_delta_pct"]
                            if rel_d == float("inf"):
                                delta_str = f" (delta: {abs_d}, rel: inf%)"
                            else:
                                delta_str = f" (delta: {abs_d:.6g}, rel: {rel_d:+.4f}%)"
                        lines.append(
                            f"    {field}: {base_val} -> {curr_val}{delta_str}"
                        )
                    elif dtype == "added":
                        lines.append(f"    {field}: [NEW] {d['current']}")
                    elif dtype == "removed":
                        lines.append(f"    {field}: [REMOVED] {d['baseline']}")
            else:
                lines.append(f"  {filename}: all {total_fields} fields match")

        if not exp_has_diff:
            lines.append("  All outputs match baseline.")

    lines.append("")
    if not any_diff:
        lines.append("RESULT: All outputs match baseline across all experiments.")
    else:
        lines.append(
            "RESULT: Differences detected. Review above to verify correctness."
        )
    lines.append("=" * 72)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The regression test
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.regression
def test_regression_vs_previous_commit(regression_env: dict[str, str]) -> None:
    """Compare simulation outputs between current code and baseline commit.

    This test always passes. It generates a diff report for human review.
    """
    venv_python = regression_env["venv_python"]
    configs_baseline = regression_env["configs_path_baseline"]
    configs_current = regression_env["configs_path_current"]
    baseline_output = regression_env["baseline_output"]
    current_output = regression_env["current_output"]
    baseline_sha = regression_env["baseline_sha"]
    worktree_path = regression_env["worktree_path"]

    runner_path = os.path.join(
        worktree_path,
        "neusim",
        "tests",
        "regression",
        "regression_runner.py",
    )

    # --- Run baseline simulations via subprocess ---
    baseline_config = json.dumps(
        {
            "experiments": EXPERIMENTS,
            "configs_path": configs_baseline,
            "output_dir": baseline_output,
        }
    )

    baseline_result = subprocess.run(
        [venv_python, runner_path, "--config", baseline_config],
        capture_output=True,
        text=True,
        timeout=600,
    )
    if baseline_result.returncode != 0:
        print(
            f"WARNING: Baseline simulation failed (exit code {baseline_result.returncode}).\n"
            f"stderr:\n{baseline_result.stderr}\n"
            f"stdout:\n{baseline_result.stdout}",
            file=sys.stderr,
        )
        pytest.skip(f"Baseline simulation failed: {baseline_result.stderr[:500]}")

    # --- Run current simulations via direct import ---
    from neusim.tests.regression.regression_runner import run_experiments

    current_results = run_experiments(EXPERIMENTS, configs_current, current_output)

    # --- Load baseline manifest ---
    baseline_manifest_path = os.path.join(baseline_output, "manifest.json")
    with open(baseline_manifest_path) as f:
        baseline_results = json.load(f)

    # --- Compare outputs ---
    all_comparisons: dict[str, dict[str, tuple[list[dict[str, Any]], int]]] = {}

    for experiment in EXPERIMENTS:
        exp_name = experiment["name"]
        exp_comparisons: dict[str, tuple[list[dict[str, Any]], int]] = {}

        baseline_files = baseline_results.get(exp_name, [])
        current_files = current_results.get(exp_name, [])

        # Build maps from filename -> full path
        baseline_by_name = {os.path.basename(f): f for f in baseline_files}
        current_by_name = {os.path.basename(f): f for f in current_files}

        all_filenames = sorted(
            set(baseline_by_name.keys()) | set(current_by_name.keys())
        )

        for filename in all_filenames:
            b_path = baseline_by_name.get(filename)
            c_path = current_by_name.get(filename)

            if b_path is None or c_path is None:
                # File only exists on one side
                if b_path is None:
                    exp_comparisons[filename] = (
                        [{"field": filename, "type": "added", "current": c_path}],
                        1,
                    )
                else:
                    exp_comparisons[filename] = (
                        [{"field": filename, "type": "removed", "baseline": b_path}],
                        1,
                    )
                continue

            if not os.path.exists(b_path) or not os.path.exists(c_path):
                continue

            if filename.endswith(".json"):
                with open(b_path) as f:
                    b_data = json.load(f)
                with open(c_path) as f:
                    c_data = json.load(f)
                diffs, total = _compare_json_recursive(b_data, c_data)
                exp_comparisons[filename] = (diffs, total)
            elif filename.endswith(".csv"):
                diffs, total = _compare_csv_files(b_path, c_path)
                exp_comparisons[filename] = (diffs, total)

        all_comparisons[exp_name] = exp_comparisons

    # --- Generate report ---
    report = _format_report(all_comparisons, baseline_sha)
    print("\n" + report)

    # Save report to file (default: results/regression/regression_report.txt)
    repo_root = Path(__file__).resolve().parents[3]
    default_report_dir = repo_root / "results" / "regression"
    default_report_path = str(default_report_dir / "regression_report.txt")
    report_path = os.environ.get("NEUSIM_REGRESSION_REPORT_PATH", default_report_path)
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report)
    print(f"Report saved to: {report_path}")

    # Test always passes — differences are for human review
