#!/usr/bin/env python3
"""Section 3.2: Parallelism Strategies.

Config: 3 models × 3 batch sizes × 5 (TP,PP) configs = 45 runs.
All on TPUv5p, 16 chips, seqlen=2048/128.

Produces:
  - parallelism_sweep.csv (all 45 rows)
"""
import json, csv, sys
from pathlib import Path
from math import ceil

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(ROOT))

from neusim.npusim.frontend.llm_ops_generator import (
    LLMOpsGeneratorInference, DeepSeekOpsGenerator, GptOssOpsGenerator,
)
from neusim.npusim.frontend.run_sim_lib import map_parallelism_to_ici_axes

OUTDIR = Path(__file__).parent
MODELS = [
    ('gpt-oss-120b',    GptOssOpsGenerator,       True),
    ('deepseekv2-236b', DeepSeekOpsGenerator,     True),
    ('llama3_1-405b',   LLMOpsGeneratorInference, False),
]
BATCH_SIZES = [1, 4, 16, 64]
TP_PP = [(1, 16), (2, 8), (4, 4), (8, 2), (16, 1)]
INPUT_SEQLEN = 2048
OUTPUT_SEQLEN = 128

def load_base(model_name):
    with open(ROOT / f'configs/models/{model_name}.json') as f: m = json.load(f)
    with open(ROOT / 'configs/chips/tpuv5p.json') as f: c = json.load(f)
    with open(ROOT / 'configs/systems/system_config.json') as f: s = json.load(f)
    return {**s, **c, **m}

all_rows = []
total = len(MODELS) * len(BATCH_SIZES) * len(TP_PP)
done = 0

for model_name, gen_cls, is_moe in MODELS:
    base = load_base(model_name)
    print(f'\n=== {model_name} ===')

    for bs in BATCH_SIZES:
        for tp, pp in TP_PP:
            done += 1
            cfg = dict(base)
            cfg['num_chips'] = 16
            cfg['tensor_parallelism_degree'] = tp
            cfg['pipeline_parallelism_degree'] = pp
            cfg['data_parallelism_degree'] = 1
            if is_moe:
                cfg['expert_parallelism_degree'] = 1
            # Use the simulator's own heuristic for ICI axis assignment
            par_cfg = {'data_parallelism_degree': 1,
                       'tensor_parallelism_degree': tp,
                       'pipeline_parallelism_degree': pp,
                       'expert_parallelism_degree': 1 if is_moe else None}
            axes = map_parallelism_to_ici_axes(model_name, '5p', par_cfg)
            if is_moe:
                dp_axes, tp_axes, pp_axes, ep_axes = axes
                cfg['num_expert_parallel_axes'] = ep_axes
            else:
                dp_axes, tp_axes, pp_axes = axes
            cfg['num_data_parallel_axes'] = dp_axes
            cfg['num_tensor_parallel_axes'] = tp_axes
            cfg['num_pipeline_parallel_axes'] = pp_axes
            cfg['global_batch_size'] = bs
            cfg['microbatch_size_ici'] = bs  # fixed microbatch = full batch for fair comparison across PP
            cfg['microbatch_size_dcn'] = bs
            cfg['input_seqlen'] = INPUT_SEQLEN
            cfg['output_seqlen'] = OUTPUT_SEQLEN
            cfg['output_file_path'] = str(OUTDIR / f'{model_name}_bs{bs}_tp{tp}_pp{pp}.csv')

            gen = gen_cls(cfg)
            _, pf_ops, dc_ops = gen.generate(
                dump_to_file=True, separate_prefill_decode=True, analyze_energy=False)

            pf_time = sum(o.stats.execution_time_ns * o.stats.count for o in pf_ops)
            dc_time = sum(o.stats.execution_time_ns * o.stats.count for o in dc_ops)
            pf_ici = sum(o.stats.ici_traffic_outbound_bytes * o.stats.count for o in pf_ops)
            dc_ici = sum(o.stats.ici_traffic_outbound_bytes * o.stats.count for o in dc_ops)
            mb = ceil(bs / pp)

            # End-to-end latency
            ttft_ms = pf_time / 1e6 * pp
            tpot_ms = (dc_time / 1e6 / OUTPUT_SEQLEN) * pp
            # Steady-state throughput (NeuSim default)
            ss_pf_tps = (mb * INPUT_SEQLEN) / (pf_time / 1e9) if pf_time > 0 else 0
            ss_dc_tps = mb / ((dc_time / 1e9) / OUTPUT_SEQLEN) if dc_time > 0 else 0
            # Single-request throughput
            sr_pf_tps = (bs * INPUT_SEQLEN) / (ttft_ms / 1e3) if ttft_ms > 0 else 0
            sr_dc_tps = bs / (tpot_ms / 1e3) if tpot_ms > 0 else 0

            row = {
                'model': model_name, 'batch_size': bs, 'microbatch': mb,
                'tp': tp, 'pp': pp,
                'pf_stage_ms': round(pf_time / 1e6, 3),
                'dc_stage_ms': round(dc_time / 1e6, 3),
                'ttft_ms': round(ttft_ms, 2),
                'tpot_ms': round(tpot_ms, 3),
                'pf_ici_bytes': pf_ici, 'dc_ici_bytes': dc_ici,
                'ss_pf_tps': round(ss_pf_tps, 1),
                'ss_dc_tps': round(ss_dc_tps, 1),
                'sr_pf_tps': round(sr_pf_tps, 1),
                'sr_dc_tps': round(sr_dc_tps, 1),
            }
            all_rows.append(row)
            print(f'  [{done:>2}/{total}] bs={bs:>2} TP={tp:>2} PP={pp:>2}: '
                  f'TTFT={ttft_ms:>9.1f}ms TPOT={tpot_ms:>8.2f}ms '
                  f'pf_ici={pf_ici/1024**2:>6.0f}MB dc_ici={dc_ici/1024**2:>6.0f}MB')

with open(OUTDIR / 'parallelism_sweep.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=all_rows[0].keys())
    w.writeheader()
    w.writerows(all_rows)
print(f'\nSaved {OUTDIR / "parallelism_sweep.csv"} ({len(all_rows)} rows)')
