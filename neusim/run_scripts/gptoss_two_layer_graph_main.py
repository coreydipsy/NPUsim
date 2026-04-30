#!/usr/bin/env python3

"""Generate two-layer GPT-OSS computational graphs for prefill and decode.

The output diagrams cover the first two layers of GPT-OSS-20B:
  - Layer 0: sliding-window attention
  - Layer 1: full attention

Each diagram includes the operator sequence and tensor dimensions for inputs,
weights, and outputs. The script writes both Graphviz DOT and SVG files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "configs").is_dir() and (candidate / "neusim").is_dir():
            return candidate
    raise FileNotFoundError(f"Could not locate NeuSim repo root from {start}.")


REPO_ROOT = _find_repo_root(Path(__file__).resolve().parent)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from neusim.configs.models.LLMConfig import GptOssConfig
from neusim.npusim.frontend import llm_ops_lib as ops_lib


DEFAULT_MODEL = REPO_ROOT / "configs/models/gpt-oss-20b.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results/gptoss_two_layer_graph"
DEFAULT_INPUT_SEQLEN = 4096
DEFAULT_OUTPUT_SEQLEN = 128


def _fmt_shape(shape: tuple[int, ...] | list[int]) -> str:
    return "[" + ",".join(str(x) for x in shape) + "]"


def _escape(text: str) -> str:
    return text.replace('"', '\\"')


def _make_node(node_id: str, title: str, lines: list[str], fill: str) -> str:
    label = _escape(title + "\\n" + "\\n".join(lines))
    return (
        f'{node_id} [shape=box, style="rounded,filled", fillcolor="{fill}", '
        f'fontname="Helvetica", fontsize=10, label="{label}"];'
    )


def _make_tensor(node_id: str, title: str, shape: str) -> str:
    label = _escape(f"{title}\\n{shape}")
    return (
        f'{node_id} [shape=ellipse, style="filled", fillcolor="#f2f2f2", '
        f'fontname="Helvetica", fontsize=10, label="{label}"];'
    )


def _attention_prefill_ops(cfg: GptOssConfig, layer_type: str):
    if layer_type == "sliding":
        return ops_lib.create_sliding_window_attention(
            batch_size=cfg.global_batch_size,
            q_seqlen=cfg.input_seqlen,
            sliding_window_size=cfg.sliding_window_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            d_model=cfg.d_model,
            d_head=cfg.d_head,
            num_layers=1,
            config=cfg,
            is_decode=False,
            use_flash_attention=cfg.use_flash_attention,
        )
    return ops_lib.create_multi_head_attention(
        batch_size=cfg.global_batch_size,
        input_seqlen=cfg.input_seqlen,
        output_seqlen=cfg.output_seqlen,
        decode_width=cfg.decode_width,
        num_heads=cfg.num_heads,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        num_layers=1,
        config=cfg,
        is_decode=False,
        use_flash_attention=cfg.use_flash_attention,
        num_kv_heads=cfg.num_kv_heads,
    )


def _attention_decode_ops(cfg: GptOssConfig, layer_type: str):
    if layer_type == "sliding":
        return ops_lib.create_sliding_window_attention(
            batch_size=cfg.global_batch_size,
            q_seqlen=cfg.input_seqlen,
            sliding_window_size=cfg.sliding_window_size,
            num_heads=cfg.num_heads,
            num_kv_heads=cfg.num_kv_heads,
            d_model=cfg.d_model,
            d_head=cfg.d_head,
            num_layers=1,
            config=cfg,
            is_decode=True,
            use_flash_attention=cfg.use_flash_attention,
        )
    return ops_lib.create_multi_head_attention(
        batch_size=cfg.global_batch_size,
        input_seqlen=cfg.input_seqlen,
        output_seqlen=cfg.output_seqlen,
        decode_width=cfg.decode_width,
        num_heads=cfg.num_heads,
        d_model=cfg.d_model,
        d_head=cfg.d_head,
        num_layers=1,
        config=cfg,
        is_decode=True,
        use_flash_attention=cfg.use_flash_attention,
        num_kv_heads=cfg.num_kv_heads,
    )


def _ffn_ops(cfg: GptOssConfig, is_decode: bool):
    return ops_lib.create_ffn(
        batch_size=cfg.global_batch_size,
        input_seqlen=cfg.input_seqlen,
        output_seqlen=cfg.output_seqlen,
        decode_width=cfg.decode_width,
        d_model=cfg.d_model,
        d_ff=cfg.d_ff,
        num_layers=1,
        config=cfg,
        ffn_type="deepseek_moe",
        is_decode=is_decode,
    )


def _layer_input_shape(cfg: GptOssConfig, is_decode: bool) -> str:
    seq = cfg.decode_width if is_decode else cfg.input_seqlen
    return _fmt_shape([cfg.global_batch_size, seq, cfg.d_model])


def _build_prefill_attention_graph(prefix: str, attn_ops: list, out: list[str]) -> tuple[str, str]:
    q, k, v, flash, out_proj, ln = attn_ops
    attn_id = f"{prefix}_attn"
    out_id = f"{prefix}_out"
    ln_id = f"{prefix}_ln"

    out.extend(
        [
            _make_node(
                attn_id,
                "Attention Block",
                [
                    f"Input: {_fmt_shape(q.input_tensors[0].shape)}",
                    f"Wq: {_fmt_shape(q.input_tensors[1].shape)} -> Q {_fmt_shape(q.output_tensors[0].shape)}",
                    f"Wk: {_fmt_shape(k.input_tensors[1].shape)} -> K {_fmt_shape(k.output_tensors[0].shape)}",
                    f"Wv: {_fmt_shape(v.input_tensors[1].shape)} -> V {_fmt_shape(v.output_tensors[0].shape)}",
                    f"Attn core: {_fmt_shape(flash.output_tensors[0].shape)}",
                    f"Wo: {_fmt_shape(out_proj.input_tensors[1].shape)} -> {_fmt_shape(out_proj.output_tensors[0].shape)}",
                ],
                "#d5e8d4",
            ),
            _make_node(
                out_id,
                "Attn Residual",
                [
                    f"Attn out: {_fmt_shape(out_proj.output_tensors[0].shape)}",
                    f"Residual: {_fmt_shape(q.input_tensors[0].shape)}",
                    f"Output: {_fmt_shape(out_proj.output_tensors[0].shape)}",
                ],
                "#f5f5f5",
            ),
            _make_node(
                ln_id,
                "Post-Attn RMSNorm",
                [
                    f"Input: {_fmt_shape(ln.input_tensors[0].shape)}",
                    f"Output: {_fmt_shape(ln.output_tensors[0].shape)}",
                ],
                "#fff2cc",
            ),
        ]
    )
    out.extend([f"{prefix}_rms -> {attn_id};", f"{attn_id} -> {out_id};", f"{out_id} -> {ln_id};"])
    return ln_id, _fmt_shape(ln.output_tensors[0].shape)


def _build_decode_attention_graph(prefix: str, attn_ops: list, out: list[str]) -> tuple[str, str]:
    q, kv, qk_pre, qk_suf, softmax, av_pre, av_suf, add, out_proj, ln = attn_ops
    attn_id = f"{prefix}_attn"
    out_id = f"{prefix}_out"
    ln_id = f"{prefix}_ln"
    out.extend(
        [
            _make_node(
                attn_id,
                "Decode Attention Block",
                [
                    f"Input: {_fmt_shape(q.input_tensors[0].shape)}",
                    f"Wq: {_fmt_shape(q.input_tensors[1].shape)} -> Q {_fmt_shape(q.output_tensors[0].shape)}",
                    f"Wkv: {_fmt_shape(kv.input_tensors[1].shape)} -> New KV {_fmt_shape(kv.output_tensors[0].shape)}",
                    f"K cache: {_fmt_shape(qk_pre.input_tensors[1].shape)} + {_fmt_shape(qk_suf.input_tensors[1].shape)}",
                    f"V cache: {_fmt_shape(av_pre.input_tensors[1].shape)} + {_fmt_shape(av_suf.input_tensors[1].shape)}",
                    f"Scores: {_fmt_shape(softmax.output_tensors[0].shape)}",
                    f"Attn out: {_fmt_shape(add.output_tensors[0].shape)}",
                    f"Wo: {_fmt_shape(out_proj.input_tensors[1].shape)} -> {_fmt_shape(out_proj.output_tensors[0].shape)}",
                ],
                "#d5e8d4",
            ),
            _make_node(
                out_id,
                "Attn Residual",
                [
                    f"Attn out: {_fmt_shape(out_proj.output_tensors[0].shape)}",
                    f"Residual: {_fmt_shape(q.input_tensors[0].shape)}",
                    f"Output: {_fmt_shape(out_proj.output_tensors[0].shape)}",
                ],
                "#f5f5f5",
            ),
            _make_node(
                ln_id,
                "Post-Attn RMSNorm",
                [
                    f"Input: {_fmt_shape(ln.input_tensors[0].shape)}",
                    f"Output: {_fmt_shape(ln.output_tensors[0].shape)}",
                ],
                "#fff2cc",
            ),
        ]
    )
    out.extend([f"{prefix}_rms -> {attn_id};", f"{attn_id} -> {out_id};", f"{out_id} -> {ln_id};"])
    return ln_id, _fmt_shape(ln.output_tensors[0].shape)


def _build_ffn_graph(prefix: str, ffn_ops: list, input_node: str, input_shape: str, out: list[str]) -> tuple[str, str]:
    router_linear = ffn_ops[0]
    router_softmax = ffn_ops[1]
    gate = ffn_ops[2]
    up = ffn_ops[3]
    down = ffn_ops[5]
    combine_weights = ffn_ops[18]
    combine_output = ffn_ops[19]
    residual = ffn_ops[20]

    router_id = f"{prefix}_router"
    experts_id = f"{prefix}_experts"
    combine_id = f"{prefix}_combine"
    residual_id = f"{prefix}_residual"

    out.extend(
        [
            _make_node(
                router_id,
                "MoE Router",
                [
                    f"Input: {_fmt_shape(router_linear.input_tensors[0].shape)}",
                    f"Weight: {_fmt_shape(router_linear.input_tensors[1].shape)}",
                    f"Logits: {_fmt_shape(router_linear.output_tensors[0].shape)}",
                    f"Top-k weights: {_fmt_shape(router_softmax.output_tensors[0].shape)}",
                ],
                "#fff2cc",
            ),
            _make_node(
                experts_id,
                "Selected Experts x4",
                [
                    f"Input: {input_shape}",
                    f"Gate W: {_fmt_shape(gate.input_tensors[1].shape)}",
                    f"Up W: {_fmt_shape(up.input_tensors[1].shape)}",
                    f"Down W: {_fmt_shape(down.input_tensors[1].shape)}",
                    f"Each expert out: {_fmt_shape(down.output_tensors[0].shape)}",
                ],
                "#ffe6cc",
            ),
            _make_node(
                combine_id,
                "MoE Combine",
                [
                    f"Expert Stack: {_fmt_shape(combine_weights.input_tensors[0].shape)}",
                    f"Top-4 Weights: {_fmt_shape(combine_weights.input_tensors[1].shape)}",
                    f"Output: {_fmt_shape(combine_output.output_tensors[0].shape)}",
                ],
                "#e1d5e7",
            ),
            _make_node(
                residual_id,
                "Residual Add",
                [
                    f"Attn Path: {input_shape}",
                    f"FFN Path: {_fmt_shape(residual.input_tensors[1].shape)}",
                    f"Output: {_fmt_shape(residual.output_tensors[0].shape)}",
                ],
                "#f5f5f5",
            ),
        ]
    )
    out.extend(
        [
            f"{input_node} -> {router_id};",
            f"{input_node} -> {experts_id};",
            f"{router_id} -> {combine_id} [style=dashed, label=\"top-4\"];",
            f"{experts_id} -> {combine_id};",
            f"{combine_id} -> {residual_id};",
            f"{input_node} -> {residual_id};",
        ]
    )
    return residual_id, _fmt_shape(residual.output_tensors[0].shape)


def _build_phase_dot(cfg: GptOssConfig, phase: str) -> str:
    is_decode = phase == "decode"
    lines = [
        "digraph G {",
        'graph [rankdir=TB, splines=ortho, pad=0.2, nodesep=0.35, ranksep=0.5, bgcolor="white"];',
        'node [shape=box, style="rounded,filled", fontname="Helvetica"];',
        'edge [fontname="Helvetica", fontsize=9, color="#555555"];',
        f'label="GPT-OSS-20B Two-Layer {phase.capitalize()} Graph\\nbatch={cfg.global_batch_size}, input_seqlen={cfg.input_seqlen}, output_seqlen={cfg.output_seqlen}";',
        'labelloc=t;',
        'fontsize=18;',
    ]

    prev_output_node = None
    prev_output_shape = None
    for layer_idx, layer_type in enumerate(["sliding", "full"]):
        prefix = f"{phase}_l{layer_idx}"
        layer_label = f"Layer {layer_idx}: {'Sliding Window Attention' if layer_type == 'sliding' else 'Full Attention'}"
        lines.append(f"subgraph cluster_{prefix} {{")
        lines.append(f'label="{layer_label}";')
        lines.append('color="#b7b7b7"; style="rounded";')

        layer_input_shape = prev_output_shape or _layer_input_shape(cfg, is_decode)
        input_id = f"{prefix}_input"
        rms_id = f"{prefix}_rms"
        lines.append(_make_tensor(input_id, f"X{layer_idx}", layer_input_shape))
        lines.append(
            _make_node(
                rms_id,
                "RMSNorm",
                [f"Input: {layer_input_shape}", f"Output: {layer_input_shape}"],
                "#fff2cc",
            )
        )
        lines.append(f"{input_id} -> {rms_id};")

        if is_decode:
            attn_ops = _attention_decode_ops(cfg, layer_type)
            attn_output_node, attn_output_shape = _build_decode_attention_graph(prefix, attn_ops, lines)
        else:
            attn_ops = _attention_prefill_ops(cfg, layer_type)
            attn_output_node, attn_output_shape = _build_prefill_attention_graph(prefix, attn_ops, lines)

        ffn_ops = _ffn_ops(cfg, is_decode)
        layer_output_node, layer_output_shape = _build_ffn_graph(
            prefix, ffn_ops, attn_output_node, attn_output_shape, lines
        )
        output_id = f"{prefix}_output"
        lines.append(_make_tensor(output_id, f"X{layer_idx + 1}", layer_output_shape))
        lines.append(f"{layer_output_node} -> {output_id};")

        lines.append("}")

        if prev_output_node is not None:
            lines.append(f"{prev_output_node} -> {input_id} [penwidth=2, color=\"#888888\"];")

        prev_output_node = output_id
        prev_output_shape = layer_output_shape

    lines.append("}")
    return "\n".join(lines)


def _write_graph(dot_text: str, dot_path: Path, svg_path: Path) -> None:
    dot_path.write_text(dot_text)
    try:
        subprocess.run(
            ["dot", "-Tsvg", str(dot_path), "-o", str(svg_path)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"Failed to render {svg_path.name} with Graphviz: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate two-layer GPT-OSS computational graphs for prefill and decode."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--input_seqlen", type=int, default=DEFAULT_INPUT_SEQLEN)
    parser.add_argument("--output_seqlen", type=int, default=DEFAULT_OUTPUT_SEQLEN)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    with open(args.model) as f:
        cfg = GptOssConfig.model_validate(json.load(f))
    cfg = cfg.model_copy(
        update={
            "input_seqlen": args.input_seqlen,
            "output_seqlen": args.output_seqlen,
            "global_batch_size": args.batch_size,
            "microbatch_size_ici": args.batch_size,
            "microbatch_size_dcn": args.batch_size,
            "num_chips": 1,
            "data_parallelism_degree": 1,
            "tensor_parallelism_degree": 1,
            "pipeline_parallelism_degree": 1,
            "expert_parallelism_degree": 1,
            "data_parallel_degree_dcn": 1,
            "tensor_parallel_degree_dcn": 1,
            "pipeline_parallel_degree_dcn": 1,
            "expert_parallel_degree_dcn": 1,
            "decode_width": 1,
        }
    )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for phase in ["prefill", "decode"]:
        dot_text = _build_phase_dot(cfg, phase)
        dot_path = output_dir / f"gptoss_two_layers_{phase}.dot"
        svg_path = output_dir / f"gptoss_two_layers_{phase}.svg"
        _write_graph(dot_text, dot_path, svg_path)
        print(f"Wrote {dot_path}")
        print(f"Wrote {svg_path}")


if __name__ == "__main__":
    main()
