#!/usr/bin/env python3
"""脚手架：从 config 生成模型记录草稿。

能从 config 机械推出的字段（层数、注意力形状、MoE 配置、量化、上下文）
由脚本填并标 official；判断类字段（瓶颈、infra 信号、机制命名）留 TODO 由人补。

这样做的理由：手写这些字段既慢又容易抄错，而它们本来就是 config 的搬运。
把搬运交给脚本，人只花精力在真正需要判断的地方 —— 也正是核查该盯的地方。

配置布局有两种，都要认：
  扁平   所有字段在顶层（DeepSeek V2–V4）
  嵌套   语言主干在 text_config、视觉在 vision_config（DeepSeek V4.1、Qwen3.8）

用法:
  python3 scripts/scaffold.py --id ds-v3 --family deepseek --repo deepseek-ai/DeepSeek-V3
  python3 scripts/scaffold.py --id ds-v3 --family deepseek --repo ... -o data/models/deepseek/ds-v3.yaml
"""
import argparse
import json
import os
import sys
from datetime import date

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TODO = "TODO"


def layout(cfg: dict):
    """把两种 config 布局归一成 (主干字段, 视觉字段, 量化字段, 原始)。"""
    text = dict(cfg.get("text_config") or {})
    for k, v in cfg.items():
        if k not in ("text_config", "vision_config", "audio_config") and k not in text:
            text[k] = v
    vision = dict(cfg.get("vision_config") or {})
    if not vision:
        # 扁平布局：视觉字段以 vision_ 前缀散落在顶层
        pref = {k[len("vision_"):]: v for k, v in cfg.items() if k.startswith("vision_")}
        if pref:
            vision = pref
    return text, vision, dict(cfg.get("quantization_config") or {}), cfg


def pick(d: dict, *keys):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return None


def build_attention(t: dict) -> dict:
    """注意力块。类型由可用字段判断，不靠 model_type 猜。"""
    a = {}
    heads = pick(t, "num_attention_heads")
    kvh = pick(t, "num_key_value_heads")
    hd = pick(t, "head_dim")
    if not hd and t.get("qk_nope_head_dim") and t.get("qk_rope_head_dim"):
        hd = t["qk_nope_head_dim"] + t["qk_rope_head_dim"]
    lora = pick(t, "kv_lora_rank")

    # MLA 的判据是存在 KV 低秩压缩，或 kv 头数被压到 1 而 head_dim 很大
    if lora:
        a["type"] = "MLA"
    elif kvh == 1 and hd and hd >= 256:
        a["type"] = "MLA"
    elif t.get("layer_types") or t.get("linear_num_value_heads"):
        a["type"] = "hybrid"
    elif kvh and heads and kvh < heads:
        a["type"] = "GQA"
    elif kvh and heads and kvh == heads:
        a["type"] = "MHA"

    for key, src in [
        ("num_heads", "num_attention_heads"), ("num_kv_heads", "num_key_value_heads"),
        ("qk_rope_head_dim", "qk_rope_head_dim"), ("qk_nope_head_dim", "qk_nope_head_dim"),
        ("kv_lora_rank", "kv_lora_rank"), ("q_lora_rank", "q_lora_rank"),
        ("o_lora_rank", "o_lora_rank"), ("o_groups", "o_groups"),
        ("sliding_window", "sliding_window"),
        ("partial_rotary_factor", "partial_rotary_factor"),
    ]:
        if t.get(src) is not None:
            a[key] = t[src]
    if hd:
        a["head_dim"] = hd

    # 稀疏选块
    if t.get("index_n_heads"):
        a["sparse"] = {
            "mechanism": TODO,
            "index_n_heads": t["index_n_heads"],
            "index_head_dim": t.get("index_head_dim"),
            "index_topk": t.get("index_topk"),
        }
        for k, src in [("candidate_source_layer", "candidate_source_layer_id"),
                       ("candidate_topk_blocks", "candidate_topk_blocks"),
                       ("candidate_block_size", "candidate_block_size")]:
            if t.get(src) is not None:
                a["sparse"][k] = t[src]

    # 跨层 KV 与逐层压缩。两者不是一回事：压缩比决定每层存多少，
    # 源层列表决定哪些层才真正产出 KV，漏掉后者会让 KV 估算高一个量级。
    share = {}
    if t.get("kv_source_layer_ids"):
        share["source_layers"] = t["kv_source_layer_ids"]
    if t.get("index_source_layer_ids"):
        # 故意不填 index_kv_source_layers：config 的这个列表把「产出 index K 的层」和
        # 「只复用已有 K 重新打分的层」混在一起，照搬会让 indexer 缓存高估一倍。
        # 两者只能靠权重区分（有没有 indexer.wk），脚手架没有权重，所以把问题留给人。
        share["note"] = (
            f"config 的 index_source_layer_ids = {t['index_source_layer_ids']}，"
            f"但它混了两类层。{TODO}：查权重清单里哪些层有 indexer.wk，"
            f"有的填 index_kv_source_layers，其余填 index_rescore_layers。"
            f"照搬整个列表会让 indexer 缓存高估一倍"
        )
    if t.get("compress_ratios"):
        share["compress_ratios"] = t["compress_ratios"]
    if share:
        a["kv_sharing"] = share

    # 混合线性注意力
    if t.get("linear_num_value_heads"):
        a["linear_state"] = {
            "mechanism": TODO,
            "num_key_heads": t.get("linear_num_key_heads"),
            "num_value_heads": t.get("linear_num_value_heads"),
            "key_head_dim": t.get("linear_key_head_dim"),
            "value_head_dim": t.get("linear_value_head_dim"),
            "conv_kernel_dim": t.get("linear_conv_kernel_dim"),
            "state_dtype": (t.get("mamba_ssm_dtype") or "").upper().replace("FLOAT32", "FP32") or None,
        }
    if t.get("layer_types"):
        lt = t["layer_types"]
        a["hybrid"] = {
            "layer_types_count": {
                "linear": sum(1 for x in lt if "linear" in str(x) or "mamba" in str(x)),
                "full": sum(1 for x in lt if "full" in str(x)),
            },
            "interval": t.get("full_attention_interval"),
        }
    if t.get("attn_output_gate"):
        a["output_gate"] = {"enabled": True, "type": t.get("output_gate_type")}
    return {k: v for k, v in a.items() if v not in (None, {}, [])}


def build_record(cfg: dict, info: dict, mid: str, family: str, repo: str) -> dict:
    t, v, q, raw = layout(cfg)
    kind = "vlm" if v else "llm"

    arch = {
        "layers": pick(t, "num_hidden_layers"),
        "hidden": pick(t, "hidden_size"),
        "vocab": pick(t, "vocab_size"),
        "norm": "RMSNorm" if t.get("rms_norm_eps") is not None else TODO,
        "norm_eps": pick(t, "rms_norm_eps"),
        "activation": TODO,
        "position": TODO,
        "attention": build_attention(t),
    }

    experts = pick(t, "n_routed_experts", "num_experts")
    if experts:
        arch["ffn"] = {"type": "moe"}
        moe = {"routed_experts": experts, "top_k": pick(t, "num_experts_per_tok")}
        for key, src in [("shared_experts", "n_shared_experts"),
                         ("intermediate", "moe_intermediate_size"),
                         ("first_k_dense", "first_k_dense_replace"),
                         ("expert_groups", "n_group")]:
            if t.get(src) is not None:
                moe[key] = t[src]
        sf, tm = t.get("scoring_func"), t.get("topk_method")
        if sf or tm:
            moe["routing"] = f"{sf or '?'} 打分 / {tm or '?'} 选路"
        # dense 前缀层的 FFN 宽度也要记：它往往是全模型最宽的单个 GEMM
        # （DeepSeek V3 的 18432 是专家宽度 2048 的 9 倍），而早先这里写成 elif，
        # MoE 模型永远取不到 intermediate_size，这个大头就一直是空的
        if t.get("first_k_dense_replace") and t.get("intermediate_size"):
            moe["dense_intermediate"] = t["intermediate_size"]
            arch["ffn"]["layer_pattern"] = (
                f"第 0–{t['first_k_dense_replace'] - 1} 层稠密（intermediate "
                f"{t['intermediate_size']}），其余为 MoE")
        arch["moe"] = moe
    elif t.get("intermediate_size"):
        arch["ffn"] = {"type": "dense", "intermediate": t["intermediate_size"]}

    if t.get("hc_mult"):
        arch["residual"] = {"type": TODO, "multiplier": t["hc_mult"]}
    if t.get("engram_layer_ids"):
        arch["retrieval"] = {
            "mechanism": TODO,
            "layers": t["engram_layer_ids"],
            "table_rows": t.get("engram_num_embeddings"),
            "entry_dim": t.get("engram_head_dim"),
        }

    dec = {}
    if t.get("num_nextn_predict_layers") is not None:
        dec["mtp_layers"] = t["num_nextn_predict_layers"]
    if t.get("dspark_block_size"):
        spec = {"method": "DSpark", "num_tokens": t["dspark_block_size"]}
        if t.get("dspark_target_layer_ids"):
            spec["target_layers"] = t["dspark_target_layer_ids"]
        if t.get("dspark_n_routed_experts"):
            spec["own_moe"] = {"routed_experts": t["dspark_n_routed_experts"],
                               "top_k": t.get("dspark_num_experts_per_tok"),
                               "rank": t.get("dspark_markov_rank")}
        dec["speculative"] = spec
    if dec:
        arch["decoding"] = dec

    if v:
        arch["vision"] = {k: val for k, val in {
            "encoder": TODO,
            "layers": pick(v, "num_hidden_layers", "depth", "n_layers"),
            "hidden": pick(v, "hidden_size", "dim"),
            "num_heads": pick(v, "num_heads", "n_heads"),
            "patch_size": pick(v, "patch_size"),
            "downsample_ratio": pick(v, "downsample_ratio"),
            "spatial_merge_size": pick(v, "spatial_merge_size"),
            "temporal_patch_size": pick(v, "temporal_patch_size"),
            "out_hidden": pick(v, "out_hidden_size"),
            "max_image_tokens": pick(v, "max_image_tokens", "max_n_token"),
            "fusion": TODO,
        }.items() if val is not None}

    prec = {}
    if q.get("quant_method"):
        qq = {"method": q["quant_method"]}
        for key, src in [("activation_scheme", "activation_scheme"),
                         ("weight_block_size", "weight_block_size")]:
            if q.get(src) is not None:
                qq[key] = q[src]
        # fmt 是量化权重的元素格式，scale_fmt 是 scale 张量的格式 —— 两个不同的东西。
        # 用 `scale_fmt or fmt` 兜底会把 fmt 写进 scale_format，凭空造出一次格式迁移
        # （DeepSeek V3 没声明 scale_fmt，V3.1 才加；兜底会让它看起来是 e4m3 → ue8m0）
        if q.get("fmt"):
            qq["weight_format"] = q["fmt"]
        if q.get("scale_fmt"):
            qq["scale_format"] = q["scale_fmt"]
        exp = t.get("expert_dtype") or q.get("expert_dtype")
        if exp:
            # 只写 expert_dtype（config 的声明值）。per_subsystem 留给权重清单实测的结果 ——
            # 把 config 值回填进去会让「声明」冒充「实测」，而两者经常对不上
            # （V4.1 的 config 说块是 [32,32]，实测专家是沿 K 的 1×32）
            qq["expert_dtype"] = exp.upper()
        prec["quantization"] = qq
    dt = (raw.get("dtype") or raw.get("torch_dtype") or "").upper().replace("BFLOAT16", "BF16")
    if dt:
        prec["training"] = TODO
        prec["weights_release"] = [TODO]

    ctx = {"max": pick(t, "max_position_embeddings")}
    rs = t.get("rope_scaling") or {}
    if rs.get("original_max_position_embeddings"):
        ctx["train_native"] = rs["original_max_position_embeddings"]
    if rs.get("rope_type"):
        ctx["extension_method"] = f"{rs['rope_type']}, factor {rs.get('factor')}"

    rec = {
        "id": mid,
        "family": family,
        "kind": kind,
        "display_name": repo.split("/")[-1],
        "released": (info.get("createdAt") or "")[:10] or TODO,
        "license": next((x.split(":", 1)[1] for x in info.get("tags", [])
                         if x.startswith("license:")), TODO),
        "visibility": "public",
        "sources": [{
            "kind": "modelcard",
            "url": f"https://huggingface.co/{repo}",
            "locator": f"config.json @ {(info.get('sha') or '')[:12]}",
            "accessed": date.today().isoformat(),
        }],
        "hf": {"repo": repo, "sha": info.get("sha"),
               "last_modified": info.get("lastModified"),
               "pipeline_tag": info.get("pipeline_tag"), "gated": info.get("gated", False)},
        "lineage": {"track": "main", "changed_from_predecessor": [{"axis": "scale", "text": TODO}]},
        "params": {"counting_note": TODO},
        "context": {k: v2 for k, v2 in ctx.items() if v2 is not None},
        "arch": {k: v2 for k, v2 in arch.items() if v2 not in (None, {}, [])},
        "precision": prec,
        "infra_signals": [{"text": TODO, "evidence": "judgement", "horizon": "watch"}],
        "evidence": {
            "id": "official", "family": "official", "kind": "derived",
            "display_name": "official", "released": "official", "license": "official",
            "hf": "official", "lineage": "judgement", "params": "judgement",
            "context": "official", "arch": "official",
            "arch.attention.type": "derived",
            "arch.norm": "derived", "arch.activation": "judgement", "arch.position": "judgement",
            "arch.moe.routing": "derived", "arch.ffn.type": "derived",
            "precision": "official", "precision.training": "judgement",
            "precision.weights_release": "judgement",
            "infra_signals": "judgement", "notes": "judgement", "open_questions": "judgement",
        },
        "notes": TODO,
        "open_questions": [TODO],
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--family", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--facts-dir", default="facts")
    ap.add_argument("-o", "--out", default=None)
    args = ap.parse_args()

    fd = os.path.join(args.facts_dir, args.id)
    cfg = json.load(open(os.path.join(fd, "cfg.json")))
    ip = os.path.join(fd, "repoinfo.json")
    info = json.load(open(ip)) if os.path.exists(ip) else {}

    rec = build_record(cfg, info, args.id, args.family, args.repo)

    import yaml
    out = args.out or os.path.join("data", "models", args.family, f"{args.id}.yaml")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(f"# 由 scripts/scaffold.py 从 config 生成的草稿。\n"
                f"# 标 {TODO} 的字段需要人工判断后填写，填完前这份记录不应合并。\n")
        yaml.safe_dump(rec, f, allow_unicode=True, sort_keys=False, width=100)
    todos = open(out).read().count(TODO)
    print(f"-> {out}  ({todos} 处待填)")


if __name__ == "__main__":
    main()
