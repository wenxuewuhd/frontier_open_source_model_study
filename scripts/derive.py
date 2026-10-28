#!/usr/bin/env python3
"""推导：从 config 与权重清单算出负载特征，并把两条路的差异摊出来。

两条路：
  config 路   套公式算。能覆盖 KV、FLOPs、通信这些 shapes 里没有的量。
  shapes 路   逐张量求和。是实测值，参数量和权重存储以它为准。

两条路都能算的量（参数量、层数、hidden）一律做交叉核对，不一致就报到
reconciliation 里 —— 这是人工 Gate B 唯一要看的东西。config 字段与 checkpoint
不符是真实存在的情况，静默取其一会让后面所有结论都带错。

产出 data/derived/<id>.json，不写回模型 YAML：derived 与手写事实分文件存放，
「推导值不手写」这条规则才是结构上成立的，而不是靠纪律。

用法:
  python3 scripts/derive.py --id deepseek-v41-flash
  python3 scripts/derive.py --id <id> --facts-dir facts --model data/models/deepseek/x.yaml
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from math import prod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import dtypes  # noqa: E402

# 张量名 -> 子系统。顺序有意义，先匹配到的赢。
SUBSYSTEM = [
    ("retrieval", re.compile(r"engram|memory_table|retriev", re.I)),
    ("vision", re.compile(r"vision|visual|vit\.|aligner|image_", re.I)),
    ("audio", re.compile(r"audio|speech|whisper", re.I)),
    ("draft", re.compile(r"mtp|draft|dspark|eagle", re.I)),
    ("expert", re.compile(r"\.experts?\.\d+\.|\bexperts\.", re.I)),
    ("shared_expert", re.compile(r"shared_expert", re.I)),
    ("router", re.compile(r"\bgate\b|router|e_score|expert_bias", re.I)),
    ("embed", re.compile(r"embed_tokens|\blm_head\b|\bwte\b", re.I)),
    ("attention", re.compile(r"attn|\bwq|\bwk|\bwv|\bwo\b|q_proj|k_proj|v_proj|o_proj|kv_[ab]|q_[ab]|indexer", re.I)),
    ("dense_ffn", re.compile(r"mlp|ffn|gate_proj|up_proj|down_proj|w1|w2|w3", re.I)),
    ("norm", re.compile(r"norm|\bln\b|layernorm", re.I)),
]
_EXPERT_IDX = re.compile(r"experts?\.(\d+)\.", re.I)
_LAYER_IDX = re.compile(r"layers?\.(\d+)\.", re.I)


def classify(name: str) -> str:
    for label, rx in SUBSYSTEM:
        if rx.search(name):
            return label
    return "other"


def numel(shape) -> float:
    return float(prod(shape)) if shape else 0.0


# ---------------------------------------------------------------- shapes 路

def from_shapes(shapes: dict, logical: dict = None) -> dict:
    """逐张量求和。

    三个口径互不混用：
      参数量  排除 scale；低位打包的按因子还原（numel 是容器数，不是参数数）
      字节数  包含 scale；不还原打包（容器就是真实存储）
      元素数  safetensors 里的原始 numel，仅用于核对

    logical 给出每个子系统声明的逻辑精度（来自 precision.quantization.per_subsystem），
    用来识别打包。没给就按不打包处理，并在 packing_suspect 里提示可疑之处。
    """
    logical = logical or {}
    if not shapes:
        return {"available": False}

    params = {}
    nbytes = {}
    by_dtype = {}
    unknown_dtypes = set()
    packing_applied = {}
    packing_suspect = set()
    expert_ids, layer_ids, moe_layers = set(), set(), set()

    for name, meta in shapes.items():
        n = numel(meta.get("shape"))
        dt = dtypes.norm(meta.get("dtype", ""))
        b = dtypes.nbytes(dt)
        if b == 0.0:
            unknown_dtypes.add(meta.get("dtype"))
        sub = classify(name)

        nbytes[sub] = nbytes.get(sub, 0.0) + n * b
        by_dtype[dt] = by_dtype.get(dt, 0.0) + n * b
        if not dtypes.is_scale_tensor(name, dt):
            f = dtypes.packing_factor(logical.get(sub, ""), dt)
            if f > 1:
                packing_applied[sub] = f
            elif dt in ("INT8", "UINT8") and sub not in logical:
                # 整型容器里装低位权重很常见。没声明逻辑精度就无法还原，
                # 参数量会少算整数倍，必须提示而不是默默算小。
                packing_suspect.add(sub)
            params[sub] = params.get(sub, 0.0) + n * f

        m = _EXPERT_IDX.search(name)
        if m:
            expert_ids.add(int(m.group(1)))
        m = _LAYER_IDX.search(name)
        if m:
            layer_ids.add(int(m.group(1)))
            if sub in ("expert", "shared_expert"):
                moe_layers.add(int(m.group(1)))

    # 抽样快照检测：层索引稀疏说明只读了部分 shard。
    # 做算子表只要形状，抽样够用；拿来算参数总量会静默少算一大截，必须拦住。
    layer_max = (max(layer_ids) + 1) if layer_ids else 0
    complete = bool(layer_ids) and len(layer_ids) == layer_max
    missing = sorted(set(range(layer_max)) - layer_ids) if layer_ids else []

    return {
        "available": True,
        "complete": complete,
        "tensor_count": len(shapes),
        "params_total": sum(params.values()),
        "params_by_subsystem": params,
        "weight_bytes_total": sum(nbytes.values()),
        "weight_bytes_by_subsystem": nbytes,
        "weight_bytes_by_dtype": by_dtype,
        "experts_seen": (max(expert_ids) + 1) if expert_ids else 0,
        "layers_present": len(layer_ids),
        "layer_index_max": layer_max,
        "layers_missing": missing[:20],
        "moe_layers_seen": len(moe_layers),
        "unknown_dtypes": sorted(x for x in unknown_dtypes if x),
        "packing_applied": packing_applied,
        "packing_suspect": sorted(packing_suspect),
        "incomplete_note": (
            f"只有 {len(layer_ids)} 层出现在清单里（索引最大 {layer_max}），"
            "这是抽样快照而非完整清单。形状可用，参数量与权重存储不可用。"
            "跑 fetch_hf.py 取全量后再算。" if not complete else None
        ),
    }


# ---------------------------------------------------------------- config 路

def text_cfg(cfg: dict) -> dict:
    """多模态模型把语言主干放在 text_config 下，拍平成一层方便取字段。"""
    t = dict(cfg.get("text_config") or {})
    for k, v in cfg.items():
        if k not in ("text_config", "vision_config", "audio_config") and k not in t:
            t[k] = v
    return t


def kv_per_token(t: dict, attn: dict, kv_dtype: str, scale_dtype: str = None) -> dict:
    """每 token 的 KV 字节。公式按注意力类型分支，用到哪些字段一并记下备查。"""
    layers = t.get("num_hidden_layers")
    if not layers:
        return {"status": "unknown", "why": "config 里没有 num_hidden_layers"}

    b = dtypes.nbytes(kv_dtype)
    if b == 0:
        return {"status": "unknown", "why": f"KV 精度 {kv_dtype} 未知"}

    atype = (attn.get("type") or "").upper()
    terms, per_layer = {}, None

    if atype == "MLA":
        lora = attn.get("kv_lora_rank") or t.get("kv_lora_rank")
        rope = attn.get("qk_rope_head_dim") or t.get("qk_rope_head_dim")
        if not lora:
            # 有的实现不设 kv_lora_rank，而是用 num_kv_heads=1 的单头 latent
            kvh = attn.get("num_kv_heads") or t.get("num_key_value_heads")
            hd = attn.get("head_dim") or t.get("head_dim")
            if kvh and hd:
                lora = kvh * hd
                terms["latent_from"] = f"num_kv_heads({kvh}) * head_dim({hd})"
        if lora and rope:
            per_layer = (lora + rope) * b
            terms.update({"latent": lora, "qk_rope_head_dim": rope,
                          "formula": "(latent + qk_rope_head_dim) * bytes"})
    elif atype in ("GQA", "MHA", "MQA"):
        kvh = attn.get("num_kv_heads") or t.get("num_key_value_heads")
        hd = attn.get("head_dim") or t.get("head_dim")
        if not hd and t.get("hidden_size") and t.get("num_attention_heads"):
            hd = t["hidden_size"] // t["num_attention_heads"]
        if kvh and hd:
            per_layer = 2 * kvh * hd * b
            terms = {"num_kv_heads": kvh, "head_dim": hd,
                     "formula": "2 * num_kv_heads * head_dim * bytes"}

    if per_layer is None:
        return {"status": "unknown", "why": f"注意力类型 {atype or '未声明'} 缺少必要字段",
                "note": "hybrid / linear 结构请在模型 YAML 里按层分布显式给出"}

    # 跨层 KV 共享：只有部分层产出 KV。按总层数乘会高估一个量级。
    share = attn.get("kv_sharing") or {}
    src = share.get("source_layers")
    eff_layers, caveat = layers, None
    if src:
        eff_layers = len(src)
        caveat = (f"只有 {eff_layers} 层产出 KV（共 {layers} 层），其余层跨层读取；"
                  "按总层数乘会高估")
    ratios = share.get("compress_ratios")

    out = {
        "status": "ok", "attention_type": atype, "kv_dtype": kv_dtype,
        "bytes_per_layer": per_layer,
        "layers": layers, "kv_producing_layers": eff_layers,
        "bytes_per_token": per_layer * eff_layers, "terms": terms,
    }
    if caveat:
        out["kv_sharing_note"] = caveat
    if ratios:
        # 逐层压缩比本公式表达不了，不替用户折算，直接降级并要求人工确认
        out["status"] = "partial"
        out["why"] = ("存在逐层压缩比（compress_ratios），本公式未建模。"
                      "此处数值只是未压缩上界，真实值请以官方披露或实测为准，"
                      "并记入模型 YAML 的 official 字段与算子表 accounting")
        out["compress_ratios"] = ratios

    # 稀疏注意力的 indexer 另有一份缓存，漏掉会低估 KV
    sp = attn.get("sparse") or {}
    inh = sp.get("index_n_heads") or t.get("index_n_heads")
    ihd = sp.get("index_head_dim") or t.get("index_head_dim")
    if inh and ihd:
        # indexer 也只装在部分层上，按总层数乘同样会高估
        isrc = share.get("index_source_layers") or t.get("index_source_layer_ids")
        ilayers = len(isrc) if isrc else layers
        extra = inh * ihd * b * ilayers
        out["indexer_bytes_per_token"] = extra
        out["indexer_layers"] = ilayers
        out["bytes_per_token"] += extra
        out["terms"]["indexer"] = (
            f"index_n_heads({inh}) * index_head_dim({ihd}) * bytes * "
            f"{'index_source_layers' if isrc else 'layers'}({ilayers})")

    if scale_dtype:
        out["scale_note"] = (
            f"scale 容器声明为 {scale_dtype}；若实现用了更宽的类型，"
            "每 token 会多出一份，属于可修的实现差异，记在算子表 accounting 里"
        )
    return out


def active_params(shp: dict, t: dict, attn: dict, moe: dict) -> dict:
    """每 token 激活的线性参数。不等于计算量，也不含归一化与查表。"""
    if not shp.get("available"):
        return {"status": "unknown", "why": "没有 shapes，无法按子系统拆分"}
    if not shp.get("complete"):
        return {"status": "unknown", "why": "权重清单是抽样快照，按子系统求和会少算",
                "detail": shp.get("incomplete_note")}

    p = shp["params_by_subsystem"]
    experts = moe.get("routed_experts") or t.get("n_routed_experts") or shp.get("experts_seen") or 0
    top_k = moe.get("top_k") or t.get("num_experts_per_tok") or 0

    expert_total = p.get("expert", 0.0)
    if experts and top_k and expert_total:
        expert_active = expert_total * top_k / experts
        ratio = f"{top_k}/{experts}"
    else:
        expert_active, ratio = expert_total, "全部（未识别到路由）"

    # 每 token 全走的部分：注意力、dense FFN、共享专家、router、norm、输出头
    always = sum(p.get(k, 0.0) for k in
                 ("attention", "dense_ffn", "shared_expert", "router", "norm", "other"))
    # embed 与 lm_head 只触及少量行/列，不计入「每 token 搬运的线性权重」
    # retrieval 表每 token 只读极少行，同理不计入，另由 lookup 单独记
    return {
        "status": "ok",
        "total": always + expert_active,
        "expert_active": expert_active,
        "expert_total": expert_total,
        "expert_ratio": ratio,
        "always_active": always,
        "excluded": {
            "embed+lm_head": p.get("embed", 0.0),
            "retrieval_table": p.get("retrieval", 0.0),
            "vision": p.get("vision", 0.0),
            "draft": p.get("draft", 0.0),
        },
        "note": "embed/输出头/检索表/视觉塔不计入：它们每 token 只触及极少行，"
                "计入会把 decode 的带宽需求算高一个量级",
    }


def comm_per_token(t: dict, moe: dict, hidden: int, act_dtype: str, ep: int) -> dict:
    """EP 部署下每 token 的 all-to-all 字节（dispatch + combine）。"""
    top_k = moe.get("top_k") or t.get("num_experts_per_tok")
    layers = t.get("num_hidden_layers")
    first_dense = moe.get("first_k_dense") or t.get("first_k_dense_replace") or 0
    if not (top_k and layers and hidden):
        return {"status": "unknown", "why": "缺 top_k / 层数 / hidden"}
    b = dtypes.nbytes(act_dtype)
    moe_layers = max(layers - first_dense, 0)
    per_layer = 2 * top_k * hidden * b  # 发出去再收回来
    return {
        "status": "ok", "ep_size": ep, "moe_layers": moe_layers,
        "bytes_per_token": per_layer * moe_layers,
        "bytes_per_layer": per_layer,
        "formula": "2(dispatch+combine) * top_k * hidden * bytes * moe_layers",
        "note": "EP=1 时为 0；跨机时该量按互联带宽折算成时间，是 decode 尾延迟的主因之一",
    }


# ---------------------------------------------------------------- 交叉核对

def reconcile(cfg_t: dict, shp: dict, declared: dict) -> list:
    """两条路都能算的量做核对。有差异就报，不替用户选一个。"""
    out = []

    def add(item, a, b, la, lb, tol=0.01):
        if a is None or b is None:
            out.append({"item": item, "status": "skip",
                        "why": f"{la if a is None else lb} 缺失"})
            return
        # YAML 里 748.0e9 这种写法会被 PyYAML 解析成字符串（指数必须带显式符号），
        # 这类静默的类型错误要报出来，不能让它往下传
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            out.append({"item": item, "status": "MISMATCH",
                        "why": f"数值解析失败：{la}={a!r} {lb}={b!r}。"
                               "YAML 里的科学计数法要写成 7.48e+11（指数带符号）"})
            return
        if not a or not b:
            out.append({"item": item, "status": "skip", "why": "有一侧为 0"})
            return
        d = (b - a) / a
        out.append({
            "item": item, la: a, lb: b,
            "delta_pct": round(d * 100, 2),
            "status": "ok" if abs(d) <= tol else "MISMATCH",
        })

    if shp.get("available"):
        add("路由专家数",
            cfg_t.get("n_routed_experts") or cfg_t.get("num_experts"),
            shp.get("experts_seen") or None, "config", "shapes", tol=0.0)
        if not shp.get("complete"):
            # 清单不完整时，参数量核对无意义 —— 不报假的 -63%，直接说清单不能用
            out.append({"item": "权重清单完整性", "status": "MISMATCH",
                        "why": shp.get("incomplete_note")})
            out.append({"item": "参数总量", "status": "skip",
                        "why": "清单不完整，跳过核对（否则会得到一个无意义的差值）"})
        else:
            add("层数", cfg_t.get("num_hidden_layers"), shp.get("layers_present") or None,
                "config", "shapes", tol=0.0)
            excl = declared.get("excludes") or []
            sub = shp.get("params_by_subsystem") or {}
            measured = shp["params_total"] - sum(sub.get(k, 0.0) for k in excl)
            label = ("shapes 实测(扣除 " + "+".join(excl) + ")") if excl else "shapes 实测"
            add("参数总量", declared.get("params_total"), measured,
                "YAML 声明", label, tol=0.02)
            if excl:
                out.append({"item": "口径说明", "status": "ok",
                            "why": f"声明值不含 {'+'.join(excl)}；"
                                   f"含在内的实测总量为 {shp['params_total']/1e9:.1f}B"})
        if shp.get("unknown_dtypes"):
            out.append({"item": "未知数据类型", "status": "MISMATCH",
                        "detail": shp["unknown_dtypes"],
                        "why": "这些类型没有字节数定义，权重存储被低估。请在 lib/dtypes.py 里补"})
    else:
        out.append({"item": "全部", "status": "skip", "why": "没有 shapes.json，只有 config 单路"})
    return out


# ---------------------------------------------------------------- 主流程

def load_yaml(path):
    from lib import yamlio
    return yamlio.load(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--facts-dir", default="facts")
    ap.add_argument("--model", default=None, help="模型 YAML；给了就用它的注意力类型和精度，并核对声明值")
    ap.add_argument("--out-dir", default="data/derived")
    ap.add_argument("--ep", type=int, default=1, help="通信量按这个 EP 规模算")
    ap.add_argument("--context-lens", default="4096,32768,131072",
                    help="按这些上下文长度给出 KV 容量")
    args = ap.parse_args()

    fd = os.path.join(args.facts_dir, args.id)
    cfg = json.load(open(os.path.join(fd, "cfg.json"))) if os.path.exists(os.path.join(fd, "cfg.json")) else {}
    shapes = json.load(open(os.path.join(fd, "shapes.json"))) if os.path.exists(os.path.join(fd, "shapes.json")) else {}
    if not cfg and not shapes:
        sys.exit(f"! {fd} 下既没有 cfg.json 也没有 shapes.json，先跑 fetch_hf.py")

    model = load_yaml(args.model) if args.model else {}
    arch = model.get("arch") or {}
    attn = arch.get("attention") or {}
    moe = arch.get("moe") or {}
    prec = model.get("precision") or {}

    t = text_cfg(cfg)
    hidden = arch.get("hidden") or t.get("hidden_size")
    kv_dtype = prec.get("kv_cache")
    kv_dtype_assumed = kv_dtype is None
    if kv_dtype_assumed:
        # 没披露就用模型 dtype 兜底，但必须标出来：KV 精度直接决定容量结论
        kv_dtype = t.get("torch_dtype") or cfg.get("dtype") or "BF16"
    act_dtype = prec.get("training") or cfg.get("dtype") or "BF16"

    logical = ((prec.get("quantization") or {}).get("per_subsystem") or {})
    shp = from_shapes(shapes, logical)
    kv = kv_per_token(t, attn, kv_dtype, prec.get("kv_cache_scale"))
    if kv_dtype_assumed and kv.get("status") in ("ok", "partial"):
        kv["dtype_assumed"] = True
        kv["dtype_assumed_note"] = (
            f"precision.kv_cache 未填，按模型 dtype {kv_dtype} 兜底。"
            "KV 精度直接决定容量结论，确认后请填进模型 YAML")
    act = active_params(shp, t, attn, moe)
    comm = comm_per_token(t, moe, hidden, act_dtype, args.ep)

    # decode 的线性 FLOPs 与算术强度。注意力部分随上下文变化，单列不混进来。
    flops = {}
    if act.get("status") == "ok":
        flops["linear_per_token"] = 2 * act["total"]
        flops["note"] = "仅线性矩阵乘；注意力、状态更新、路由与通信不在内"
    intensity = {}
    if act.get("status") == "ok" and kv.get("status") in ("ok", "partial") and shp.get("available"):
        # 搬运量按激活权重占总权重的比例折算实际字节（低位量化已体现在 shapes 里）
        frac = act["total"] / shp["params_total"] if shp["params_total"] else 0
        moved = shp["weight_bytes_total"] * frac
        intensity = {
            "decode_weight_bytes_per_token": moved,
            "decode_flops_per_byte": (flops.get("linear_per_token", 0) / moved) if moved else None,
            "note": "每 token 搬运的权重字节按激活参数占比折算；"
                    "该值远低于 1 说明 decode 是访存瓶颈而非算力瓶颈",
        }

    kv_at = {}
    if kv.get("status") in ("ok", "partial"):
        for L in [int(x) for x in args.context_lens.split(",") if x.strip()]:
            kv_at[str(L)] = kv["bytes_per_token"] * L

    declared = {"params_total": (model.get("params") or {}).get("total"),
                "excludes": (model.get("params") or {}).get("excludes")}
    rec = reconcile(t, shp, declared)

    out = {
        "id": args.id,
        "generated_by": "scripts/derive.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "facts_dir": fd,
            "config_present": bool(cfg),
            "shapes_present": bool(shapes),
            "model_yaml": args.model,
            "ep_size": args.ep,
        },
        "from_shapes": shp,
        "kv_cache": kv,
        "kv_capacity_bytes_at_context": kv_at,
        "active_params": act,
        "flops": flops,
        "arithmetic_intensity": intensity,
        "ep_communication": comm,
        "reconciliation": rec,
    }

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"{args.id}.json")
    with open(path, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)

    # 给人看的摘要：Gate B 看的就是这一屏
    print(f"\n=== {args.id}  推导摘要 ===")
    if shp.get("available") and not shp.get("complete"):
        print(f"权重清单         ! 抽样快照（{shp['layers_present']}/{shp['layer_index_max']} 层，"
              f"{shp['tensor_count']} 个张量）—— 形状可用，参数量与存储不可用")
    if shp.get("available") and shp.get("complete"):
        print(f"参数总量(实测)   {dtypes.fmt_count(shp['params_total'])}"
              f"    权重存储 {dtypes.fmt_bytes(shp['weight_bytes_total'])}")
        top = sorted(shp["weight_bytes_by_subsystem"].items(), key=lambda kv_: -kv_[1])[:5]  # noqa
        print("  按子系统       " + ", ".join(f"{k} {dtypes.fmt_bytes(v)}" for k, v in top))
        print("  按精度         " + ", ".join(
            f"{k} {dtypes.fmt_bytes(v)}" for k, v in
            sorted(shp["weight_bytes_by_dtype"].items(), key=lambda kv_: -kv_[1])))
    if act.get("status") == "ok":
        print(f"激活线性参数     {dtypes.fmt_count(act['total'])}  (专家 {act['expert_ratio']})")
    if kv.get("status") in ("ok", "partial"):
        warn = " [精度兜底]" if kv.get("dtype_assumed") else ""
        warn += " [逐层压缩未建模,此为上界]" if kv.get("status") == "partial" else ""
        print(f"KV / token       {kv['bytes_per_token']:.0f} B  "
              f"({kv['attention_type']}, {kv['kv_dtype']}, "
              f"KV源层 {kv.get('kv_producing_layers')}/{kv.get('layers')}"
              f"{', indexer层 ' + str(kv['indexer_layers']) if kv.get('indexer_layers') else ''}){warn}")
        for L, v in kv_at.items():
            print(f"    @{int(L):>7d} tok  {dtypes.fmt_bytes(v)}")
    else:
        print(f"KV / token       未知 — {kv.get('why')}")
    if comm.get("status") == "ok" and args.ep > 1:
        print(f"EP all-to-all    {comm['bytes_per_token']:.0f} B/token  (EP={args.ep})")
    if intensity.get("decode_flops_per_byte"):
        print(f"算术强度(decode) {intensity['decode_flops_per_byte']:.2f} FLOP/B")

    bad = [r for r in rec if r.get("status") == "MISMATCH"]
    print(f"\n--- 交叉核对 ({len(bad)} 处不一致) ---")
    for r in rec:
        mark = {"ok": "  ok ", "skip": " skip", "MISMATCH": " !!! "}[r["status"]]
        extra = r.get("why") or r.get("detail") or ""
        nums = " ".join(f"{k}={v}" for k, v in r.items()
                        if k not in ("item", "status", "why", "detail"))
        print(f"{mark} {r['item']}: {nums} {extra}")
    print(f"\n-> {path}")
    if bad:
        print("\n! 有不一致项。这是 Gate B 要看的内容：先弄清差异根因，再往下做分析。")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
