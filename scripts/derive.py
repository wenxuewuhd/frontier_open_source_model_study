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
from lib.subsystems import classify  # noqa: E402

# 张量名 -> 子系统的归类在 lib/subsystems.py，fetch_dtypes.py 共用同一份 ——
# 两边用不同的归类会让「参数量里的 expert」和「精度表里的 expert」指的不是同一批张量。

_EXPERT_IDX = re.compile(r"experts?\.(\d+)\.", re.I)
_LAYER_IDX = re.compile(r"layers?\.(\d+)\.", re.I)




def numel(shape) -> float:
    return float(prod(shape)) if shape else 0.0


# ---------------------------------------------------------------- shapes 路

def from_shapes(shapes: dict, logical: dict = None, mtp_from: int = None) -> dict:
    """逐张量求和。

    三个口径互不混用：
      参数量  排除 scale；低位打包的按因子还原（numel 是容器数，不是参数数）
      字节数  包含 scale；不还原打包（容器就是真实存储）
      元素数  safetensors 里的原始 numel，仅用于核对

    logical 给出每个子系统声明的逻辑精度（来自 precision.quantization.per_subsystem），
    用来识别打包。没给就按不打包处理，并在 packing_suspect 里提示可疑之处。

    mtp_from 传 config 的 num_hidden_layers：V3 一代的 MTP 借用层索引，
    名字里没有 mtp 标记，不传的话那一层会整块算进主干。
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
    mtp_layer_ids = set()
    stacked_experts = set()

    for name, meta in shapes.items():
        n = numel(meta.get("shape"))
        dt = dtypes.norm(meta.get("dtype", ""))
        b = dtypes.nbytes(dt)
        if b == 0.0:
            unknown_dtypes.add(meta.get("dtype"))
        sub = classify(name, mtp_from)

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
        elif sub == "expert" and len(meta.get("shape") or []) == 3:
            # 堆叠存储：一个张量装一层的全部专家，专家数是首维。
            # 这种布局下 grouped GEMM 吃 3 维权重，算子签名与逐专家存储不同。
            stacked_experts.add(meta["shape"][0])
        m = _LAYER_IDX.search(name)
        if m:
            idx = int(m.group(1))
            # MTP 层不算主干层。V3 一代的 MTP 借用层索引（第 61 层，而主干是 0–60），
            # 算进去会让「层数」比 config 多一层，看起来像清单出了问题，其实是口径问题。
            if mtp_from is None or idx < mtp_from:
                layer_ids.add(idx)
                if sub in ("expert", "shared_expert"):
                    moe_layers.add(idx)
            else:
                mtp_layer_ids.add(idx)

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
        "experts_seen": ((max(expert_ids) + 1) if expert_ids
                         else (max(stacked_experts) if stacked_experts else 0)),
        "expert_storage": ("individual" if expert_ids
                           else "stacked" if stacked_experts else None),
        "layers_present": len(layer_ids),
        "mtp_layers_seen": sorted(mtp_layer_ids),
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


def _kv_out(comp, status, atype, kv_dtype, layers, full_layers, linear_layers, notes):
    return {
        "status": status, "attention_type": atype, "kv_dtype": kv_dtype,
        "layers": layers, "full_attention_layers": full_layers,
        "linear_attention_layers": linear_layers,
        "per_token_bytes": sum(c.get("bytes_per_token", 0) for c in comp.values()),
        "per_sequence_bytes": sum(c.get("bytes_per_sequence", 0) for c in comp.values()),
        "components": comp, "notes": notes,
    }


def kv_memory(t: dict, attn: dict, kv_dtype: str, scale_dtype: str = None) -> dict:
    """注意力的显存模型，分两个分量返回。

    混合结构里这两者的增长方式根本不同，混成一个数就没法做容量规划：
      per_token_bytes     随上下文线性增长 —— full attention 的 KV cache
      per_sequence_bytes  每序列固定、与上下文长度无关 —— 线性注意力的递归状态

    纯注意力模型 per_sequence_bytes 为 0；全线性模型 per_token_bytes 为 0。
    full attention 的口径按可用字段判断（有 kv_lora_rank 走 MLA，否则走 GQA），
    不依赖 attention.type 的声明，这样混合结构也能算。
    """
    layers = t.get("num_hidden_layers")
    if not layers:
        return {"status": "unknown", "why": "config 里没有 num_hidden_layers"}
    b = dtypes.nbytes(kv_dtype)
    if b == 0:
        return {"status": "unknown", "why": f"KV 精度 {kv_dtype} 未知"}

    atype = (attn.get("type") or "").upper()
    hyb = attn.get("hybrid") or {}
    share = attn.get("kv_sharing") or {}
    comp, notes, status = {}, [], "ok"

    # --- 层分布：先分清哪些层走 full attention、哪些走线性
    types = t.get("layer_types") or hyb.get("layer_types")
    if types:
        full_layers = sum(1 for x in types if "full" in str(x))
        linear_layers = sum(1 for x in types if "linear" in str(x) or "mamba" in str(x))
    else:
        cnt = hyb.get("layer_types_count") or {}
        linear_layers = cnt.get("linear", 0)
        full_layers = cnt.get("full", layers - linear_layers)

    # 跨层 KV 共享会覆盖 full 层数：只有部分层真正产出 KV
    src = share.get("source_layers")
    if src:
        full_layers = len(src)
        notes.append(f"只有 {full_layers} 层产出 KV（共 {layers} 层），其余层跨层读取；"
                     "按总层数乘会高估")

    # --- full attention：随上下文增长
    if full_layers:
        lora = attn.get("kv_lora_rank") or t.get("kv_lora_rank")
        rope = attn.get("qk_rope_head_dim") or t.get("qk_rope_head_dim")
        kvh = attn.get("num_kv_heads") or t.get("num_key_value_heads")
        hd = attn.get("head_dim") or t.get("head_dim")
        if not hd and t.get("hidden_size") and t.get("num_attention_heads"):
            hd = t["hidden_size"] // t["num_attention_heads"]
        per_layer, terms = None, {}
        if not lora and atype in ("MLA", "MQA-LATENT") and kvh and hd:
            # 不设 kv_lora_rank、而用 num_kv_heads=1 的单头 latent 的写法
            lora = kvh * hd
            terms["latent_from"] = f"num_kv_heads({kvh}) * head_dim({hd})"
            terms["rope_caveat"] = ("qk_rope_head_dim 是否在 latent 维之内未确认。"
                                    "若在其内，每层应为 latent*bytes 而非 (latent+rope)*bytes，"
                                    "当前取值为上界")
        if lora and rope:
            per_layer = (lora + rope) * b
            terms.update({"latent": lora, "qk_rope_head_dim": rope,
                          "formula": "(latent + qk_rope_head_dim) * bytes"})
        elif kvh and hd:
            per_layer = 2 * kvh * hd * b
            terms = {"num_kv_heads": kvh, "head_dim": hd,
                     "formula": "2 * num_kv_heads * head_dim * bytes"}
        if per_layer:
            comp["full_attention"] = {
                "layers": full_layers, "bytes_per_layer": per_layer,
                "bytes_per_token": per_layer * full_layers, "dtype": kv_dtype,
                "terms": terms,
            }
        else:
            status = "partial"
            notes.append(f"有 {full_layers} 层 full attention 但字段不足，这部分 KV 未计入")

    # --- 线性注意力：每序列固定状态
    ls = attn.get("linear_state") or {}
    nvh = ls.get("num_value_heads") or t.get("linear_num_value_heads")
    nkh = ls.get("num_key_heads") or t.get("linear_num_key_heads")
    kd = ls.get("key_head_dim") or t.get("linear_key_head_dim")
    vd = ls.get("value_head_dim") or t.get("linear_value_head_dim")
    conv = ls.get("conv_kernel_dim") or t.get("linear_conv_kernel_dim")
    sdt = dtypes.norm(ls.get("state_dtype") or t.get("mamba_ssm_dtype") or kv_dtype)
    sb = dtypes.nbytes(sdt)
    if linear_layers and nvh and kd and vd and sb:
        mat = nvh * kd * vd * sb
        cv = conv * (nkh or nvh) * kd * sb if conv else 0
        comp["linear_state"] = {
            "layers": linear_layers,
            "bytes_per_layer": mat + cv,
            "bytes_per_sequence": (mat + cv) * linear_layers,
            "dtype": sdt,
            "terms": {
                "matrix_state": f"num_value_heads({nvh}) * key_head_dim({kd})"
                                f" * value_head_dim({vd}) * bytes({sb})",
                "conv_state": f"conv_kernel_dim({conv}) * heads * key_head_dim" if conv else None,
            },
            "assumption": "递归状态按 [num_value_heads, key_head_dim, value_head_dim] 估，"
                          "conv 状态按卷积窗估。实际布局依实现而定，需核实后再用于容量规划",
        }
        status = "partial"
        notes.append("线性注意力的状态布局依实现而定，此处是按常见布局的估算")
    elif linear_layers:
        status = "partial"
        notes.append(f"有 {linear_layers} 层线性注意力但缺少 linear_* 字段，其状态未计入")

    # --- 实测优先：有完整权重清单量出来的每 token 元素数就直接用它。
    # 逐层压缩的结构（V4 一代的 compressor.ape）公式表达不了，而按未压缩宽度乘层数
    # 会高估五倍以上。量出来的数比任何通用公式都可靠，所以它一旦存在就接管。
    meas = share.get("measured_per_token") or {}
    if meas.get("global_kv") or meas.get("indexer"):
        comp.clear()
        if meas.get("global_kv"):
            comp["global_kv"] = {
                "bytes_per_token": meas["global_kv"] * b,
                "terms": {"elements_per_token": meas["global_kv"], "source": "完整权重清单实测"},
            }
        if meas.get("indexer"):
            comp["indexer"] = {
                "bytes_per_token": meas["indexer"] * b,
                "terms": {"elements_per_token": meas["indexer"], "source": "完整权重清单实测"},
            }
        if meas.get("local_window_bytes_per_sequence"):
            comp["local_window"] = {
                "bytes_per_sequence": meas["local_window_bytes_per_sequence"],
                "terms": {"note": "滑窗局部 KV，每序列固定，不随上下文增长"},
            }
        status = "measured"
        if meas.get("note"):
            notes.append(meas["note"])
        return _kv_out(comp, status, atype, kv_dtype, layers, full_layers, linear_layers, notes)

    # --- 稀疏注意力的 indexer 缓存：漏掉会低估
    sp = attn.get("sparse") or {}
    inh = sp.get("index_n_heads") or t.get("index_n_heads")
    ihd = sp.get("index_head_dim") or t.get("index_head_dim")
    ikv = sp.get("index_kv_dim")
    if (inh and ihd) or ikv:
        # 只有真正产出 index K 的层才占缓存；只重打分的层复用已有 K。
        # 把两者混在一起会让 indexer 缓存高估一倍 —— 而它往往是 KV 的大头。
        isrc = share.get("index_kv_source_layers")
        src_kind = "index_kv_source_layers"
        if not isrc:
            isrc = share.get("index_source_layers") or t.get("index_source_layer_ids")
            if isrc:
                src_kind = "index_source_layers"
                status = "partial"
                notes.append(
                    "indexer 层数取自 index_source_layers，它没区分「产出 index K」与"
                    "「只复用已有 K 重新打分」两类层，可能高估一倍。"
                    "查权重清单里哪些层有 indexer.wk，拆成 index_kv_source_layers 与 "
                    "index_rescore_layers")
            else:
                src_kind = "layers(全部层，未声明产出层)"
                status = "partial"
                notes.append(
                    f"没有任何 index 产出层声明，indexer 按全部 {layers} 层计 —— 这是上界。"
                    "实测 indexer 往往只在部分层上（DeepSeek V4 一代只在 compress_ratios==4 "
                    "的层，约一半）")
        ilayers = len(isrc) if isrc else layers

        # 宽度必须用实测的 index_kv_dim。index K 在各 index 头之间共享（MQA 式），
        # 产出它的张量输出维就是缓存宽度：V4.1 的 indexer.wk 是 [128, 512]，
        # 而 index_n_heads × index_head_dim = 32×128 = 4096 —— 差 32 倍。
        # 乘积只是上界，拿不到实测值时用它，但要标出来。
        if ikv:
            width, wsrc = ikv, f"index_kv_dim({ikv})，实测自 {sp.get('index_kv_source') or '产出张量输出维'}"
        else:
            width = inh * ihd
            wsrc = f"index_n_heads({inh}) * index_head_dim({ihd}) —— 上界，未填 index_kv_dim"
            status = "partial"
            notes.append(
                "indexer 缓存按 index_n_heads × index_head_dim 估，这是上界不是实际值："
                "index K 通常在各 index 头之间共享，实际宽度要取产出张量（indexer.wk 或 "
                "indexer.compressor.wkv）的输出维，填进 sparse.index_kv_dim")
        comp["indexer"] = {
            "layers": ilayers, "width": width, "bytes_per_token": width * b * ilayers,
            "terms": {"formula": f"{wsrc} * bytes * {src_kind}({ilayers})"},
        }

    # --- 逐层压缩比表达不了，降级而不是替人折算
    ratios = share.get("compress_ratios")
    if ratios:
        status = "partial"
        notes.append("存在逐层压缩比（compress_ratios），本公式未建模；"
                     "per_token 只是未压缩上界，真实值以官方披露或实测为准")

    if not comp:
        return {"status": "unknown",
                "why": f"注意力类型 {atype or '未声明'} 缺少必要字段",
                "note": "混合结构请在 YAML 的 attention.hybrid / linear_state 里显式给出"}

    out = _kv_out(comp, status, atype, kv_dtype, layers, full_layers, linear_layers, notes)
    if scale_dtype:
        out["scale_note"] = (
            f"scale 容器声明为 {scale_dtype}；实现若用了更宽的类型，每 token 会多一份，"
            "属可修的实现差异，记在算子表 accounting 里")
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
        expert_active, ratio = expert_total, ("dense，无 MoE" if not expert_total
                                               else "全部（未识别到路由）")

    # 每 token 全量搬运的部分。lm_head 必须计入：它对整个词表做全量 GEMM，
    # 不是「只触及少量行」—— 大词表模型里这一项可观。
    always = sum(p.get(k, 0.0) for k in
                 ("attention", "compressor", "indexer", "dense_ffn", "shared_expert", "router", "norm",
                  "lm_head", "other"))
    embed = p.get("embed", 0.0)
    return {
        "status": "ok",
        "total": always + expert_active,
        "total_with_embed": always + expert_active + embed,
        "semantics": "total 是带宽口径（每 token 真正搬运的线性权重，不含只 gather 一行的"
                     "输入词表）；total_with_embed 含输入词表，用于和各家公布的"
                     "「激活参数」对比 —— 两者不是同一个量，混用会同时犯数值和概念错误",
        "expert_active": expert_active,
        "expert_total": expert_total,
        "expert_ratio": ratio,
        "always_active": always,
        "lm_head": p.get("lm_head", 0.0),
        "dense": not bool(expert_total),
        "excluded": {
            "embed_tokens": p.get("embed", 0.0),
            "retrieval_table": p.get("retrieval", 0.0),
            "vision": p.get("vision", 0.0),
            "draft": p.get("draft", 0.0),
        },
        "note": "embed_tokens 只 gather 一行、检索表每 token 只读极少行、"
                "视觉塔与草稿模型不在主路径上，这四项不计入；"
                "lm_head 计入，因为它每步对全词表做 GEMM",
    }


def comm_per_token(t: dict, moe: dict, hidden: int, act_dtype: str, ep: int) -> dict:
    """EP 部署下每 token 的 all-to-all 字节（dispatch + combine）。"""
    top_k = moe.get("top_k") or t.get("num_experts_per_tok")
    layers = t.get("num_hidden_layers")
    first_dense = moe.get("first_k_dense") or t.get("first_k_dense_replace") or 0
    if not top_k:
        return {"status": "ok", "bytes_per_token": 0,
                "why": "dense 模型，无专家并行通信"}
    if not (layers and hidden):
        return {"status": "unknown", "why": "缺层数 / hidden"}
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
        n_exp = cfg_t.get("n_routed_experts") or cfg_t.get("num_experts")
        if not n_exp and not shp.get("experts_seen"):
            out.append({"item": "路由专家数", "status": "ok", "why": "dense 模型，无 MoE"})
        else:
            add("路由专家数", n_exp, shp.get("experts_seen") or None,
                "config", "shapes", tol=0.0)
            if shp.get("expert_storage"):
                out.append({"item": "专家存储布局", "status": "ok",
                            "why": f"{shp['expert_storage']}"
                                   + ("（一个张量装一层全部专家，grouped GEMM 吃 3 维权重）"
                                      if shp["expert_storage"] == "stacked" else "（逐专家独立张量）")})
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
    ap.add_argument("--context-lens", default=None,
                    help="按这些上下文长度给出 KV 容量；默认取 4K/32K 与模型自身上限")
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
    # 层索引 >= num_hidden_layers 的归草稿侧：V3 一代的 MTP 是「第 61 层」，
    # 名字里没有 mtp 标记，不这样分会把整层算进主干
    shp = from_shapes(shapes, logical, mtp_from=t.get("num_hidden_layers"))
    kv = kv_memory(t, attn, kv_dtype, prec.get("kv_cache_scale"))
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
        if args.context_lens:
            lens = [int(x) for x in args.context_lens.split(",") if x.strip()]
        else:
            # 默认带上模型自身的上下文上限 —— 容量结论最该看的就是满上下文那一档
            cap = (model.get("context") or {}).get("max") or t.get("max_position_embeddings")
            lens = sorted({x for x in [4096, 32768, cap] if x and x <= (cap or x)})
        for L in lens:
            kv_at[str(L)] = kv["per_sequence_bytes"] + kv["per_token_bytes"] * L

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
        if shp.get("expert_storage"):
            print(f"  专家存储       {shp['expert_storage']}"
                  f"（{shp.get('experts_seen')} 专家）")
        print("  按精度         " + ", ".join(
            f"{k} {dtypes.fmt_bytes(v)}" for k, v in
            sorted(shp["weight_bytes_by_dtype"].items(), key=lambda kv_: -kv_[1])))
    if act.get("status") == "ok":
        tail = ("dense，无 MoE" if act.get("dense") else f"专家 {act['expert_ratio']}")
        if act.get("lm_head"):
            tail += f"，含 lm_head {dtypes.fmt_count(act['lm_head'])}"
        print(f"激活(带宽口径)   {dtypes.fmt_count(act['total'])}  ({tail})")
        if act.get("total_with_embed", 0) > act.get("total", 0):
            print(f"激活(可比官方)   {dtypes.fmt_count(act['total_with_embed'])}"
                  f"  (含输入词表 {dtypes.fmt_count(act['excluded']['embed_tokens'])})")
    if kv.get("status") in ("ok", "partial"):
        warn = " [精度兜底]" if kv.get("dtype_assumed") else ""
        warn += " [含估算项]" if kv.get("status") == "partial" else ""
        fa = kv.get("full_attention_layers"); la = kv.get("linear_attention_layers")
        print(f"注意力显存       {kv['per_token_bytes']:.0f} B/token"
              f" + {dtypes.fmt_bytes(kv['per_sequence_bytes'])}/序列"
              f"  ({kv['attention_type']}, full {fa} 层 / 线性 {la} 层){warn}")
        for name, c in (kv.get("components") or {}).items():
            bit = (f"{c['bytes_per_token']:.0f} B/token" if c.get("bytes_per_token")
                   else f"{dtypes.fmt_bytes(c.get('bytes_per_sequence', 0))}/序列")
            print(f"    {name:16s} {c['layers']:>3d} 层  {bit}")
        for n in kv.get("notes") or []:
            print(f"    ! {n}")
        for L, v in kv_at.items():
            print(f"    @{int(L):>8d} tok  {dtypes.fmt_bytes(v)}")
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
