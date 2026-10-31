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
from lib import cfgio  # noqa: E402
from lib import subsystems  # noqa: E402
from lib.subsystems import classify, classify2  # noqa: E402

# 张量名 -> 子系统的归类在 lib/subsystems.py，fetch_dtypes.py 共用同一份 ——
# 两边用不同的归类会让「参数量里的 expert」和「精度表里的 expert」指的不是同一批张量。

_EXPERT_IDX = re.compile(r"experts?\.(\d+)\.", re.I)
# 层号锚点。生成类 DiT 普遍用 `blocks.N.`（Wan）或 `block.N.`（MAGI-2）而不是
# `layers.N.`，锚死在 layers 上会把完整清单判成抽样快照 —— 参数量与权重存储
# 会因此整块算不出来。视觉塔那边在调用处用 is_tower() 排除，所以 `vision.blocks.N.`
# 不会被咬进来（实扫 48 份记录，非塔的 blocks.N. 只有生成类记录有）。
_LAYER_IDX = re.compile(r"(?:layers?|blocks?)\.(\d+)\.", re.I)
# 第一个「.数字.」就是块号，不管这个块叫 layers 还是 mtp
_BLOCK_IDX = re.compile(r"\.(\d+)\.")




def numel(shape) -> float:
    return float(prod(shape)) if shape else 0.0


# ---------------------------------------------------------------- shapes 路

def from_shapes(shapes: dict, logical: dict = None, mtp_from: int = None) -> dict:
    """逐张量求和。

    三个口径互不混用：
      参数量  排除 scale；低位打包的按因子还原（numel 是容器数，不是参数数）
      字节数  包含 scale；不还原打包（容器就是真实存储）。但**按什么精度报**要用
              逻辑精度：FP4 专家打包在 INT8 容器里，报 INT8 是报了容器类型，
              读者会以为专家是 8 位。字节不变，标签改成 FP4，容器另记
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
    # 组件维：生成类的一份权重里同时装着 DiT / 文本编码器 / VAE / 声码器，
    # 先分组件再分子系统，否则 text_encoder 的 mlp 会和 DiT 的 FFN 合成一桶。
    # 语言模型的张量没有组件前缀，全部落 backbone —— 单组件时下面不输出这两个字段
    params_comp = {}
    bytes_comp = {}
    unknown_dtypes = set()
    packing_applied = {}
    params_container = {}
    dtype_containers = {}
    packing_suspect = set()
    expert_ids, layer_ids, moe_layers = set(), set(), set()
    mtp_layer_ids = set()
    stacked_experts = set()
    scale_elems = 0.0
    # 按「剥掉块号后的后缀」记两个集合，用来判断草稿块有没有自己的注意力。
    # 不用张量名硬匹配 —— 各家名字不同，而本仓栽在名字上已经不止一次。
    # 结构自相似是更稳的判据：草稿块若带着与主干注意力同名的那套投影，
    # 它就有自己的 KV，容量按主干层数算会少掉这些块。
    # 两代的块命名不同（V3 借用层号 layers.61.*，V4 是 mtp.0.*），所以按
    # 第一个 `.<数字>.` 剥前缀，不认具体名字
    trunk_attn_suffixes, draft_suffixes, draft_blocks = set(), set(), set()

    for name, meta in shapes.items():
        n = numel(meta.get("shape"))
        dt = dtypes.norm(meta.get("dtype", ""))
        b = dtypes.nbytes(dt)
        if b == 0.0:
            unknown_dtypes.add(meta.get("dtype"))
        comp, sub = classify2(name, mtp_from)

        bytes_comp[comp] = bytes_comp.get(comp, 0.0) + n * b
        if dtypes.counts_as_param(name, dt):
            params_comp[comp] = params_comp.get(comp, 0.0) + n * dtypes.packing_factor(
                logical.get(sub, ""), dt)

        nbytes[sub] = nbytes.get(sub, 0.0) + n * b
        # 字节按逻辑精度归类。scale 张量不参与：它本身就是那个精度，没有打包
        label = dt
        if not dtypes.is_scale_tensor(name, dt):
            lg = dtypes.norm(logical.get(sub, ""))
            if lg and dtypes.packing_factor(lg, dt) > 1:
                label = lg
                dtype_containers.setdefault(label, set()).add(dt)
        by_dtype[label] = by_dtype.get(label, 0.0) + n * b
        if dtypes.is_scale_tensor(name, dt):
            # 单独记下来，好跟外部的参数量 tally 对账：有的仓把 scale 算进了
            # 参数量，有的没算，两种都要能解释（见 reconcile 的外部对账一项）
            scale_elems += n
        if dtypes.counts_as_param(name, dt):
            f = dtypes.packing_factor(logical.get(sub, ""), dt)
            if f > 1:
                packing_applied[sub] = f
            elif dtypes.is_container(dt) and sub not in logical:
                # 整型容器里装低位权重很常见。没声明逻辑精度就无法还原，
                # 参数量会少算整数倍，必须提示而不是默默算小。
                packing_suspect.add(sub)
            params[sub] = params.get(sub, 0.0) + n * f
            # 容器口径：numel 原样累计，不乘打包因子。它与本仓的 packing_factor
            # 无关，所以能在 HF 的 tally 不还原打包时用作对账的另一侧 ——
            # 但也正因为无关，它**对打包因子本身没有任何约束力**（见 reconcile）
            params_container[sub] = params_container.get(sub, 0.0) + n

        m = _EXPERT_IDX.search(name)
        if m:
            expert_ids.add(int(m.group(1)))
        elif sub == "expert" and len(meta.get("shape") or []) == 3:
            # 堆叠存储：一个张量装一层的全部专家，专家数是首维。
            # 这种布局下 grouped GEMM 吃 3 维权重，算子签名与逐专家存储不同。
            stacked_experts.add(meta["shape"][0])
        if sub in ("attention", "draft"):
            mb = _BLOCK_IDX.search(name)
            if mb:
                if sub == "attention":
                    trunk_attn_suffixes.add(name[mb.end():])
                else:
                    draft_suffixes.add(name[mb.end():])
                    draft_blocks.add(mb.group(1))

        # 视觉 / 音频塔自己也有 layers.N.，而塔的层数可以超过语言主干（Step-3：塔 63 层、
        # 主干 61 层）。classify() 那边已经按 _TOWER 绕开了层号捷径，这里也必须绕 ——
        # 否则塔的第 61、62 层仍会进 mtp_layers_seen，在一个没有 MTP 的模型上留个假阳性；
        # 塔的 0–60 层也会混进 layer_ids，和主干层号搅在一起
        m = None if subsystems.is_tower(name) else _LAYER_IDX.search(name)
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
        "params_total_container": sum(params_container.values()),
        "params_container_by_subsystem": params_container,
        "scale_elements": scale_elems,
        "weight_bytes_total": sum(nbytes.values()),
        "weight_bytes_by_subsystem": nbytes,
        # 只有真跨组件时才输出。语言模型全部落 backbone，多一个恒等于总量的字段
        # 只会让页面多一行废话，也会让回归对比出现无意义的差异
        **({"params_by_component": params_comp,
            "weight_bytes_by_component": bytes_comp} if len(bytes_comp) > 1 else {}),
        "weight_bytes_by_dtype": by_dtype,
        # 哪些精度是打包存放的，以及装在什么容器里。页面要把两个值一起显示 ——
        # 只报 FP4 会让人以为能按 4 位寻址，只报 INT8 会让人以为专家是 8 位
        "weight_dtype_containers": {k: sorted(v) for k, v in dtype_containers.items()},
        "experts_seen": ((max(expert_ids) + 1) if expert_ids
                         else (max(stacked_experts) if stacked_experts else 0)),
        "expert_storage": ("individual" if expert_ids
                           else "stacked" if stacked_experts else None),
        "layers_present": len(layer_ids),
        "mtp_layers_seen": sorted(mtp_layer_ids),
        # 草稿块里与主干注意力同名的那套投影。非空说明草稿块自带注意力，
        # 于是它也产出 KV —— 按 config 的 num_hidden_layers 算容量会少掉这些块
        "mtp_attention_suffixes": sorted(trunk_attn_suffixes & draft_suffixes),
        "draft_blocks_seen": len(draft_blocks),
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
    """多模态模型把语言主干放在 text_config 下，拍平成一层方便取字段。

    实现在 lib/cfgio.py —— 读 config 的唯一入口。这里只留个名字，
    免得四个脚本各写一套读法（已经发生过一次，见那个模块的头注释）。
    """
    return cfgio.flatten(cfg)


def _kv_out(comp, status, atype, kv_dtype, layers, full_layers, linear_layers, notes):
    # 元素口径与精度无关，跨代比 KV 结构只能用它：两侧字节口径不同（一侧落盘精度、
    # 一侧 dtype 兜底）时，字节的倍数里混着精度假设的差，不是结构的差。
    # 任何一个随上下文增长的分量缺元素数，合计就不报，免得少算一项还像是精确值
    per_tok = [c for c in comp.values() if c.get("bytes_per_token")]
    elems = [c.get("elements_per_token") for c in per_tok]
    return {
        "status": status, "attention_type": atype, "kv_dtype": kv_dtype,
        "layers": layers, "full_attention_layers": full_layers,
        "linear_attention_layers": linear_layers,
        "per_token_bytes": sum(c.get("bytes_per_token", 0) for c in comp.values()),
        "per_sequence_bytes": sum(c.get("bytes_per_sequence", 0) for c in comp.values()),
        "per_token_elems": sum(elems) if per_tok and all(elems) else None,
        "per_token_elems_bound": ("upper_bound" if any(c.get("elements_upper_bound") for c in per_tok)
                                  else "exact"),
        "components": comp, "notes": notes,
    }


def _component_bpe(components: dict, key: str, fallback: float):
    """一个 KV 分量的每元素字节（含 scale）与口径。

    有 precision.kv_cache_components 就按它算 —— 元素数 × 落盘精度是精确计数，
    但不含分页、环页、对齐这些分配粒度，所以是理论下限，不是部署占用。
    没有就退回模型 dtype 兜底，口径另记，两种数不能混着比。
    """
    c = (components or {}).get(key)
    if not c:
        return fallback, "dtype_fallback", None
    d = dtypes.nbytes(c.get("dtype"))
    if d == 0:
        return fallback, "dtype_fallback", None
    blk = c.get("block")
    s = dtypes.nbytes(c.get("scale_dtype")) / blk if blk and c.get("scale_dtype") else 0.0
    return d + s, "theoretical_floor", c


# measured 是 R28 加的第三种可用状态。下游凡是判「KV 算出来了没有」的地方都要认它 ——
# 当初只改了 kv_memory 没改这几处，实测记录的容量表、精度兜底标记、算术强度全部静默为空
KV_USABLE = ("ok", "partial", "measured")

_FLOOR_NOTE = ("字节是理论下限：元素数 × 落盘精度（含 scale）的精确计数，不含分页、环页与对齐粒度。"
               "实际部署在它之上，不能直接当显存预算")


def kv_memory(t: dict, attn: dict, kv_dtype: str, scale_dtype: str = None,
              components: dict = None) -> dict:
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
    if b == 0 and not components:
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

    # --- full attention：随上下文增长。KV 精度是 MIXED 这类非单值时套不了公式，
    # 那种记录必然带 measured_per_token，交给下面的实测分支
    if full_layers and b:
        lora = attn.get("kv_lora_rank") or t.get("kv_lora_rank")
        rope = attn.get("qk_rope_head_dim") or t.get("qk_rope_head_dim")
        kvh = attn.get("num_kv_heads") or t.get("num_key_value_heads")
        hd = attn.get("head_dim") or t.get("head_dim")
        if not hd and t.get("hidden_size") and t.get("num_attention_heads"):
            hd = t["hidden_size"] // t["num_attention_heads"]
        per_layer, terms, upper = None, {}, False
        if not lora and atype in ("MLA", "MQA-LATENT") and kvh and hd:
            # 不设 kv_lora_rank、而用 num_kv_heads=1 的单头 latent 的写法
            lora = kvh * hd
            terms["latent_from"] = f"num_kv_heads({kvh}) * head_dim({hd})"
            terms["rope_caveat"] = ("qk_rope_head_dim 是否在 latent 维之内未确认。"
                                    "若在其内，每层应为 latent*bytes 而非 (latent+rope)*bytes，"
                                    "当前取值为上界")
            upper = True
        if lora and rope:
            el = lora + rope
            per_layer = el * b
            terms.update({"latent": lora, "qk_rope_head_dim": rope,
                          "formula": "(latent + qk_rope_head_dim) * bytes"})
        elif kvh and hd:
            el = 2 * kvh * hd
            per_layer = el * b
            terms = {"num_kv_heads": kvh, "head_dim": hd,
                     "formula": "2 * num_kv_heads * head_dim * bytes"}
        if per_layer:
            comp["full_attention"] = {
                "layers": full_layers, "bytes_per_layer": per_layer,
                "bytes_per_token": per_layer * full_layers, "dtype": kv_dtype,
                "elements_per_token": el * full_layers,
                "terms": terms,
            }
            if upper:
                comp["full_attention"]["elements_upper_bound"] = True
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
        # 短卷积是 depthwise，作用在整条 qkv 流上 —— 通道数 = q + k + v 三段之和，
        # 不是 key 那一段。原先写成 (nkh or nvh) * kd 只数了 key 一段，旗舰少 10 倍、
        # 27B 少 5 倍。权重清单给的恒等式把它钉死了：conv1d.weight 的首维
        # 恒等于 in_proj_qkv 的输出维（旗舰 69/69 层、27B 48/48 层，零例外）。
        # q 段按 key 头数展开（q 与 k 同形），v 段按 value 头数 —— 两个模型的
        # key/value 头数比不同（1:8 与 1:3），两档同时成立才说明这是规律不是巧合。
        qkv_ch = 2 * (nkh or nvh) * kd + nvh * vd
        cv = conv * qkv_ch * sb if conv else 0
        comp["linear_state"] = {
            "layers": linear_layers,
            "bytes_per_layer": mat + cv,
            "bytes_per_sequence": (mat + cv) * linear_layers,
            "dtype": sdt,
            "terms": {
                "matrix_state": f"num_value_heads({nvh}) * key_head_dim({kd})"
                                f" * value_head_dim({vd}) * bytes({sb})",
                "conv_state": (
                    f"conv_kernel_dim({conv}) * qkv_channels({qkv_ch}"
                    f" = 2 * {nkh or nvh} * {kd} + {nvh} * {vd}) * bytes({sb})"
                ) if conv else None,
            },
            "assumption": "递归状态按 [num_value_heads, key_head_dim, value_head_dim] 估；"
                          "conv 状态的通道数取自权重清单（= in_proj_qkv 输出维），"
                          "窗长按 conv_kernel_dim 全窗计，是上界 —— 实现若只存 k-1 个位置会略低。"
                          "矩阵状态那一侧的布局仍依实现而定，需核实后再用于容量规划",
        }
        status = "partial"
        notes.append("线性注意力的状态布局依实现而定，此处是按常见布局的估算")
    elif linear_layers:
        status = "partial"
        notes.append(f"有 {linear_layers} 层线性注意力但缺少 linear_* 字段，其状态未计入")

    # --- 实测优先：有完整权重清单量出来的每 token 元素数就直接用它。
    # 逐层压缩的结构（V4 一代的 compressor.ape）公式表达不了，而按未压缩宽度乘层数
    # 会高估五倍以上。量出来的数比任何通用公式都可靠，所以它一旦存在就接管。
    # 元素数和精度分开取：元素数实测自权重，精度往往另有来源（实现、文档），
    # 两者的可信度不同，乘出来的字节要带上精度那一侧的口径
    meas = share.get("measured_per_token") or {}
    if meas.get("global_kv") or meas.get("indexer"):
        comp.clear()
        bases = set()

        def put(key, elems, per):
            bpe, basis, fmt = _component_bpe(components, key, b)
            if not bpe:
                notes.append(f"{key} 没有落盘精度、模型 dtype 也无法兜底，未计入")
                return
            bases.add(basis)
            terms = {"elements" + per: elems, "bytes_per_element": bpe, "source": "完整权重清单实测"}
            if fmt:
                terms["storage"] = fmt
            comp[key] = {"bytes" + per: elems * bpe, "elements" + per: elems,
                         "bytes_basis": basis, "terms": terms}

        for key in ("global_kv", "indexer"):
            if meas.get(key):
                put(key, meas[key], "_per_token")
        if meas.get("local_window_elems_per_sequence"):
            put("local_window", meas["local_window_elems_per_sequence"], "_per_sequence")
            comp.get("local_window", {}).setdefault("terms", {})["note"] = "滑窗局部 KV，每序列固定，不随上下文增长"
        elif meas.get("local_window_bytes_per_sequence"):
            bases.add("dtype_fallback")
            comp["local_window"] = {
                "bytes_per_sequence": meas["local_window_bytes_per_sequence"],
                "bytes_basis": "dtype_fallback",
                "terms": {"note": "滑窗局部 KV，每序列固定，不随上下文增长。记录直接给的字节数，按模型 dtype 算"},
            }
        status = "measured"
        if meas.get("note"):
            notes.append(meas["note"])
        out = _kv_out(comp, status, atype, kv_dtype, layers, full_layers, linear_layers, notes)
        out["bytes_basis"] = next(iter(bases)) if len(bases) == 1 else "mixed"
        if "theoretical_floor" in bases:
            notes.append(_FLOOR_NOTE)
        if out["bytes_basis"] == "mixed":
            notes.append("各分量字节口径不一：有的按落盘精度（理论下限），有的按模型 dtype 兜底，合计不是同一种数")
        return out

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
            "elements_per_token": width * ilayers,
            "terms": {"formula": f"{wsrc} * bytes * {src_kind}({ilayers})"},
        }
        # 层数没声明产出层、或宽度用的是头数乘积，元素数都只是上界
        if src_kind != "index_kv_source_layers" or not ikv:
            comp["indexer"]["elements_upper_bound"] = True

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
    # residual 也是常驻：mHC 与 hyper-connection 的系数是每层的可学习权重，每个 token 都要
    # 走一遍，不是「只触及少量行」。漏掉它在 DeepSeek V4 一代只差 0.1%（39M / 37B），
    # 但在 Qwen3.8-Flash-Next 是 640,624,640 / 6.67B = 9.6%，会直接渲染到站点上
    always = sum(p.get(k, 0.0) for k in
                 ("attention", "compressor", "indexer", "dense_ffn", "shared_expert", "router", "norm",
                  "residual", "lm_head", "other"))
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


# ---------------------------------------------------------------- 生成类负载
#
# LLM 那套一切围绕「每 token」：自回归一步出一个 token，瓶颈在权重带宽。
# 生成类没有自回归 token，基本量是「单步 × 步数」：
#   token 数 = latent 体积 ÷ DiT patch，而 latent 体积由 VAE 的时空压缩率决定
#   单步注意力是 O(n²) 且 n 很大 —— 瓶颈是算力受限，不是 decode 那种带宽受限
#   端到端 = Σ各阶段(步数 × 每步前向次数 × 单步) + 文本编码一次 + VAE decode 一次
# 所以步数压缩（蒸馏、CFG 蒸馏、少步采样）是直接乘在总量上的。
#
# 口径纪律：VAE 压缩率与 DiT patch 分两个字段存，这里相乘；两者合成一个数存进记录
# 是范畴错误（CLAUDE.md 的口径一节）。每个乘式都把因子连同来源一起输出，
# 好让 Gate B 逐因子验 —— 只验一个因子的乘积不该被提交。

GEN_KINDS = ("image-gen", "video-gen")


def _patch3(gen: dict):
    """DiT 侧 patch 归一成 (t, h, w)。标量与二元写法补齐，image-gen 的时间维记 1。"""
    p = (gen.get("backbone") or {}).get("patch_size")
    if p is None:
        return None
    if isinstance(p, (int, float)):
        p = [p]
    p = [x for x in p]
    if len(p) == 1:
        return [1, p[0], p[0]]
    if len(p) == 2:
        return [1, p[0], p[1]]
    return p[:3]


def _default_bucket(gen: dict):
    """取标了 default 的那一档。没标就不猜 —— 猜错会让后面每个数都错。"""
    bs = gen.get("buckets") or []
    d = [b for b in bs if b.get("default")]
    if len(d) == 1:
        return d[0], None
    if not bs:
        return None, "generation.buckets 为空，没有可推导的分辨率/帧数档位"
    if not d:
        return None, f"generation.buckets 有 {len(bs)} 档但没有一档标 default，不替你选"
    return None, f"generation.buckets 里有 {len(d)} 档标了 default，只能有一档"


def gen_workload(gen: dict, kind: str, dit_params: float, dit_params_src: str,
                 layers: int, hidden: int) -> dict:
    """生成类的单步与端到端负载。

    报不出来就说为什么，不返回 0 —— 空结果和「真的没有」在代码里长得一样。
    """
    if not gen:
        return {"status": "unknown", "why": "记录里没有 generation 块"}

    bucket, why = _default_bucket(gen)
    if bucket is None:
        return {"status": "unknown", "why": why}

    vae = gen.get("vae") or {}
    s_c = vae.get("spatial_compression")
    t_c = vae.get("temporal_compression")
    patch = _patch3(gen)
    missing = [k for k, v in (("vae.spatial_compression", s_c),
                              ("backbone.patch_size", patch)) if not v]
    if kind == "video-gen" and not t_c:
        missing.append("vae.temporal_compression")
    if missing:
        return {"status": "unknown", "why": "缺少 " + "、".join(missing) + "，token 数算不出"}

    W, H = bucket.get("width"), bucket.get("height")
    F = bucket.get("frames") or 1
    if not W or not H:
        return {"status": "unknown", "why": "默认档位缺 width/height"}

    t_c = t_c or 1
    p_t, p_h, p_w = patch
    # 时间维不是简单的 F / t_c。**时间因果 VAE 的第一帧单独编码**，后面每 t_c 帧压成一帧：
    #   latent 帧数 = (F - 1) / t_c + 1
    # 这就是各家帧数都写成 4n+1 / 8n+1 的原因（Wan 的 121、HunyuanVideo 的 129、
    # LTX 的 8n+1）。按 F / t_c 算，Wan2.2 的 121 帧会得到 30.25 这种非整数，
    # 而真值是 31 —— 2% 的偏差不大，但它是个口径错误而不是精度误差，会一路传到 FLOPs。
    # **记录声明了 latent 帧数就用声明的，不套公式。**
    # (F-1)/t+1 只在「每 t 帧压一帧、首帧单独编」这一种分块规则下成立。实际各家不同：
    # MiniMax-H3 按 17 帧一块、先补 3 帧到 20 再除以 4，每块出 5 个 token、整段末尾丢 3 个，
    # T = ceil(F/17)×5 − 3 —— 默认档 243 帧真值 72，公式给 61.5，低 17%。
    # token 低 17% 经注意力的 O(n²) 放大成低 31%，而这几个数是模型页上最显眼的三个。
    # 各家的帧数约束（4n+1 / 8n+1 / 17n+5）都是分块规则的外在表现，
    # 单一除数表达不了任何一个精确值，所以留一条「实测值优先」的路。
    causal = vae.get("causal")
    declared_lf = bucket.get("latent_frames")
    if declared_lf:
        n_t_latent = float(declared_lf)
        t_formula = f"latent_frames = {declared_lf}（记录声明，未套公式）"
        t_source = "declared"
    elif causal:
        n_t_latent = (F - 1) / t_c + 1
        t_formula = "(F - 1) / t_vae + 1  （时间因果 VAE：首帧单独编码）"
        t_source = "formula_causal"
    else:
        n_t_latent = F / t_c
        t_formula = "F / t_vae"
        t_source = "formula_plain"
    n_t = n_t_latent / p_t
    n_h = H / (s_c * p_h)
    n_w = W / (s_c * p_w)
    latent_tokens = n_t * n_h * n_w

    # 逐因子：每一项都写清来自哪个字段，Gate B 照着一项项核
    factors = {
        "frames": F, "height": H, "width": W,
        "vae_temporal_compression": t_c, "vae_spatial_compression": s_c,
        "vae_causal": causal,
        "patch_t": p_t, "patch_h": p_h, "patch_w": p_w,
    }

    out = {
        "status": "ok",
        "bucket": bucket,
        "effective_compression": {
            "temporal": t_c * p_t,
            "spatial_h": s_c * p_h,
            "spatial_w": s_c * p_w,
            "note": "VAE 压缩率 × DiT patch。两者在记录里分开存，只在这里相乘 —— "
                    "合成一个数存进记录是范畴错误",
        },
        "latent_tokens": latent_tokens,
        "latent_tokens_terms": {
            "formula": f"[{t_formula}] / p_t × (H / s_vae / p_h) × (W / s_vae / p_w)",
            "factors": factors,
            "per_axis": {"t": n_t, "h": n_h, "w": n_w},
            "latent_frames": n_t_latent,
            # 声明值还是公式算的，要能一眼看出来 —— 否则「这个数可不可信」没法判
            "latent_frames_source": t_source,
            # 端到端有效时间压缩率。它**不等于** vae.temporal_compression：
            # 后者是块内卷积比（一手事实），前者是 F / latent_frames，受分块规则影响，
            # 而且各档位之间不是常数。两个数并排放着，省得有人拿其中一个去推另一个
            "effective_temporal_ratio": (F / n_t_latent) if n_t_latent else None,
        },
    }

    frac = [k for k, v in (("t", n_t), ("h", n_h), ("w", n_w)) if abs(v - round(v)) > 1e-9]
    if frac:
        hint = ("；vae.causal 未填 —— 时间因果 VAE 的 latent 帧数是 (F-1)/t_vae+1，"
                "填上 causal: true 多半就整除了" if "t" in frac and causal is None else "")
        out.setdefault("_frac_axes", frac)
        out["_frac_hint"] = (f"{'/'.join(frac)} 轴的 token 数不是整数，档位与压缩率对不齐"
                             + hint)

    # 音频 token：和视频 token 拼在同一条注意力序列里，所以要分两项记后相加
    audio = gen.get("audio") or {}
    audio_tokens = 0.0
    if audio.get("latent_hz") and bucket.get("duration_s"):
        audio_tokens = audio["latent_hz"] * bucket["duration_s"]
        out["audio_tokens"] = audio_tokens
        out["audio_tokens_terms"] = {
            "formula": "latent_hz × duration_s",
            "factors": {"latent_hz": audio["latent_hz"], "duration_s": bucket["duration_s"]},
            "note": "单声道一条序列。左右声道各跑一遍编解码器的结构里，"
                    "DiT 侧是否也翻倍取决于实现，未确认前不乘声道数",
        }

    notes = []
    tc = (gen.get("backbone") or {}).get("text_conditioning")
    n_txt_seq = bucket.get("text_tokens") or 0
    # **in_sequence 的文本 token 占的是同一条序列**，所以要进 seq，
    # 而注意力是 O(n²) —— 漏掉它整条注意力项偏低。
    # 此前 text_tokens 只进了线性项，序列长度那一半没接上，
    # 而且提示语照旧说「记录没给文本长度」—— 记录给了。
    seq = latent_tokens + audio_tokens + (n_txt_seq if tc == "in_sequence" else 0)
    lower_bound = tc == "in_sequence" and not n_txt_seq
    if lower_bound:
        notes.append("文本 token 拼在同一条序列里，但档位没给 text_tokens —— "
                     "序列长度是下界，注意力 FLOPs 同样是下界。填 buckets[].text_tokens 可以补上")
    out["sequence_tokens"] = seq
    out["sequence_tokens_is_lower_bound"] = lower_bound

    # 单步 FLOPs。线性与注意力分开 —— 两者随 n 的增长阶不同（n vs n²），
    # 合成一个数就看不出瓶颈在哪一侧
    flops = {}
    # 参数按「乘什么」分桶了就逐桶算。**AdaLN 调制是逐样本的** —— 它吃条件向量不是
    # token 流（H3 的 adaln_proj 入口宽 2688 而 hidden 5376，形状上就对不上），
    # 而它在新一代生成类里占三分之一上下。按 `2 × 参数量 × token 数` 一刀切，
    # hy15 的线性项高 3.05 倍、单步高 20.4%，注意力占比从真值 90.0% 被压成 74.8% ——
    # **误差专挑新模型放大**，恰好是最该看准的那几份。
    psc = (gen.get("backbone") or {}).get("param_scaling") or {}
    # n_txt 提到分支外：**单步与逐段必须用同一个文本 token 数**，
    # 而它原先定义在下面那个 if 里，逐段那段取不到 —— 于是逐段把文本项整项漏掉了（见下）
    n_txt = bucket.get("text_tokens") or 0
    if psc.get("per_video_token") or psc.get("per_text_token") or psc.get("per_sample"):
        p_vid = psc.get("per_video_token") or 0
        p_txt = psc.get("per_text_token") or 0
        p_smp = psc.get("per_sample") or 0
        # **这条注释原先写着「文本 token 数本仓目前没有字段装」** —— 写的时候是对的，
        # 后来 `buckets[].text_tokens` 加上了（上面二十多行就在读它），注释没跟上。
        # 留着的后果不是编译错，是让人**不去查**：它看起来已经回答了「为什么不算这一项」。
        # 现在的真实情况是：档位可以给 text_tokens，没给的那些才漏掉这一项并标明是下界 ——
        # 不替它猜一个长度。
        flops["linear"] = 2 * (p_vid * seq + p_txt * n_txt + p_smp)
        flops["linear_terms"] = {
            "formula": "2 × (每视频 token 参数 × token 数 + 每文本 token 参数 × 文本 token 数"
                       " + 每样本参数 × 1)",
            "factors": {"per_video_token": p_vid, "per_sample": p_smp,
                        "per_text_token": p_txt, "tokens": seq, "text_tokens": n_txt},
            "note": "AdaLN 这类吃条件向量的参数每次前向只过一遍，不随 token 数增长",
        }
        if p_txt and not n_txt:
            notes.append(f"per_text_token 声明了 {p_txt:,.0f} 但档位没给 text_tokens，"
                         "线性项漏掉了这一项 —— 是下界。填 buckets[].text_tokens 可以补上")
    elif dit_params:
        flops["linear"] = 2 * dit_params * seq
        flops["linear_terms"] = {
            "formula": "2 × 激活参数 × 序列 token 数",
            "factors": {"active_params": dit_params, "source": dit_params_src, "tokens": seq},
            "note": "**这是上界。** 公式隐含「每个参数都过每个 token」，而 AdaLN 调制"
                    "吃的是条件向量、每次前向只过一遍。填 backbone.param_scaling 可以算准",
        }
        if (gen.get("backbone") or {}).get("layers"):
            notes.append("线性 FLOPs 是上界：没填 backbone.param_scaling，"
                         "逐样本参数（AdaLN 这类）被当成逐 token 算了")
    else:
        notes.append("没拿到主干激活参数，单步线性 FLOPs 算不出")
    if layers and hidden:
        flops["attention"] = 4 * layers * hidden * seq * seq
        flops["attention_terms"] = {
            "formula": "每层 QK^T 2·n²·d + (softmax·V) 2·n²·d = 4·n²·d，再乘层数",
            "factors": {"layers": layers, "hidden": hidden, "tokens": seq},
            "note": "稠密全注意力口径。块稀疏（SSTA）或线性注意力要另按其形态折算，"
                    "本函数不替你折 —— 折算比例是记录里的事实，不是公式里的常数",
        }
    else:
        notes.append("缺 backbone.layers 或 backbone.hidden，单步注意力 FLOPs 算不出")
    if "linear" in flops and "attention" in flops:
        flops["total"] = flops["linear"] + flops["attention"]
        flops["attention_share"] = flops["attention"] / flops["total"]
    out["flops_per_step"] = flops

    # 端到端：多段流水线逐段相加。单段的用 sampling 那一组
    sampling = gen.get("sampling") or {}
    cfg = sampling.get("cfg") or {}
    stages = gen.get("pipeline_stages") or []
    runnable = [st for st in stages if st.get("steps") and st.get("forwards_per_step")]
    e2e = None
    if flops.get("total"):
        if runnable:
            # **逐段按自己的分辨率算。** 原来是「Σ步数 × 同一个单步 FLOPs」——
            # 对分辨率不变的流水线成立，对带超分的不成立：注意力是 O(n²)，
            # HunyuanVideo-1.5 的 1080p SR 段输出 1904×1072，token 是主干段的 2.2 倍、
            # 单步 4.2 倍，端到端因此差 18%。
            # 段里给了 out_width / out_height 就用它自己的 token 数，没给才退回主干段的。
            per_stage, est_only = [], []
            for st in runnable:
                ow, oh = st.get("out_width"), st.get("out_height")
                of = st.get("out_frames") or F
                if ow and oh:
                    s_nt = ((of - 1) / t_c + 1 if causal else of / t_c) / p_t
                    s_seq = s_nt * (oh / (s_c * p_h)) * (ow / (s_c * p_w)) + audio_tokens
                    # 线性项要和单步用同一套口径。这里原来写死了 2 × dit_params × seq，
                    # 于是 param_scaling 填上之后单步用新口径、端到端用旧口径 ——
                    # 同一个页面上两个数按两种公式算，而且谁也看不出来
                    # **上面那条注释写对了病根，却只治了一半。** 它说「原来写死
                    # 2 × dit_params × seq，于是单步用新口径、端到端用旧口径」——
                    # 改的时候补了 per_video_token 与 per_sample 两项，**漏了 per_text_token**，
                    # 于是同一份产物里顶层 linear 含文本项、逐段不含，差值恰好
                    # 2 × per_text_token × text_tokens（hunyuanvideo-1.5 每步 1.140e13，
                    # 114 步合计 1.30e15、占端到端 0.15%）。量级可忽略，**口径不可**。
                    # 文本 token 不随分辨率变，所以逐段与单步用同一个 n_txt。
                    if (psc.get("per_video_token") or psc.get("per_text_token")
                            or psc.get("per_sample")):
                        s_lin = 2 * ((psc.get("per_video_token") or 0) * s_seq
                                     + (psc.get("per_text_token") or 0) * n_txt
                                     + (psc.get("per_sample") or 0))
                    else:
                        s_lin = 2 * dit_params * s_seq
                    s_step = s_lin + 4 * layers * hidden * s_seq * s_seq
                else:
                    s_seq, s_step = seq, flops["total"]
                    est_only.append(st.get("name"))
                per_stage.append({
                    "name": st.get("name"), "steps": st["steps"],
                    "forwards_per_step": st["forwards_per_step"],
                    "tokens": s_seq, "flops_per_step": s_step,
                    "flops": st["steps"] * st["forwards_per_step"] * s_step,
                    "tokens_source": "stage" if (ow and oh) else "default_bucket",
                })
            e2e = sum(x["flops"] for x in per_stage)
            skipped = [st.get("name") for st in stages if st not in runnable]
            out["end_to_end_terms"] = {
                "formula": "Σ各阶段(步数 × 每步前向次数 × 该段单步 FLOPs)",
                "stages": per_stage,
                "stages_skipped": skipped,
            }
            if est_only:
                out["end_to_end_terms"]["note"] = (
                    "这几段没给 out_width / out_height，按主干段的 token 数估："
                    + "、".join(str(x) for x in est_only)
                    + "。分辨率不同的段（SR、上采样）实际 token 数不同，"
                    "而注意力是 O(n²)，所以这个数是粗估，不要当精确值")
                notes.append(
                    f"{len(est_only)} 个阶段缺 out_width / out_height，按主干段 token 数估算，"
                    "端到端是粗估")
            if skipped:
                notes.append(f"{len(skipped)} 个阶段缺步数或前向次数，未计入端到端："
                             + "、".join(str(x) for x in skipped))
        elif sampling.get("steps") and cfg.get("forwards_per_step"):
            e2e = sampling["steps"] * cfg["forwards_per_step"] * flops["total"]
            out["end_to_end_terms"] = {
                "formula": "步数 × 每步前向次数 × 单步 FLOPs",
                "factors": {"steps": sampling["steps"],
                            "forwards_per_step": cfg["forwards_per_step"],
                            "cfg_distilled": cfg.get("distilled"),
                            "cfg_embedded": cfg.get("embedded")},
            }
        else:
            notes.append("缺 sampling.steps 或 sampling.cfg.forwards_per_step，端到端算不出")
    if e2e is not None:
        out["flops_end_to_end"] = e2e
    out["excluded"] = [
        "文本编码一次（编码器参数量在 generation.text_encoders，本函数不计）",
        "VAE decode 一次（VAE 结构不在 config 里，解码器是 Transformer 时这一项不是小量）",
        "音频 VAE 与声码器",
    ]
    if out.pop("_frac_axes", None):
        notes.insert(0, out.pop("_frac_hint"))
    if notes:
        out["notes"] = notes
        out["status"] = "partial"
    return out


# ---------------------------------------------------------------- 交叉核对

def reconcile(cfg_t: dict, shp: dict, declared: dict, external: dict = None,
              kv: dict = None) -> list:
    """两条路都能算的量做核对。有差异就报，不替用户选一个。

    `external` 是 repoinfo.json 里 HF 自己从分片头统计的 safetensors tally。
    它是唯一一条不依赖本仓任何代码的参数量来源 —— 本仓的归类、打包还原和排
    scale 全都错过，而这个 tally 不会跟着错。scale 判据漏了一整代后缀那次
    （params 偏高 0.006%），Gate B 的 2% 容差吞掉了差异，只有它能接住。

    **这道闸锚在哪一侧、因此守不住什么**（skill 规则 11）：各仓的 tally 口径不统一，
    所以本仓侧要按几种口径各算一遍再比。口径之间的强弱差很大：

      逐位相同 / 含 scale  —— 最强。本仓的归类、打包还原、排 scale 全都被约束住
      容器数（±excludes） —— 弱。容器数是原始 numel，与本仓的 packing_factor 无关，
                             于是**打包因子乘几都能对上**：MiniMax-M3.1-Flash 的专家
                             按 2 倍（真值）、4 倍、8 倍还原，这一项都报 ok。
                             扣 excludes 的那一档同理，不约束 excludes 列对不对

    落到容器数那两档时，`why` 里会带上这句话，不要把「闸绿了」读成打包也查过了 ——
    那一侧的依据只能由记录自己给（M3.1-Flash 给了三条：hidden_size 6144、
    M3 未打包时的同名张量形状、scale 列数 × 16 = 逻辑列数）。
    """
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
            # 外部对账：逐位比，不给容差。这里不是「声明值取整了」的场合 ——
            # 两边都是从同一批分片头数出来的，对不上就是有一侧算错了
            tally = (external or {}).get("total")
            if tally:
                mine = shp["params_total"]
                scale = shp.get("scale_elements") or 0.0
                # HF 的 tally 在三个口径上各有两种可能，且各仓不一样：
                #   含不含 scale        ds-v3 不含、ds-v3.2-exp 含
                #   还不还原低位打包     V4 一代的 FP4-in-INT8 还原了、
                #                       MiniMax-M3.1-Flash 的 NVFP4-in-U8 没有
                #   统计哪些分片         通常是全部，但草稿侧若放在**自带索引的子目录**
                #                       （M3.1-Flash 的 dspark/）就不在 tally 里
                # 所以这里把本仓侧按这几种口径各算一遍，命中哪种就报哪种。
                # **这不是放宽判据**：每一种都是逐位比、不给容差，而且必须说得出
                # 它是哪种口径；全都对不上才是 MISMATCH。
                by_sub = shp.get("params_by_subsystem") or {}
                cont_sub = shp.get("params_container_by_subsystem") or {}
                container = shp.get("params_total_container")
                excl_names = declared.get("excludes") or []
                excl_cont = sum(cont_sub.get(k, 0.0) for k in excl_names)
                excl_txt = "+".join(excl_names)
                # 容器口径命中时，这道闸就不再约束打包因子了 —— 容器数是原始 numel，
                # 乘几都一样。必须随结论一起说出来，否则「闸绿了」会被当成打包也查过
                WEAK = ("。注意：这一档的对账不约束打包因子 —— 容器数是原始 numel，"
                        "本仓把它还原成几倍都能对上；打包因子的依据要在记录里自己给")
                # 口径 1（逐位相同）与口径 2（含 scale）保持原有措辞，
                # 这样既有记录重跑 derive 的产物逐字节不变，diff 里只会出现真正的变化
                cands = [(mine, None)]
                if scale:
                    cands.append((mine + scale,
                                  f"相同，但那边把 {scale:,.0f} 个 scale 元素算进了参数量；"
                                  "本仓排 scale，差值恰等于 scale 总数"))
                if container is not None and abs(container - mine) >= 1:
                    cands.append((container,
                                  "相同，但那是容器数 —— 那边没有还原低位打包，本仓还原了；"
                                  "两边的 scale 口径一致" + WEAK))
                    if excl_names:
                        cands.append((container - excl_cont,
                                      f"相同，但那是容器数、且不含 {excl_txt} —— "
                                      f"那边既没还原低位打包，也没统计 {excl_txt}"
                                      "（它在自带索引的子目录里，不在主索引的分片中）" + WEAK))
                hit = next((c for c in cands if abs(tally - c[0]) < 1), None)
                if hit:
                    why = (f"与 HF 的 safetensors tally 逐位相同（{tally:,.0f}）"
                           if hit[1] is None else
                           f"与 HF 的 safetensors tally（{tally:,.0f}）{hit[1]}")
                    out.append({"item": "外部参数量对账", "status": "ok", "why": why})
                else:
                    # scale 为 0 时不提「加 scale」那个备选 —— 它等于本仓参数量本身，
                    # 印出来是同一个数两遍。而 scale 恰好为 0 本身就是个线索：
                    # 量化发布的仓里一个 scale 都没认出来，多半是判据没命中
                    alt = (f"，也不等于本仓参数量加 scale（{mine + scale:,.0f}）"
                           if scale else "")
                    hint = ("本仓在这份清单里一个 scale 张量都没认出来 —— "
                            "若它是量化发布，先查 lib/dtypes.py 的 scale 判据"
                            if not scale else
                            "先查本仓的子系统归类、打包还原与 scale 判据")
                    out.append({"item": "外部参数量对账", "status": "MISMATCH",
                                "HF tally": tally, "本仓": mine,
                                "why": "HF 的 tally 由它自己从分片头统计、不依赖本仓代码，"
                                       f"所以对不上时问题在本仓侧：不等于本仓参数量{alt}。"
                                       + hint})
            elif external is not None:
                out.append({"item": "外部参数量对账", "status": "skip",
                            "why": "repoinfo.json 里没有 safetensors tally"})
        # KV 的层覆盖：公式路取 config 的 num_hidden_layers，而 MTP 块若自带注意力
        # 也会产出 KV。两路都锚在同一份 config 上，所以这件事不会表现为「不一致」——
        # 它只会表现为一个偏小的容量。不报 MISMATCH（按主干口径记账是合法选择），
        # 但每次 derive 都把两个数摆出来，让 Gate B 的人决定按哪个口径写
        mtp_attn = shp.get("mtp_attention_suffixes") or []
        blocks = shp.get("draft_blocks_seen") or 0
        # 混合结构只有 full attention 层累积 KV，「主干 N 层 + 草稿块」会把 N 读成 KV 层数、
        # 高估好几倍。草稿块匹配的是主干注意力的同名投影，所以它加的是 full 层
        full = kv.get("full_attention_layers") if kv and kv.get("linear_attention_layers") else None
        if mtp_attn and blocks and full:
            out.append({"item": "KV 层覆盖口径", "status": "ok",
                        "why": f"公式按主干 {full} 层 full attention 算（主干共 {kv['layers']} 层，"
                               f"其余是不累积 KV 的线性层）；清单里另有 {blocks} 个草稿块，"
                               f"带着与主干 full attention 同名的 {len(mtp_attn)} 个投影，"
                               f"即它们也产出 KV。开草稿侧时累积 KV 的层要按 "
                               f"{full + blocks} 层算 —— 记录里要写明用哪个口径"})
        elif mtp_attn and blocks and kv and kv.get("layers"):
            out.append({"item": "KV 层覆盖口径", "status": "ok",
                        "why": f"公式按主干 {kv['layers']} 层算（config "
                               f"num_hidden_layers）；清单里另有 {blocks} 个草稿块，"
                               f"带着与主干注意力同名的 {len(mtp_attn)} 个投影，"
                               f"即它们也产出 KV。开草稿侧时容量要按 "
                               f"{kv['layers'] + blocks} 层算 —— 记录里要写明用哪个口径"})
        # 缓存侧的外部对账。只有本仓能逐分量算出字节时才是真对账 —— 采用了公布值的
        # 情况下两边必然相等，那时报 ok 是假的，改成把两个数并排摆出来
        if kv and kv.get("declared_per_token_bytes"):
            if kv.get("declared_covers") != "per_token":
                # 没采用的情况：两个数各算各的，并排摆出来但不判对错 ——
                # 覆盖面不同的两个量本来就不该比，比了就是范畴错误
                out.append({"item": "外部 KV 字节对账", "status": "ok",
                            "why": f"厂商公布 {kv['declared_per_token_bytes']:,.0f} B/token，"
                                   f"covers={kv.get('declared_covers')!r} —— 覆盖面与本仓的"
                                   f"每 token 口径不同或来源未交代，**未采用**；"
                                   f"本仓逐分量算得 {kv.get('per_token_bytes', 0):,.0f} B 保持不变。"
                                   "两个数覆盖面不同，不比对、也不算倍数"})
            elif kv.get("bytes_basis") == "theoretical_floor":
                add("外部 KV 字节对账", kv["declared_per_token_bytes"],
                    kv.get("per_token_bytes"), "厂商公布", "本仓逐分量算", tol=0.005)
            else:
                want = kv["declared_per_token_bytes"]
                got = kv.get("computed_per_token_bytes") or 0
                out.append({"item": "外部 KV 字节对账", "status": "ok",
                            "why": f"采用厂商公布值 {want:,.0f} B/token；本仓逐分量算得 "
                                   f"{got:,.0f} B（{got / want - 1:+.0%}），差额在落盘格式"
                                   "未公布的那个分量上 —— 根因见记录的 open_questions"})
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


def _gate_b(cfg: dict, shapes: dict, shp: dict) -> dict:
    """Gate B 的两路还剩几路，以及缺的那一路为什么缺。

    判据只看「这一路的原料在不在」，不替记录判断结论可不可信 —— 后者是人的事。
    清单不完整（抽样快照）算半路：形状可用、参数量不可用，所以单独标出来。
    """
    has_cfg = bool(cfg)
    has_shapes = bool(shapes)
    incomplete = bool(shp.get("incomplete_note")) if has_shapes else False
    paths = (1 if has_cfg else 0) + (1 if (has_shapes and not incomplete) else 0)
    why = None
    if not has_cfg and not has_shapes:
        why = "config 与权重清单都没有，两路都走不了"
    elif not has_cfg:
        why = ("仓里没有任何 config（顶层与组件目录都没有）。"
               "结构字段只有清单一路，config 侧的交叉核对整条跳过")
    elif not has_shapes:
        why = ("仓里零 safetensors（权重是 .pt / .pth 这类 pickle），读头拿不到清单。"
               "参数量与权重存储只能取厂商声明，**不是本仓实测**")
    elif incomplete:
        why = shp.get("incomplete_note")
    return {"config": has_cfg, "manifest": has_shapes and not incomplete,
            "manifest_incomplete": incomplete, "paths": paths,
            **({"why": why} if why else {})}


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
    rip = os.path.join(fd, "repoinfo.json")
    repoinfo = json.load(open(rip)) if os.path.exists(rip) else None
    if not cfg and not shapes:
        sys.exit(f"! {fd} 下既没有 cfg.json 也没有 shapes.json，先跑 fetch_hf.py")

    # --help 一直写着「缺省按 id 在 data/models 下找」，而代码是
    # `load_yaml(args.model) if args.model else {}` —— **这个查找从来没实现过**。
    # 后果不是报错：model={} → kind 为 None → 整条走 LLM 路，
    # 生成类记录的 generation_workload **整块消失**，而五道闸全过
    # （gatea-video 的一个提交就是这么错的）。
    # 「空结果和真的没有长得一样」的又一个实例，这次让帮助文本说的那件事真的发生。
    model_path = args.model
    if not model_path:
        import glob as _glob
        # 记录在 data/models 下，而 --out-dir 默认是 data/derived —— 取它的上一级，
        # 这样 --out-dir 指到别处时查找也跟着走，不写死路径
        _root = os.path.dirname(os.path.normpath(args.out_dir)) or "data"
        hits = sorted(_glob.glob(os.path.join(_root, "models", "*", f"{args.id}.yaml")))
        if len(hits) == 1:
            model_path = hits[0]
            print(f"  （未给 --model，按 id 找到 {model_path}）", file=sys.stderr)
        elif len(hits) > 1:
            sys.exit(f"! 按 id 找到多份记录，用 --model 点名：{hits}")
    model = load_yaml(model_path) if model_path else {}
    if not model:
        print(f"! 没有记录可读（--model 未给且 data/models 下没有 {args.id}.yaml）。"
              "生成类记录会因此整块跳过 generation_workload", file=sys.stderr)
    kind = model.get("kind")
    is_gen = kind in GEN_KINDS
    gen = model.get("generation") or {}
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

    # 生成类走另一条口径：非自回归，推理过程中没有跨步复用的 KV，也没有
    # 「每 token 激活」和 EP all-to-all。**跳过要留显式标记，不能返回 0** ——
    # 空结果和「真的没有」在代码里长得一样（model-audit 规则 7），
    # 站点看到 0 会把「不适用」渲染成「这个模型不占缓存」
    _NA = "非自回归生成模型：推理过程中无跨步复用的 KV，不适用本口径"
    if is_gen:
        kv = {"status": "not_applicable", "why": _NA}
        act = {"status": "not_applicable", "why": "生成类每步对全部 latent token 做一次前向，"
                                                  "没有「每 token 激活」这个量；单步负载见 generation_workload"}
        comm = {"status": "not_applicable", "why": "生成类不做 EP all-to-all；"
                                                   "对应的通信是序列并行(Ulysses)的 all-to-all，口径不同，本脚本未建模"}
    else:
        kv = kv_memory(t, attn, kv_dtype, prec.get("kv_cache_scale"), prec.get("kv_cache_components"))
        if kv_dtype_assumed and kv.get("status") in KV_USABLE:
            kv["dtype_assumed"] = True
            kv["dtype_assumed_note"] = (
                f"precision.kv_cache 未填，按模型 dtype {kv_dtype} 兜底。"
                "KV 精度直接决定容量结论，确认后请填进模型 YAML")
        if kv.get("status") in KV_USABLE and not kv.get("bytes_basis"):
            # 公式路径：字节按单一 KV dtype 算，口径取决于这个 dtype 是填的还是兜底的
            kv["bytes_basis"] = "dtype_fallback" if kv_dtype_assumed else "declared_dtype"

        # 厂商公布的每 token 全局 KV 字节：缓存侧唯一的外部锚。参数量有 HF 的 safetensors
        # tally 可对，缓存侧此前什么都没有 —— 元素数与落盘格式两侧都由本仓决定，错了只会
        # 得到一个自洽的错数（V4 一代按 BF16 兜底高了 1.7 倍，直到读厂商的跨代图才发现）。
        # 公布值不依赖本仓任何代码，所以它一旦存在就当权威：本仓每个分量都有公布格式时
        # （basis 已是 theoretical_floor）拿它对账，否则直接采用它，别让兜底值盖过厂商。
        decl = (attn.get("kv_sharing") or {}).get("declared_per_token_bytes") or {}
        if decl.get("value") and kv.get("status") in KV_USABLE:
            want = float(decl["value"])
            kv["declared_per_token_bytes"] = want
            kv["declared_source"] = decl.get("source")
            kv["declared_tier"] = decl.get("tier")
            kv["declared_covers"] = decl.get("covers")
            # covers 决定它能不能进 per_token_bytes 这个槽。本仓 47 份里有 17 份带
            # 「每序列固定」分量（线性注意力的递归状态、滑窗缓冲区），两个分量口径不同、
            # 不能相加。公布值若把它们算在一起、或来源根本没交代，塞进 per_token_bytes
            # 就是把一个更大的量当成每 token 量，再乘上下文长度会一路放大。
            # 这类只登记不采用 —— 字段留着，口径不动。
            if decl.get("covers") != "per_token":
                kv.setdefault("notes", []).append(
                    f"厂商公布 {want:,.0f} B/token，但 covers={decl.get('covers')!r} —— "
                    "不是「只含随上下文线性增长的那一半」，不采用进 per_token_bytes，"
                    "本仓逐分量算出的值与 bytes_basis 保持不变。只登记备查")
            elif kv.get("bytes_basis") != "theoretical_floor":
                kv["computed_per_token_bytes"] = kv.get("per_token_bytes")
                kv["per_token_bytes"] = want
                kv["bytes_basis"] = "vendor_declared"
                got = kv["computed_per_token_bytes"] or 0
                kv.setdefault("notes", []).append(
                    f"每 token 字节采用厂商公布值 {want:,.0f} B。本仓按分量算得 {got:,.0f} B"
                    + (f"（{got / want - 1:+.0%}）" if want else "")
                    + "，差额落在落盘格式未公布、只能按模型 dtype 兜底的那个分量上；"
                    "元素数不受影响，它实测自权重清单")
        act = active_params(shp, t, attn, moe)
        comm = comm_per_token(t, moe, hidden, act_dtype, args.ep)

    # decode 的线性 FLOPs 与算术强度。注意力部分随上下文变化，单列不混进来。
    flops = {}
    if act.get("status") == "ok":
        flops["linear_per_token"] = 2 * act["total"]
        flops["note"] = "仅线性矩阵乘；注意力、状态更新、路由与通信不在内"
    intensity = {}
    if act.get("status") == "ok" and kv.get("status") in KV_USABLE and shp.get("available"):
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
    if kv.get("status") in KV_USABLE:
        if args.context_lens:
            lens = [int(x) for x in args.context_lens.split(",") if x.strip()]
        else:
            # 默认带上模型自身的上下文上限 —— 容量结论最该看的就是满上下文那一档
            cap = (model.get("context") or {}).get("max") or t.get("max_position_embeddings")
            lens = sorted({x for x in [4096, 32768, cap] if x and x <= (cap or x)})
        for L in lens:
            kv_at[str(L)] = kv["per_sequence_bytes"] + kv["per_token_bytes"] * L

    # 生成类的单步与端到端负载
    gw = None
    if is_gen:
        bb = gen.get("backbone") or {}
        p_model = model.get("params") or {}
        brk = p_model.get("breakdown") or {}
        # 主干激活参数，按可信度取：记录里声明的 > 权重按组件分出来的 dit > 总量。
        # 口径不同的三个值，取了哪个要跟着报出去 —— 不然下游不知道这个乘式的因子是哪来的
        if p_model.get("active_decode"):
            dit_params, dit_src = p_model["active_decode"], "params.active_decode（记录声明）"
        elif brk.get("dit"):
            dit_params, dit_src = brk["dit"], "params.breakdown.dit（记录声明）"
        elif (shp.get("params_by_component") or {}).get("dit"):
            dit_params, dit_src = shp["params_by_component"]["dit"], "权重按组件分出的 dit"
        elif shp.get("complete"):
            dit_params, dit_src = shp["params_total"], "权重总量（未按组件分，含文本编码器/VAE 时会高估）"
        else:
            dit_params, dit_src = None, None
        gw = gen_workload(gen, kind, dit_params, dit_src,
                          bb.get("layers"), bb.get("hidden"))

    declared = {"params_total": (model.get("params") or {}).get("total"),
                "excludes": (model.get("params") or {}).get("excludes")}
    rec = reconcile(t, shp, declared, external=(repoinfo.get("safetensors") or {})
                    if repoinfo else None, kv=kv)

    out = {
        "id": args.id,
        "generated_by": "scripts/derive.py",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "inputs": {
            "facts_dir": fd,
            "config_present": bool(cfg),
            "shapes_present": bool(shapes),
            "model_yaml": model_path,
            "ep_size": args.ep,
        },
        # Gate B 还剩几路。**这个事实一直在 inputs 里（config_present / shapes_present），
        # 但埋着、没人渲染，于是「这份只对过一路」只能靠人在 reviewed.method 的散文里写** ——
        # 写不写全看记录的人记不记得。现在由 derive 直接报。
        #   两路   config 与权重清单互相对账，本仓的常态
        #   仅 config  仓里零 safetensors（权重是 .pt / .pth），读头拿不到清单
        #              —— hunyuanvideo-13b、sana-video-2.0-5b
        #   仅清单    仓里没有任何 config —— ltx-2.3
        # **「仅一路」不等于不能录，等于这份记录的某一类断言没有第二个来源可对。**
        # 它必须在页面上看得见，否则读者会以为所有记录的可信度是一样的。
        "gate_b": _gate_b(cfg, shapes, shp),
        "from_shapes": shp,
        "kv_cache": kv,
        "kv_capacity_bytes_at_context": kv_at,
        # 容量数跟着字节口径走：theoretical_floor 是下限，不是部署占用
        "kv_capacity_basis": kv.get("bytes_basis"),
        "active_params": act,
        "flops": flops,
        "arithmetic_intensity": intensity,
        "ep_communication": comm,
        "reconciliation": rec,
        **({"generation_workload": gw} if gw is not None else {}),
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
        if shp.get("weight_bytes_by_component"):
            comp = sorted(shp["weight_bytes_by_component"].items(), key=lambda kv_: -kv_[1])
            print("  按组件         " + ", ".join(
                f"{subsystems.COMPONENT_LABEL.get(k, k)} {dtypes.fmt_bytes(v)}" for k, v in comp))
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
    if kv.get("status") in KV_USABLE:
        # 口径直接印 bytes_basis 的原值，不在这里维护一张中文标签表。
        # 原先是三个 if 各挂一个中文标签（精度兜底 / 厂商公布 / 理论下限），
        # 于是 declared_dtype 与 mixed 跑 CLI 时**一个口径标记都不带** ——
        # 读的人会把它当成「这个数没有口径问题」。补成五个标签能修，
        # 但那样仓里就有第四份 basis 标签表（另三份在 data.ts、[id].astro、
        # docs/kv-cache-accounting.md），正是 validate 第 5 道闸在数的东西。
        # CLI 是开发者工具，印原值反而更有用：和字段值、和 grep 的对象一致。
        basis = kv.get("bytes_basis")
        warn = f" [{basis}]" if basis else ""
        warn += " [含估算项]" if kv.get("status") == "partial" else ""
        fa = kv.get("full_attention_layers"); la = kv.get("linear_attention_layers")
        print(f"注意力显存       {kv['per_token_bytes']:.0f} B/token"
              f" + {dtypes.fmt_bytes(kv['per_sequence_bytes'])}/序列"
              f"  ({kv['attention_type']}, full {fa} 层 / 线性 {la} 层){warn}")
        for name, c in (kv.get("components") or {}).items():
            bit = (f"{c['bytes_per_token']:.0f} B/token" if c.get("bytes_per_token")
                   else f"{dtypes.fmt_bytes(c.get('bytes_per_sequence', 0))}/序列")
            print(f"    {name:16s} {str(c.get('layers', '—')):>3s} 层  {bit}")
        for n in kv.get("notes") or []:
            print(f"    ! {n}")
        for L, v in kv_at.items():
            print(f"    @{int(L):>8d} tok  {dtypes.fmt_bytes(v)}")
    elif kv.get("status") == "not_applicable":
        print(f"KV / token       不适用 — {kv.get('why')}")
    else:
        print(f"KV / token       未知 — {kv.get('why')}")
    if gw:
        if gw.get("status") == "unknown":
            print(f"生成负载         未知 — {gw.get('why')}")
        else:
            b = gw["bucket"]
            ec = gw["effective_compression"]
            dims = f"{b.get('width')}×{b.get('height')}"
            if b.get("frames"):
                dims += f"×{b['frames']}f"
            bound = " (下界)" if gw.get("sequence_tokens_is_lower_bound") else ""
            print(f"生成负载         {dims}  有效压缩 "
                  f"t{ec['temporal']:g}×h{ec['spatial_h']:g}×w{ec['spatial_w']:g}")
            print(f"    latent token  {gw['latent_tokens']:,.0f}"
                  + (f"  + 音频 {gw['audio_tokens']:,.0f}" if gw.get("audio_tokens") else "")
                  + f"  = 序列 {gw['sequence_tokens']:,.0f}{bound}")
            f = gw.get("flops_per_step") or {}
            if f.get("total"):
                print(f"    单步 FLOPs    {f['total']:.3e}"
                      f"  (线性 {f['linear']:.3e} / 注意力 {f['attention']:.3e}，"
                      f"注意力占 {f['attention_share']:.0%})")
            if gw.get("flops_end_to_end"):
                print(f"    端到端 FLOPs  {gw['flops_end_to_end']:.3e}"
                      "   不含文本编码与 VAE decode")
            for n in gw.get("notes") or []:
                print(f"    ! {n}")
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
