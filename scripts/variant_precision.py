#!/usr/bin/env python3
"""把厂商另发的官方量化仓逐张量读成一份 `official_variants` 条目。

**为什么要有这个脚本，而不是手抄 HF 的 tally。**
HF 的 `?expand[]=safetensors` 对 GPTQ 仓给的那三档数**既不是容器数也不是参数数**，
是两种口径拌在一起的：`Qwen/Qwen3.5-27B-GPTQ-Int4` 报 `I32 17,112,760,320`，而
清单里 I32 的真实 numel 是 2,157,576,192 —— 那个 171 亿是 `qweight` 还原打包后的
**逻辑参数数**（2,139,095,040 × 8），挂在容器 dtype 的名下，同时把 `qzeros`
（1,671 万个 I32 零点）折掉、把 `scales`（1.34 亿个 F16）整类丢掉。
所以它既不能当 dtype 直方图，也不能当参数量 —— 两种读法都解释不通，
本仓的「numel 是容器数不是参数数」那条坑在这里有第三种变体。唯一的办法是逐张量读。

口径与 derive.py 的 from_shapes 一致，三个互不混用：
  参数量  排除 scale / 元数据；`qweight` 按 bits 还原打包（8 个 INT4 装进一个 I32）
  字节数  包含 scale 与元数据；不还原打包（容器就是真实存储）
  元素数  原始 numel，只用于核对

**对账是硬闸**：量化仓与主仓是同一份权重的两种存法，逐子系统参数量必须逐位相同。
不同就非零退出 —— 那说明归类或打包因子错了，而错了之后页面上看不出来。
对账的另一侧是 `data/derived/<id>.json`（提交过的），不是 facts/（在 gitignore 里）。

用法:
  .venv/bin/python scripts/variant_precision.py qwen3.5-27b \
      --variant-facts qwen3.5-27b-gptq-int4 --dtype INT4
"""
import argparse
import collections
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import dtypes, subsystems  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 占比低于这个数就记 MIXED，和 fetch_dtypes.py 同一条规则 —— 两处用不同的阈值，
# 同一个子系统在主仓和量化仓会得到不同写法的结论，而那是脚本之差不是模型之差。
MIXED_BELOW = 0.60


def numel(shape):
    n = 1
    for x in shape or []:
        n *= x
    return n


def logical_bits(cfg):
    """量化仓声明的权重位宽。GPTQ 写 bits，compressed-tensors 写 num_bits，
    modelopt 只写 quant_algo（NVFP4 / FP8）。都没有就返回 None，交给签名去判。"""
    qc = cfg.get("quantization_config") or {}
    for k in ("bits", "num_bits", "weight_bits"):
        if qc.get(k):
            return int(qc[k])
    algo = str(qc.get("quant_algo") or qc.get("quant_method") or "").upper()
    if "FP4" in algo or "INT4" in algo or "W4" in algo:
        return 4
    if "FP8" in algo or "INT8" in algo or "W8" in algo:
        return 8
    return None


# scale 的 dtype → (逻辑精度, 每组多少个元素)。**这是「认签名不认名字」那条规矩的落地**：
# NVFP4 与 MXFP4 的元素都是 E2M1，分开它们的是块大小与 scale 格式 ——
# 块 16 + E4M3 是 NVFP4 式，块 32 + E8M0 是 OCP MXFP4。
# 只在 config 什么都没说时用（LTX-2.3-nvfp4 整个仓没有 config），而且推出来的结论
# **必须过下面那道形状一致性检查**才算数，不然宁可报错也不猜。
_SIG = {
    "FP8_E4M3": ("NVFP4", 16),
    "FP8_E8M0": ("MXFP4", 32),
}
# GPTQ 式：scale 是 F16/BF16，块大小由 config 的 group_size 给，这里只给位宽
_GPTQ_SCALE = {"FP16", "BF16"}

# 8 bit 那一档的块格式名：元素是 E4M3，每 32 个配一个 E8M0 scale 就是 OCP MXFP8。
# 和 4 bit 那边同一条规矩 —— 元素格式定不了名字，块大小与 scale 格式才定得了。
_MX8_SCALE = {"FP8_E8M0", "UINT8"}
_MX8_GROUP = 32

_WEIGHT_LEAVES = {"weight", "qweight", "weight_packed"}
_SCALE_SIBLINGS = ("scales", "weight_scale", "weight_scale_inv", "scale")


def scale_sibling(stem, shape_of):
    for suf in _SCALE_SIBLINGS:
        k = f"{stem}.{suf}"
        if k in shape_of:
            return k
    return None


def _mx_name(name, dt, shape_of, dtype_of):
    """没打包的 F8_E4M3 权重：按 scale 签名看看它是不是 MXFP8。"""
    if dt != "FP8_E4M3" or name.lower().rsplit(".", 1)[-1] != "weight":
        return dt
    sk = scale_sibling(name.rsplit(".", 1)[0], shape_of)
    if not sk or dtypes.norm(dtype_of[sk]) not in _MX8_SCALE:
        return dt
    w, sc = shape_of.get(name) or [], shape_of.get(sk) or []
    if len(w) == len(sc) and sc and sc[-1] and w[-1] == sc[-1] * _MX8_GROUP \
            and all(a == b for a, b in zip(w[:-1], sc[:-1])):
        return "MXFP8"
    return dt


def unpack(name, meta, shape_of, dtype_of, bits, group_hint):
    """这个权重张量的 (逻辑精度, 打包因子)，以及用来复核的 (轴, 组大小)。

    **判据是结构，不是名字。** 三家格式的张量名完全不同 —— GPTQ 叫 qweight 配 scales、
    modelopt 叫 weight 配 weight_scale、compressed-tensors 叫 weight_packed 配
    weight_scale —— 但它们是同一件事：一个整型容器装着若干个低位权重，旁边一个
    per-group 的 scale。所以认的是「整型容器 + 有 scale 兄弟」这个形状。

    位宽从三处依次取：config 的 bits / num_bits、config 的 quant_algo、scale 的签名。
    取到之后**必须过一致性检查**：容器那一维还原出来的逻辑长度，要等于 scale 的组数
    乘以组大小。不等就抛错 —— 猜错一个打包因子就是整数倍的参数量误差，
    而那种错在页面上看不出来（本仓的 kimi-k2-thinking 少算过 8 倍）。
    """
    dt = dtypes.norm(meta.get("dtype", ""))
    leaf = name.lower().rsplit(".", 1)[-1]
    if leaf not in _WEIGHT_LEAVES or not dtypes.is_container(dt):
        # 没打包，但**名字仍可能不是元素格式**：MXFP8 的元素就是 E4M3，
        # 分开它们的同样是签名（每 32 个元素一个 E8M0 scale）。
        # 不认这一条，MiniMax-M3-MXFP8 量出来会是 FP8_E4M3，
        # 而 weights_release 声明的是 MXFP8 —— 两处对不上，闸口照报，
        # 而真相是两边说的是同一件事的两个面。「认签名不认名字」在这里也要成立
        return _mx_name(name, dt, shape_of, dtype_of), 1, None
    stem = name.rsplit(".", 1)[0]
    sk = scale_sibling(stem, shape_of)
    if not sk:
        # 整型容器但没有 scale 兄弟：它就是一个真的整型张量，不是打包
        return dtypes.norm(dt), 1, None
    sdt = dtypes.norm(dtype_of[sk])
    group = group_hint
    if bits:
        lg = {4: "INT4", 8: "INT8"}.get(bits, f"INT{bits}")
        if sdt in _SIG and bits == 4:
            lg, group = _SIG[sdt][0], (group or _SIG[sdt][1])
    elif sdt in _SIG:
        lg, group = _SIG[sdt]
    elif sdt in _GPTQ_SCALE:
        lg = "INT4"
    else:
        raise SystemExit(f"! {name}：容器 {dt} 带 scale {sdt}，但位宽既没在 config 里"
                         f"声明、也配不上任何已知签名。**不要猜** —— 补 config 的解析"
                         f"或把这个签名加进 _SIG，别让参数量按 1 倍算过去")
    f = dtypes.packing_factor(lg, dt)
    if f <= 1:
        raise SystemExit(f"! {name}：判成逻辑精度 {lg} 装在 {dt} 里，而打包因子算出来是 "
                         f"{f} —— lib/dtypes.py 的 BITS / _CONTAINERS 少了一档")
    g, group = _check_shapes(name, meta["shape"], shape_of[sk], f, group, lg, dt, sk)
    return lg, f, (g, group)


# scale 在非分组轴上会被**补齐到 128 的整数倍**（modelopt 的 blockscale swizzle）。
# LTX-2.3 的 `to_gate_logits` 就是这样：权重 [32, 4096]，scale 却是 [128, 256] ——
# 256 = 4096/16 是真的组数，而 128 是 32 补上去的。不认这件事就会把它判成「组大小 4」，
# 而同一个仓里别的张量是 16，于是要么报假错、要么把块写错。
_SCALE_PAD = 128


def _check_shapes(name, w, s, f, group, lg, dt, sk):
    """打包因子与 scale 形状对不对得上。返回 (分组轴, 组大小)。

    **三家格式把「打包轴」和「分组轴」放在不同的地方**，所以只能逐轴试：
      GPTQ      qweight [in/8, out] + scales [in/group, out]  —— 两者都在 0 轴
      AWQ       qweight [in, out/8] + scales [in/group, out]  —— 打包在 1 轴、分组在 0 轴
      modelopt  weight  [out, in/2] + weight_scale [out', in/group] —— 都在末轴，
                而 out' 是 out 补齐到 128 的倍数
    非分组轴要么与权重同长，要么是补齐上去的；分组轴上「还原后的长度 ÷ 组数」就是组大小。
    对不上就抛错 —— 打包因子判错是整数倍的参数量误差，而那种错在页面上看不出来。
    """
    if len(w) != len(s):
        raise SystemExit(f"! {name}：权重 {w} 与 scale {sk} {s} 维数都不同，判不了打包")

    def pad_ok(i):
        return s[i] == w[i] or (s[i] > w[i] and s[i] % _SCALE_PAD == 0)

    for g in range(len(w)):
        if not all(pad_ok(j) for j in range(len(w)) if j != g):
            continue
        if not s[g]:
            continue
        # 分组轴上可能带着打包（GPTQ / modelopt），也可能不带（AWQ）
        for logical in (w[g] * f, w[g]):
            if logical % s[g]:
                continue
            cand = logical // s[g]
            if group and cand != group:
                continue
            return g, (group or cand)
    raise SystemExit(
        f"! {name}：打包因子 {f}（{lg} 装 {dt}）与 scale 形状对不上 —— "
        f"权重 {w}、scale {sk} {s}、期望组大小 {group or '未知'}。"
        f"逐轴都试过了（含补齐到 {_SCALE_PAD} 的倍数），没有一根说得通。"
        f"两头至少有一个判错了，不要落库")


def measure(shapes, cfg, mtp_from, only_component=None):
    bits = logical_bits(cfg)
    params = collections.Counter()          # sub -> 逻辑参数数
    nbytes = collections.Counter()          # sub -> 落盘字节
    hist = collections.defaultdict(collections.Counter)   # sub -> 逻辑 dtype -> 参数数
    tensors = collections.defaultdict(collections.Counter)  # sub -> 逻辑 dtype -> 张量数
    containers = collections.defaultdict(set)             # 逻辑 dtype -> 容器 dtype
    scale_dt = collections.defaultdict(collections.Counter)  # sub -> scale dtype -> 元素数
    # 块的两个因子分开量：group 式（scales 沿 in 分组）与 tile 式（scale_inv 二维）
    blocks = collections.defaultdict(collections.Counter)  # sub -> 块字符串 -> 张量数
    # sub -> group/channel/tensor/block -> 张量数。逐通道与逐块是两种粒度，
    # 不是「块等于整行」—— 混为一谈会让 `+` 的约定报出一个不存在的「多方案」
    strategy = collections.defaultdict(collections.Counter)
    shape_of = {}

    dtype_of = {}
    for name, meta in shapes.items():
        shape_of[name] = meta.get("shape") or []
        dtype_of[name] = meta.get("dtype", "")
    group_hint = (cfg.get("quantization_config") or {}).get("group_size")
    for name, meta in shapes.items():
        n = numel(meta.get("shape"))
        dt = dtypes.norm(meta.get("dtype", ""))
        comp, sub = subsystems.classify2(name, mtp_from)
        # 生成类的量化仓常常是**整包**（DiT + VAE + 文本编码器 + 声码器），而本仓那份
        # 记录的清单可能只收到 DiT 那一个组件（facts/ltx-2 取的是 transformer/*）。
        # 两边口径不同，整包对账必然红 —— 但那不是算错，是量的不是同一个东西。
        # 所以给出组件就只量那一个组件，并把它写进记录，让「这一条覆盖什么」是显式的。
        if only_component and comp != only_component:
            continue
        nbytes[sub] += n * dtypes.nbytes(dt)
        if dtypes.is_scale_tensor(name, dt):
            scale_dt[sub][dt] += n
            continue
        if dtypes.is_metadata_tensor(name, dt):
            continue
        lg, f, axgrp = unpack(name, meta, shape_of, dtype_of, bits, group_hint)
        if f > 1:
            containers[lg].add(dt)
        params[sub] += n * f
        hist[sub][lg] += n * f
        tensors[sub][lg] += 1
        b = block_of(name, shape_of, axgrp)
        if b:
            kind, blk = b
            strategy[sub][kind] += 1
            if blk:
                blocks[sub][blk] += 1
    return {"bits": bits, "params": params, "bytes": nbytes, "hist": hist,
            "tensors": tensors, "containers": containers, "scale_dt": scale_dt,
            "blocks": blocks, "strategy": strategy}


def block_of(name, shape_of, axgrp):
    """这个权重张量的实测量化块，写成 `[沿 out, 沿 in]`。

    两种形状，所以两条路：
      打包的（axgrp 不为 None）  沿单轴分组，一个 scale 覆盖 group 个元素、
                                 另一维一个通道 → `[1, N]`。组大小与轴在 unpack()
                                 做一致性检查时就定下来了，这里不再独立猜一遍 ——
                                 两处各猜一次，迟早会在某个仓上给出两个不同的答案
      没打包的（FP8 的 tile）    `weight [out, in]` 配 `weight_scale_inv
                                 [ceil(out/T), ceil(in/T)]`，两维都分块 → `[T, T]`
    **不抄 config 的 weight_block_size**（model-audit 规则 2：config 描述模块配置，
    不描述落盘形状；V4 一代声明 [128,128] 而 FP4 专家实际是 [1,32]）。
    """
    if axgrp:
        _ax, group = axgrp
        return ("group", f"[1,{group}]") if group else None
    if name.lower().rsplit(".", 1)[-1] != "weight":
        return None
    stem = name.rsplit(".", 1)[0]
    w = shape_of.get(name) or []
    for suffix in ("weight_scale_inv", "weight_scale", "scale"):
        sc = shape_of.get(f"{stem}.{suffix}")
        if not sc:
            continue
        # **逐通道不是「块 = 整行」。** compressed-tensors 的 strategy: channel 给每个
        # 输出通道一个 scale（scale 形状 [out, 1]），根本没有块这个概念 ——
        # 智谱五代官方 FP8 仓全是这一套（block_structure: null、group_size: null）。
        # 硬写成 [1, in] 在本仓的记法里是「块大小等于输入维」，而不同投影的输入维不同，
        # 于是一个桶里会出现 `[1,12288]+[1,5120]`，被 `+` 的约定读成
        # 「这个子系统里不止一种块方案」—— 那是个错结论：方案只有一种，叫逐通道。
        if len(w) == 2 and len(sc) == 2 and sc[0] == w[0] and sc[1] == 1:
            return ("channel", None)
        if not sc or all(x == 1 for x in sc) or len(sc) == 0:
            return ("tensor", None)
        if len(w) == 2 and len(sc) == 2 and sc[0] and sc[1]:
            # 向上取整的那一维按本仓记法写成 `Nceil`
            return ("block", f"[{_tile(w[0], sc[0])},{_tile(w[1], sc[1])}]")
        return None
    return None


def _tile(dim, groups):
    q, r = divmod(dim, groups)
    return f"{q}" if r == 0 else f"{-(-dim // groups)}ceil"


def dominant(counter):
    """这一桶报哪个 dtype，以及它的占比。占比不足 MIXED_BELOW 就报 MIXED。"""
    tot = sum(counter.values())
    if not tot:
        return None, 0.0
    dt, n = counter.most_common(1)[0]
    share = n / tot
    return (dt if share >= MIXED_BELOW else "MIXED"), share


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id", help="本仓记录 id，如 qwen3.5-27b")
    ap.add_argument("--variant-facts", required=True,
                    help="量化仓的 facts 子目录名，如 qwen3.5-27b-gptq-int4")
    ap.add_argument("--dtype", help="这一档的声明精度（weights_release 里的那个键）")
    ap.add_argument("--origin", choices=["native", "post_quantized"],
                    help="原生发布还是事后再量化。不给就只打印判据，不写进 YAML 片段")
    ap.add_argument("--component",
                    help="只量这一个组件（data/schema/components.json 的 key）。"
                         "量化仓是整包、而本记录的清单只收了其中一个组件时用")
    ap.add_argument("--file", help="变体在仓内的文件名。同仓不同文件的发布形态要填")
    ap.add_argument("--facts-dir", default=os.path.join(ROOT, "facts"))
    ap.add_argument("--derived-dir", default=os.path.join(ROOT, "data", "derived"))
    args = ap.parse_args()

    vdir = os.path.join(args.facts_dir, args.variant_facts)
    shapes = json.load(open(os.path.join(vdir, "shapes.json")))
    cfg = json.load(open(os.path.join(vdir, "cfg.json")))
    fetched = json.load(open(os.path.join(vdir, "fetched.json")))
    tcfg = cfg.get("text_config") or cfg
    mtp_from = tcfg.get("num_hidden_layers")

    m = measure(shapes, cfg, mtp_from, args.component)
    params, nbytes, hist = m["params"], m["bytes"], m["hist"]

    # ── 硬闸：与主仓逐子系统对账 ──────────────────────────────
    dp = os.path.join(args.derived_dir, f"{args.model_id}.json")
    fs = ((json.load(open(dp)) or {}).get("from_shapes") or {}) if os.path.exists(dp) else {}
    main_sub = {k: float(v) for k, v in (fs.get("params_by_subsystem") or {}).items()}
    bad = []
    print(f"=== 对账：{args.variant_facts} ↔ 主仓 derived/{args.model_id}.json ===")
    print(f"{'子系统':<14}{'主仓':>18}{'量化仓（还原打包）':>22}")
    for sub in subsystems.ORDER:
        a, b = main_sub.get(sub), params.get(sub)
        if a is None and not b:
            continue
        ok = a is not None and b is not None and abs(a - b) < 1
        print(f"{sub:<14}{(a if a is not None else 0):>18,.0f}{(b or 0):>22,.0f}  "
              f"{'✓' if ok else '✗'}")
        if not ok:
            bad.append(sub)
    ta, tb = sum(main_sub.values()), sum(params.values())
    print(f"{'合计':<14}{ta:>18,.0f}{tb:>22,.0f}  {'✓' if abs(ta - tb) < 1 else '✗'}")
    if abs(ta - tb) >= 1:
        bad.append("TOTAL")

    print(f"\n权重存储：{sum(nbytes.values()):,.0f} B（{dtypes.fmt_bytes(sum(nbytes.values()))}）"
          f"；主仓 {fs.get('weight_bytes_total', 0):,.0f} B"
          f"（{dtypes.fmt_bytes(fs.get('weight_bytes_total') or 0)}）")
    if fs.get("weight_bytes_total"):
        print(f"压缩比：{fs['weight_bytes_total'] / sum(nbytes.values()):.3f}×")
    print(f"声明位宽 bits={m['bits']}；打包容器 "
          f"{ {k: sorted(v) for k, v in m['containers'].items()} }")

    # ── YAML 片段 ────────────────────────────────────────────
    print("\n=== 可粘贴的 official_variants 条目（逐项自己过一遍，别直接信） ===")
    ind = " " * 6
    print(f"{' ' * 4}- dtype: {args.dtype or '<填 weights_release 里的键>'}")
    print(f"{ind}repo: {fetched.get('repo')}")
    print(f"{ind}rev: {str(fetched.get('rev') or '')[:12]}")
    if args.file:
        print(f"{ind}file: {args.file}")
    if args.component:
        print(f"{ind}component: {args.component}")
    if args.origin:
        print(f"{ind}origin: {args.origin}")
    print(f"{ind}tensor_count: {len(shapes)}")
    print(f"{ind}params_total: {sum(params.values()):.0f}")
    print(f"{ind}weight_bytes_total: {sum(nbytes.values()):.0f}")
    print(f"{ind}per_subsystem:")
    mixed = []
    for sub in subsystems.ORDER:
        if sub not in hist:
            continue
        dt, share = dominant(hist[sub])
        print(f"{ind}  {sub}: {dt}")
        if len(hist[sub]) > 1:
            mixed.append((sub, share))
    blk = {s: c for s, c in m["blocks"].items() if c}
    if blk:
        print(f"{ind}per_subsystem_block:")
        for sub in subsystems.ORDER:
            if sub in blk:
                keys = sorted(blk[sub])
                print(f"{ind}  {sub}: '{'+'.join(keys)}'"
                      + (f"   # {dict(blk[sub])}" if len(keys) > 1 else ""))
    if m["scale_dt"]:
        print(f"{ind}per_subsystem_scale:")
        for sub in subsystems.ORDER:
            if sub in m["scale_dt"]:
                dt, _ = dominant(m["scale_dt"][sub])
                print(f"{ind}  {sub}: {dt}")
    if mixed:
        print(f"\n{ind}# 下面这些桶内部不止一种精度，逐方案明细（按参数数降序），"
              f"占比 ≥{MIXED_BELOW:.0%} 才报多数值：")
        for sub, share in mixed:
            tot = sum(hist[sub].values())
            parts = ", ".join(
                f"{dt} {n:,}（{n / tot:.1%}，{m['tensors'][sub][dt]} 张）"
                for dt, n in hist[sub].most_common())
            print(f"{ind}#   {sub}: {parts}")

    if bad:
        print(f"\n!! 对账不过：{bad}。归类或打包因子有错，不要落库。", file=sys.stderr)
        return 1
    print("\n对账通过：逐子系统参数量与主仓逐位相同。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
