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
    """量化仓声明的权重位宽。GPTQ 写 bits，compressed-tensors 写 num_bits。"""
    qc = cfg.get("quantization_config") or {}
    for k in ("bits", "num_bits", "weight_bits"):
        if qc.get(k):
            return int(qc[k])
    return None


def logical_dtype_for(name, dtype, bits):
    """这个张量的逻辑精度。

    只有 `qweight` 需要翻译：它是容器（I32），逻辑精度由 config 的 bits 定。
    其余张量的 safetensors dtype 就是逻辑精度 —— FP8 仓的 `weight` 直接是 F8_E4M3，
    没有打包这一层。
    """
    if name.lower().rsplit(".", 1)[-1] == "qweight" and bits:
        return {4: "INT4", 8: "INT8"}.get(bits) or f"INT{bits}"
    return dtypes.norm(dtype)


def measure(shapes, cfg, mtp_from):
    bits = logical_bits(cfg)
    params = collections.Counter()          # sub -> 逻辑参数数
    nbytes = collections.Counter()          # sub -> 落盘字节
    hist = collections.defaultdict(collections.Counter)   # sub -> 逻辑 dtype -> 参数数
    tensors = collections.defaultdict(collections.Counter)  # sub -> 逻辑 dtype -> 张量数
    containers = collections.defaultdict(set)             # 逻辑 dtype -> 容器 dtype
    scale_dt = collections.defaultdict(collections.Counter)  # sub -> scale dtype -> 元素数
    # 块的两个因子分开量：group 式（scales 沿 in 分组）与 tile 式（scale_inv 二维）
    blocks = collections.defaultdict(collections.Counter)  # sub -> 块字符串 -> 张量数
    shape_of = {}

    for name, meta in shapes.items():
        shape_of[name] = meta.get("shape") or []
    for name, meta in shapes.items():
        n = numel(meta.get("shape"))
        dt = dtypes.norm(meta.get("dtype", ""))
        _comp, sub = subsystems.classify2(name, mtp_from)
        nbytes[sub] += n * dtypes.nbytes(dt)
        if dtypes.is_scale_tensor(name, dt):
            scale_dt[sub][dt] += n
            continue
        if dtypes.is_metadata_tensor(name, dt):
            continue
        lg = logical_dtype_for(name, dt, bits)
        f = dtypes.packing_factor(lg, dt)
        if f > 1:
            containers[lg].add(dt)
        params[sub] += n * f
        hist[sub][lg] += n * f
        tensors[sub][lg] += 1
        b = block_of(name, shape_of, f)
        if b:
            blocks[sub][b] += 1
    return {"bits": bits, "params": params, "bytes": nbytes, "hist": hist,
            "tensors": tensors, "containers": containers, "scale_dt": scale_dt,
            "blocks": blocks}


def block_of(name, shape_of, packing):
    """这个权重张量的实测量化块，写成 `[沿 out, 沿 in]`。

    两种声明法量出来的形状不同，所以判据也不同：
      GPTQ        `qweight [in/8, out]` 配 `scales [groups, out]` —— 沿 in 分组，
                  一个 scale 覆盖 in/groups 个元素、一个 out 通道 → `[1, N]`
      FP8 的 tile `weight [out, in]` 配 `weight_scale_inv [ceil(out/T), ceil(in/T)]`
                  —— 两维都分块 → `[T, T]`
    **不抄 config 的 weight_block_size**（model-audit 规则 2：config 描述模块配置，
    不描述落盘形状；V4 一代声明 [128,128] 而 FP4 专家实际是 [1,32]）。
    """
    stem = name.rsplit(".", 1)[0]
    leaf = name.lower().rsplit(".", 1)[-1]
    w = shape_of.get(name) or []
    if leaf == "qweight" and packing > 1:
        sc = shape_of.get(f"{stem}.scales")
        if sc and len(w) == 2 and len(sc) == 2 and sc[0]:
            logical_in = w[0] * packing
            g = logical_in // sc[0]
            return f"[1,{g}]"
        return None
    if leaf == "weight":
        for suffix in ("weight_scale_inv", "weight_scale", "scale"):
            sc = shape_of.get(f"{stem}.{suffix}")
            if not sc:
                continue
            if len(w) == 2 and len(sc) == 2 and sc[0] and sc[1]:
                # 向上取整的那一维按本仓记法写成 `Nceil`
                t0 = _tile(w[0], sc[0])
                t1 = _tile(w[1], sc[1])
                return f"[{t0},{t1}]"
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
    ap.add_argument("--facts-dir", default=os.path.join(ROOT, "facts"))
    ap.add_argument("--derived-dir", default=os.path.join(ROOT, "data", "derived"))
    args = ap.parse_args()

    vdir = os.path.join(args.facts_dir, args.variant_facts)
    shapes = json.load(open(os.path.join(vdir, "shapes.json")))
    cfg = json.load(open(os.path.join(vdir, "cfg.json")))
    fetched = json.load(open(os.path.join(vdir, "fetched.json")))
    tcfg = cfg.get("text_config") or cfg
    mtp_from = tcfg.get("num_hidden_layers")

    m = measure(shapes, cfg, mtp_from)
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
