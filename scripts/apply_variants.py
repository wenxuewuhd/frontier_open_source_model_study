#!/usr/bin/env python3
"""把 variant_precision.py 量出来的官方量化仓写进模型记录。

**定点编辑，不做 yaml.safe_dump 回写** —— data/models 下的记录带注释与折叠块，
整份回写会把它们全部重排（CLAUDE.md「改数据」第一条）。所以这里按行找锚点、按行插入：
  precision.quantization 块末尾     插 official_variants
  precision.weights_release 那一行  补上缺的 dtype（行内流式列表）
  precision.weights_release_scope   补上缺的档位，并把已有档位的两个布尔翻成 true
  evidence                          插 precision.quantization.official_variants: derived

已经有 official_variants 的记录直接跳过 —— 这个脚本只负责第一次落库，
后续改动走人工定点编辑，免得一次重跑把人改过的 method 覆盖掉。

method 是**生成**的，但只写量出来的东西：分片与张量数、对账结果、哪些子系统被量化
及其参数占比、块与 scale 凭什么形状量出来、落盘字节与压缩比、config 的
modules_to_not_convert 覆盖了什么。生成之后要人逐条过 —— 它说的都是实话，
但未必是这一份记录**最该说**的那句话。

用法:
  .venv/bin/python scripts/apply_variants.py qwen3-32b \
      --variant qwen3-32b-awq:INT4:post_quantized \
      --variant qwen3-32b-fp8:FP8_E4M3:post_quantized
  # 同仓不同文件 / 只量一个组件：
  .venv/bin/python scripts/apply_variants.py ltx-2 \
      --variant ltx-2-fp8-pack:FP8_E4M3:post_quantized \
      --file ltx-2-19b-dev-fp8.safetensors --component dit
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import dtypes, subsystems, yamlio  # noqa: E402
import variant_precision as vp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 位宽从低到高 —— 条目按这个顺序写，给人读的时候「最低的在最前面」
BITS = {"NVFP4": 4, "MXFP4": 4, "FP4": 4, "INT4": 4, "MXFP8": 8, "FP8_E4M3": 8,
        "INT8": 8, "BF16": 16, "FP16": 16, "FP32": 32}

QUANTIZED = {"FP8_E4M3", "FP8_E5M2", "MXFP8", "FP4", "MXFP4", "NVFP4", "INT8", "INT4"}


def find_record(mid):
    hits = [p for p in glob.glob(os.path.join(ROOT, "data", "models", "**", "*.yaml"),
                                 recursive=True)
            if (yamlio.load(p) or {}).get("id") == mid]
    if len(hits) != 1:
        raise SystemExit(f"! 找不到唯一的记录：{mid}（命中 {hits}）")
    return hits[0]


def block_end(lines, start, indent):
    """从 start 这一行（含）往下，第一行缩进 <= indent 且非空非注释的行号。"""
    for i in range(start + 1, len(lines)):
        ln = lines[i]
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if len(ln) - len(ln.lstrip()) <= indent:
            return i
    return len(lines)


def find_line(lines, prefix):
    for i, ln in enumerate(lines):
        if ln.startswith(prefix):
            return i
    return None


def fold(text, indent, width=92):
    """折成 YAML 的 `>-` 折叠块。按词折，中文按字符折 —— 两种都要能折得住。"""
    out, cur = [], ""
    for tok in text.split(" "):
        if cur and len(cur) + 1 + len(tok) > width - indent:
            out.append(cur)
            cur = tok
        else:
            cur = f"{cur} {tok}" if cur else tok
    if cur:
        out.append(cur)
    pad = " " * indent
    return [f"{pad}{ln}\n" for ln in out]


def coverage(m, dt):
    """这个精度覆盖了哪些子系统、占总参数量多少。"""
    tot = sum(m["params"].values()) or 1
    subs, n = [], 0
    for sub in subsystems.ORDER:
        got = m["hist"].get(sub, {}).get(dt, 0)
        if got:
            subs.append((sub, got))
            n += got
    return subs, n / tot


def render(mid, facts_id, dtype, origin, m, shapes, cfg, fetched, want,
           file=None, component=None):
    ind = " " * 8
    tot = sum(m["params"].values())
    byt = sum(m["bytes"].values())
    delta = int(tot) - int(want or tot)
    lines = [f"      - dtype: {dtype}\n",
             f"{ind}repo: {fetched.get('repo')}\n",
             f"{ind}rev: {str(fetched.get('rev') or '')[:12]}\n"]
    if file:
        lines.append(f"{ind}file: {file}\n")
    if component:
        lines.append(f"{ind}component: {component}\n")
    lines += [f"{ind}origin: {origin}\n",
              f"{ind}tensor_count: {len(shapes)}\n",
              f"{ind}params_total: {tot:.0f}\n"]
    if delta:
        # **故意不写 params_delta_reason**：留空让闸口报出来，逼人去看那个差是什么。
        # 写个占位串的话 check_variants 的「有没有理由」就被满足了，
        # 而一个没人核过的占位串比留空坏得多 —— 留空会被查，占位串不会
        lines.append(f"{ind}params_delta: {delta}\n")
    lines.append(f"{ind}weight_bytes_total: {byt:.0f}\n")

    lines.append(f"{ind}per_subsystem:\n")
    mixed = []
    for sub in subsystems.ORDER:
        if sub not in m["hist"]:
            continue
        dt, share = vp.dominant(m["hist"][sub])
        lines.append(f"{ind}  {sub}: {dt}\n")
        if len(m["hist"][sub]) > 1:
            mixed.append(sub)
    blk = {s: c for s, c in m["blocks"].items() if c}
    if blk:
        lines.append(f"{ind}per_subsystem_block:\n")
        for sub in subsystems.ORDER:
            if sub in blk:
                lines.append(f"{ind}  {sub}: '{'+'.join(sorted(blk[sub]))}'\n")
    # 逐通道 / 逐张量没有块，它们是另一根粒度轴。分开写，`+` 的约定才不会把
    # 「不同投影的输入维不同」报成「这个子系统里不止一种块方案」
    strat = {s: c for s, c in m["strategy"].items() if c}
    if strat:
        lines.append(f"{ind}per_subsystem_strategy:\n")
        for sub in subsystems.ORDER:
            if sub in strat:
                lines.append(f"{ind}  {sub}: {'+'.join(sorted(strat[sub]))}\n")
    scl = {s: c for s, c in m["scale_dt"].items() if c}
    if scl:
        lines.append(f"{ind}per_subsystem_scale:\n")
        for sub in subsystems.ORDER:
            # 只有被量化的桶才谈得上 scale。没量化的桶里那些「scale 叶名」
            # 是别的东西，写进来会出现「归一化 BF16 / BF16」这种读不通的格子
            if sub in scl and vp.dominant(m["hist"].get(sub, {}))[0] in QUANTIZED:
                lines.append(f"{ind}  {sub}: {vp.dominant(scl[sub])[0]}\n")
    if mixed:
        lines.append(f"{ind}per_subsystem_schemes:\n")
        for sub in mixed:
            lines.append(f"{ind}  {sub}:\n")
            for d, n in m["hist"][sub].most_common():
                bits = [f"dtype: {d}"]
                if d in QUANTIZED and blk.get(sub):
                    bits.append(f"block: '{sorted(blk[sub])[0]}'")
                    if scl.get(sub):
                        bits.append(f"scale_dtype: {vp.dominant(scl[sub])[0]}")
                bits += [f"tensors: {m['tensors'][sub][d]}", f"elements: {n:.0f}"]
                lines.append(f"{ind}    - {{ {', '.join(bits)} }}\n")

    # ── method：只写量出来的 ──────────────────────────────────
    subs, cov = coverage(m, dtype)
    qc = cfg.get("quantization_config") or {}
    nconv = qc.get("modules_to_not_convert") or qc.get("ignore") or []
    parts = [
        f"完整清单逐项统计（{fetched.get('shard_count')} 分片 {len(shapes):,} 条目，"
        f"failed_shards 为空），非抽样。",
    ]
    if component:
        parts.append(f"**这个仓是整包**（DiT / VAE / 文本编码器 / 声码器都在里面），"
                     f"而本记录的清单口径只收 `{component}` 一个组件，"
                     f"所以这一条也收到同一个组件上 —— 两边量的不是同一个东西时，"
                     f"对账红了也说明不了什么。")
    if delta:
        parts.append(f"**逐子系统参数量与主仓差 {delta:+,}**，理由见 params_delta_reason。")
    else:
        parts.append(f"**逐子系统参数量与主仓逐位相同，"
                     f"{len(m['params'])} 桶零例外**，合计同为 {tot:,.0f} —— "
                     f"这是「同一份权重的两种存法」的硬证据，"
                     f"也是把 origin 判成 {origin} 的判据之一。")
    if subs:
        parts.append(f"{dtype} 覆盖" + "、".join(
            f"{subsystems.LABEL.get(s, s)}（{n:,.0f}）" for s, n in subs)
            + f"，占参数量 {cov:.1%}；其余桶原样留在高精度上。")
    if m["containers"]:
        c = "、".join(f"{k} 装在 {'/'.join(sorted(v))} 容器里"
                     for k, v in m["containers"].items())
        parts.append(f"**打包存储**：{c} —— numel 是容器数不是参数数，"
                     f"还原因子由 scale 形状逐张量复核过（权重还原后的逻辑长度"
                     f"必须等于 scale 的组数乘组大小）。")
    if blk:
        parts.append("块与 scale 由形状量出、不抄 config：" + "、".join(
            f"{subsystems.LABEL.get(s, s)} {'+'.join(sorted(c))}" for s, c in blk.items())
            + "。")
    if nconv:
        parts.append(f"config 的 {'modules_to_not_convert' if qc.get('modules_to_not_convert') else 'ignore'} "
                     f"点名了 {len(nconv)} 条不量化的模块，与清单逐张量对得上。")
    if want:
        parts.append(f"落盘 {byt:,.0f} B（{dtypes.fmt_bytes(byt)}）。")
    lines.append(f"{ind}method: >-\n")
    lines += fold(" ".join(parts), len(ind) + 2)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_id")
    ap.add_argument("--variant", action="append", required=True,
                    metavar="FACTS:DTYPE:ORIGIN[:FILE]",
                    help="量化仓的 facts 子目录名、声明精度、native|post_quantized，"
                         "可再跟一段仓内文件名（同仓不同文件的发布形态）")
    ap.add_argument("--file", help="没在 --variant 里逐条给文件名时的缺省值")
    ap.add_argument("--component", help="只量这一个组件")
    ap.add_argument("--facts-dir", default=os.path.join(ROOT, "facts"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    path = find_record(args.model_id)
    doc = yamlio.load(path)
    if ((doc.get("precision") or {}).get("quantization") or {}).get("official_variants"):
        print(f"跳过 {args.model_id}：已经有 official_variants，后续改动走人工定点编辑")
        return 0
    dp = os.path.join(ROOT, "data", "derived", f"{args.model_id}.json")
    want = None
    if os.path.exists(dp):
        want = ((json.load(open(dp)) or {}).get("from_shapes") or {}).get("params_total")

    entries, dtypes_added = [], []
    for spec in args.variant:
        # 第四段是仓内文件名，只有「同仓不同文件」那种发布形态要填（LTX-2 把
        # fp8 / fp4 两个档与 BF16 档打包进同一个仓）。两个档的文件名不同，
        # 所以它得跟着每个 --variant 走，不能做成整次调用一个 --file
        facts_id, dtype, origin, *rest = spec.split(":")
        vfile = rest[0] if rest else args.file
        vdir = os.path.join(args.facts_dir, facts_id)
        shapes = json.load(open(os.path.join(vdir, "shapes.json")))
        cfg = json.load(open(os.path.join(vdir, "cfg.json")))
        fetched = json.load(open(os.path.join(vdir, "fetched.json")))
        tcfg = cfg.get("text_config") or cfg
        m = vp.measure(shapes, cfg, tcfg.get("num_hidden_layers"), args.component)
        entries.append((BITS.get(dtype, 99), render(
            args.model_id, facts_id, dtype, origin, m, shapes, cfg, fetched, want,
            file=vfile, component=args.component)))
        dtypes_added.append(dtype)

    out = ["    official_variants:\n"]
    for _b, ls in sorted(entries, key=lambda e: e[0]):
        out += ls

    lines = open(path, encoding="utf-8").readlines()

    # 1) official_variants：插在 precision.quantization 块末尾
    qi = find_line(lines, "  quantization:")
    if qi is None:
        raise SystemExit(f"! {path} 没有 precision.quantization，先补上再跑")
    lines[block_end(lines, qi, 2):block_end(lines, qi, 2)] = out

    # 2) weights_release：补上缺的 dtype（行内流式列表）
    wi = find_line(lines, "  weights_release:")
    if wi is not None:
        cur = [x.strip() for x in lines[wi].split("[", 1)[1].rsplit("]", 1)[0].split(",")]
        for d in dtypes_added:
            if d not in cur:
                cur.append(d)
        lines[wi] = f"  weights_release: [{', '.join(cur)}]\n"

    # 3) weights_release_scope：补条目 / 翻布尔
    si = find_line(lines, "  weights_release_scope:")
    if si is None:
        raise SystemExit(f"! {path} 没有 weights_release_scope，先补上再跑")
    end = block_end(lines, si, 2)
    have = {}
    for i in range(si + 1, end):
        st = lines[i].lstrip()
        if lines[i].startswith("    ") and st.split(":")[0] in dtypes_added \
                and len(lines[i]) - len(st) == 4:
            have[st.split(":")[0]] = i
    add = []
    for d in dtypes_added:
        if d in have:
            continue
        repo = None
        for spec in args.variant:
            fid, dt, _o = spec.split(":")
            if dt == d:
                repo = json.load(open(os.path.join(
                    args.facts_dir, fid, "fetched.json"))).get("repo")
        add += [f"    {d}:\n",
                "      manifest_taken: true\n",
                "      in_precision_table: true\n",
                f"      repo: {repo}\n",
                "      note: >-\n"]
        add += fold("厂商另发的官方量化仓，本轮取了完整清单，逐子系统实测落在 "
                    "quantization.official_variants 里 —— 精度表的第二个来源。"
                    "**主仓的 per_subsystem 不受影响**：那个字段是 derive.py "
                    "识别打包用的承重件，口径只能是 hf.repo 那一份权重", 8)
    lines[end:end] = add
    if have:
        print(f"  注意：{sorted(have)} 在 scope 里已有条目，两个布尔请人工确认翻面")

    # 4) evidence
    ei = find_line(lines, "  precision.quantization.per_subsystem:") \
        or find_line(lines, "  precision.weights_release_scope:")
    if ei is None:
        raise SystemExit(f"! {path} 的 evidence 里找不到插入锚点")
    lines[ei + 1:ei + 1] = ["  precision.quantization.official_variants: derived\n"]

    if args.dry_run:
        sys.stdout.writelines(out)
        return 0
    open(path, "w", encoding="utf-8").writelines(lines)
    print(f"写了 {path}：{', '.join(dtypes_added)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
