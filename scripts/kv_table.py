#!/usr/bin/env python3
"""把 docs/kv-cache-accounting.md 里的两张对照表改由数据生成，不手写。

手写的数字表会和记录各自漂移，而记录才是 validate.py 覆盖的那一份。这个脚本
从 data/tracking.yaml（追踪名册）与 data/derived/*.json（derive.py 的产物）生成
两个标记区：

  kv-basis-values —— bytes_basis 的取值表（定义段）
  kv-basis        —— 同一族系跨代的元素/字节/basis 对照，用来说明不同 basis 不能算倍数
  kv-roster       —— 六个族系旗舰的缓存账形状，待录的留空

取值表以前是手写的，于是漂了：正文写「三个值」，表里四行，而 derive.py 能出五个。
--check 当时照样报「最新」—— 它只比生成区，而定义段不在生成区里。闸口放过自己要管的事，
比没有闸口更糟。现在取值表也由本脚本生成，并且另外拿 data/derived 里实际出现的值对一遍：
出现了而表里没有的，报问题并返回非零。

缺字段一律印「未记」或「—」，不印 0：缺值和「真的是零」是两件事，而 0 会被当成测量值。

用法:
  python3 scripts/kv_table.py            # 写回 docs/kv-cache-accounting.md
  python3 scripts/kv_table.py --check    # 只检查是否已过期，过期返回 1（给 CI 用）
"""
import argparse
import glob
import json
import os
import re
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "kv-cache-accounting.md")

# 跨代对照取这几份：V4 一代按模型 dtype 兜底、V4.1 查到了落盘格式，差别正是要说的事。
BASIS_ROWS = ["ds-v4-flash", "ds-v4-pro", "deepseek-v41-flash"]

# 分量键 → 展示名。未登记的键原样印出 —— 新族系冒出新分量时要看得见，而不是被映射掉。
COMPONENT_LABEL = {
    "global_kv": "全局压缩 KV",
    "indexer": "index K",
    "local_window": "滑窗局部 KV",
    "full_attention": "full attention KV",
    "linear_state": "线性递归状态",
}

MARK = "<!-- {} | 由 scripts/kv_table.py 生成，勿手改 -->"
END = "<!-- {} 生成区结束 -->"

# bytes_basis 的全部取值，按**证据强度**从强到弱排，不是字母序也不是出现频率 ——
# 表的用处就是让人一眼看出哪一行的字节更可信。改 derive.py 里的 basis 时这里要跟着改，
# 漏了的话下面的 check_basis_coverage 会在它第一次出现在记录里时拦住。
#
# `declared_dtype` 当前没有任何一份记录在用（有 precision.kv_cache 的两份都已经逐分量
# 算到了 theoretical_floor），但 derive.py 走得到那一支，所以它在表里。
# 「没有记录在用」和「不存在这个取值」是两件事。
BASIS_DOC = [
    ("theoretical_floor",
     "精确计数的**理论下限**：元素 × 落盘格式（含 scale），**不含分页、环页、对齐粒度**。"
     "实际部署在它之上"),
    ("vendor_declared",
     "字节直接取厂商公布的每 token 数，本仓没有逐分量算出来。"
     "**只在公布值的 `covers` 是 `per_token` 时才会落到这个值**；"
     "本仓每个分量都有公布格式时，公布值改作对账用，basis 仍是 `theoretical_floor`"),
    ("declared_dtype",
     "记录给了 `precision.kv_cache`，字节按这个**声明的**缓存精度算。"
     "有厂商口径，但不是逐分量的精确计数"),
    ("dtype_fallback",
     "记录没给缓存精度，按模型权重 dtype 兜底。是个占位值，不是测量值"),
    ("mixed",
     "各分量 basis 不同，合计没有单一口径"),
]

_COUNT_CN = {1: "一", 2: "两", 3: "三", 4: "四", 5: "五", 6: "六", 7: "七", 8: "八"}


def load_records() -> dict:
    out = {}
    for p in glob.glob(os.path.join(ROOT, "data", "models", "*", "*.yaml")):
        d = yaml.safe_load(open(p)) or {}
        if d.get("id"):
            out[d["id"]] = d
    return out


def load_kv(mid: str):
    p = os.path.join(ROOT, "data", "derived", f"{mid}.json")
    if not os.path.exists(p):
        return None
    return (json.load(open(p)) or {}).get("kv_cache")


def num(v) -> str:
    """整数加千分位；None 印「未记」。0 只在真的是 0 时出现，不做缺值兜底。"""
    if v is None:
        return "未记"
    return f"{int(v):,}" if float(v).is_integer() else f"{v:,.4g}"


def mib(v) -> str:
    return "—" if v is None else f"{v / 1048576:.2f} MiB"


def basis_of(kv: dict) -> str:
    """整体 basis 取记录里写的那个；没写就看各分量是否一致，不一致记 mixed。"""
    top = kv.get("bytes_basis")
    if top:
        return top
    seen = {c.get("bytes_basis") for c in (kv.get("components") or {}).values()}
    seen.discard(None)
    if not seen:
        return "未记"
    return seen.pop() if len(seen) == 1 else "mixed"


def elems(kv: dict) -> str:
    """元素/token。公式路的层数或宽度只是上界时，derive.py 标 upper_bound。"""
    v = kv.get("per_token_elems")
    if v is None:
        return "未记"
    pre = "≤ " if kv.get("per_token_elems_bound") == "upper_bound" else ""
    return pre + num(v)


def shape(kv: dict) -> str:
    comps = kv.get("components") or {}
    if not comps:
        return "未记"
    return " + ".join(COMPONENT_LABEL.get(k, f"`{k}`") for k in comps)


def seen_bases() -> dict:
    """data/derived 里实际出现过的 bytes_basis -> 第一次见到它的文件名。

    整棵树递归找，不只看顶层：分量各自带 basis，extract_layers.py 的逐层产物也带。
    顶层对了而某个分量用了没登记的值，一样会在页面上渲染成没有标注的数。
    """
    out = {}

    def walk(node, where):
        if isinstance(node, dict):
            v = node.get("bytes_basis")
            if isinstance(v, str):
                out.setdefault(v, where)
            for x in node.values():
                walk(x, where)
        elif isinstance(node, list):
            for x in node:
                walk(x, where)

    for p in sorted(glob.glob(os.path.join(ROOT, "data", "derived", "*.json"))):
        try:
            walk(json.load(open(p)), os.path.basename(p))
        except (ValueError, OSError):
            # 读不动的 derived 不是这个闸口要管的事，derive.py / validate.py 会报
            continue
    return out


def check_basis_coverage(problems: list) -> None:
    """记录里出现了、取值表里没有的 basis -> 报问题。

    这是取值表唯一能自动验的那一半：表里多一个取值（derive.py 走得到但还没有记录用过）
    看不出来，而表里少一个会让那一列字节在页面上没有口径标注，正是出过的事故。
    """
    known = {k for k, _ in BASIS_DOC}
    for v, where in sorted(seen_bases().items()):
        if v not in known:
            problems.append(f"data/derived/{where} 用了 bytes_basis `{v}`，"
                            "而 BASIS_DOC 里没有这一行 —— 补进 BASIS_DOC（并想一想"
                            " site/src/lib/data.ts 的 KV_BASIS_LABEL 要不要一起补）")


def table_basis_values(records: dict, problems: list) -> list:
    """定义段的取值表。手写它漂过一次：正文写三个、表里四行、代码能出五个。"""
    check_basis_coverage(problems)
    n = _COUNT_CN.get(len(BASIS_DOC), str(len(BASIS_DOC)))
    rows = [f"`bytes_basis` 标明这一列字节是哪种来源，{n}个值：", "",
            "| basis | 含义 |", "|---|---|"]
    rows += [f"| `{k}` | {desc} |" for k, desc in BASIS_DOC]
    return rows


def table_basis(records: dict, problems: list) -> list:
    rows = ["| 版本 | 元素/token | 字节/token | basis |", "|---|---|---|---|"]
    for mid in BASIS_ROWS:
        kv = load_kv(mid)
        if not kv:
            # 静默跳过会让表少一行而没人发现 —— 删记录时实际发生过。
            # 要么补 derive，要么把它从 BASIS_ROWS 里删掉，两件都得有人决定
            problems.append(f"BASIS_ROWS 里的 {mid} 没有 data/derived/{mid}.json："
                            "补 derive，或把它从 BASIS_ROWS 里删掉")
            continue
        name = (records.get(mid) or {}).get("display_name", mid)
        rows.append(f"| {name} | **{elems(kv)}** | {num(kv.get('per_token_bytes'))} B "
                    f"| `{basis_of(kv)}` |")
    return rows


def table_roster(records: dict, problems: list) -> list:
    doc = yaml.safe_load(open(os.path.join(ROOT, "data", "tracking.yaml"))) or {}
    rows = ["| 族系旗舰 | 类型 | 缓存分量 | 元素/token | 字节/token | 每序列固定 | basis |",
            "|---|---|---|---|---|---|---|"]
    # 名册没给 model_id 时按家族找主线记录：记录先到、名册没跟着更新是实际发生过的漂移，
    # 这一行要把它说出来，而不是跟着名册报「待录」。
    by_family = {}
    for mid, rec in records.items():
        if (rec.get("lineage") or {}).get("track") == "main":
            by_family.setdefault(rec.get("family"), []).append((rec.get("released") or "", mid))

    for f in doc.get("families") or []:
        name = f.get("flagship", f.get("family", "?"))
        mid = f.get("model_id")
        if not mid:
            cand = sorted(by_family.get(f.get("family")) or [])
            mid = cand[-1][1] if cand else None
        kv = load_kv(mid) if mid else None
        if not kv:
            if mid:
                why = "已有记录，未 derive" + ("；名册仍记 pending" if f.get("status") == "pending" else "")
            else:
                why = "待录"
            rows.append(f"| {name} | *{why}* | | | | | |")
            continue
        rows.append(f"| {name} | {kv.get('attention_type', '未记')} | {shape(kv)} "
                    f"| **{elems(kv)}** | {num(kv.get('per_token_bytes'))} B "
                    f"| {mib(kv.get('per_sequence_bytes'))} | `{basis_of(kv)}` |")
    return rows


BLOCKS = {"kv-basis-values": table_basis_values,
          "kv-basis": table_basis,
          "kv-roster": table_roster}


def render(text: str, records: dict) -> tuple:
    """重新生成每个标记区，并把「生成不出来」的情况收集成 problems。

    这些情况以前只往 stderr 印一行就继续，于是 --check 仍然返回 0：标记区被删掉
    或 BASIS_ROWS 里的记录没了，表会停在旧值上而闸口全绿。闸口放过自己要管的事，
    比没有闸口更糟 —— 它会让人以为查过了。
    """
    problems = []
    for key, fn in BLOCKS.items():
        begin, end = MARK.format(key), END.format(key)
        i, j = text.find(begin), text.find(end)
        if i < 0 or j < 0:
            problems.append(f"docs 里找不到 {key} 的标记区（起始或结束标记被删了）—— "
                            "没有标记区就没人会发现这张表过期")
            continue
        body = "\n".join(fn(records, problems))
        text = text[:i] + begin + "\n" + body + "\n" + text[j:]
    return text, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查是否过期，不写文件")
    args = ap.parse_args()

    old = open(DOC).read()
    new, problems = render(old, load_records())

    for p in problems:
        print(f"! kv_table: {p}", file=sys.stderr)

    if args.check:
        if problems:
            return 1
        if old == new:
            print("kv_table: docs/kv-cache-accounting.md 是最新的")
            return 0
        print("! kv_table: docs/kv-cache-accounting.md 的表和记录不一致，"
              "跑 scripts/kv_table.py 重新生成", file=sys.stderr)
        return 1

    # 写入模式下也返回非零：表已按能生成的部分更新，但有一块生成不出来，
    # 需要人去决定（补 derive、改 BASIS_ROWS、还是把标记区加回去）
    if old == new:
        print("kv_table: 无变化")
        return 1 if problems else 0
    with open(DOC, "w") as fh:
        fh.write(new)
    print(f"kv_table: 已更新 {os.path.relpath(DOC, ROOT)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
