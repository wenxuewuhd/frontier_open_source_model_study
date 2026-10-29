#!/usr/bin/env python3
"""把 docs/kv-cache-accounting.md 里的两张对照表改由数据生成，不手写。

手写的数字表会和记录各自漂移，而记录才是 validate.py 覆盖的那一份。这个脚本
从 data/tracking.yaml（追踪名册）与 data/derived/*.json（derive.py 的产物）生成
两个标记区：

  kv-basis   —— 同一族系跨代的元素/字节/basis 对照，用来说明不同 basis 不能算倍数
  kv-roster  —— 六个族系旗舰的缓存账形状，待录的留空

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


def table_basis(records: dict) -> list:
    rows = ["| 版本 | 元素/token | 字节/token | basis |", "|---|---|---|---|"]
    for mid in BASIS_ROWS:
        kv = load_kv(mid)
        if not kv:
            continue
        name = (records.get(mid) or {}).get("display_name", mid)
        rows.append(f"| {name} | **{elems(kv)}** | {num(kv.get('per_token_bytes'))} B "
                    f"| `{basis_of(kv)}` |")
    return rows


def table_roster(records: dict) -> list:
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


BLOCKS = {"kv-basis": table_basis, "kv-roster": table_roster}


def render(text: str, records: dict) -> str:
    for key, fn in BLOCKS.items():
        begin, end = MARK.format(key), END.format(key)
        i, j = text.find(begin), text.find(end)
        if i < 0 or j < 0:
            print(f"  ! docs 里找不到 {key} 的标记区，跳过 —— 标记被删掉了就不会有人发现表过期",
                  file=sys.stderr)
            continue
        body = "\n".join(fn(records))
        text = text[:i] + begin + "\n" + body + "\n" + text[j:]
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只检查是否过期，不写文件")
    args = ap.parse_args()

    old = open(DOC).read()
    new = render(old, load_records())

    if args.check:
        if old == new:
            print("kv_table: docs/kv-cache-accounting.md 是最新的")
            return 0
        print("! kv_table: docs/kv-cache-accounting.md 的表和记录不一致，"
              "跑 scripts/kv_table.py 重新生成", file=sys.stderr)
        return 1

    if old == new:
        print("kv_table: 无变化")
        return 0
    with open(DOC, "w") as fh:
        fh.write(new)
    print(f"kv_table: 已更新 {os.path.relpath(DOC, ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
