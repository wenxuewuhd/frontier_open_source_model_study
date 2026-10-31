#!/usr/bin/env python3
"""新范式成熟度分级的推导：L1 / L2 / L3 从 data/models 的记录算出来，不手写。

分级是 PLAN.md §2 的 L0 论文 → L1 开源权重 → L2 主流引擎支持 → L3 昇腾可用 →
L4 生产部署。五档的事实来源不是一处：

  L0  论文         外部文献，手写在条目的 paper 块
  L1  开源权重     **本仓记录**：有没有记录命中这个范式的结构判据
  L2  主流引擎     **本仓记录**：命中记录的 platforms 里有没有非昇腾引擎在支持
  L3  昇腾可用     **本仓记录**：同上，昇腾侧
  L4  生产部署     自测与内部状态，公开仓看不见（归 data-private）

中间三档本仓已经有事实了，所以一律推，不手写 —— 手写一遍就是第二份会漂的拷贝，
而且会变成一句没有判据的断言（paradigm.schema.json 不给它们留位置）。

**三态契约在这里同样生效。** 每一档的空都要说出自己是哪一种空：
没有平台条目 = 未录（调研进度），有条目而全是 status: none = 确认无（信息）。
把两者都渲染成「不支持」，就等于把没查过说成查过了。

**产物里没有时间戳。** 它是输入的纯函数，于是 validate.py 能逐字节比对重算结果
来判它有没有过期；放个 at 字段进去，那道闸就只能改成逐字段比，而「哪些字段不参与比对」
又成了一处要维护的名单。

用法:
  python3 scripts/derive_paradigms.py              # 写 data/derived/paradigms.json
  python3 scripts/derive_paradigms.py --check      # 只对账，过期就非零退出
  python3 scripts/derive_paradigms.py --print      # 打到 stdout
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import columns as C  # noqa: E402
from lib import yamlio  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 昇腾侧的入口判据。引擎名在 schema 里是自由字符串（枚举会在每次出新引擎时卡住录入），
# 所以这里只能按词匹配 —— **判据写进产物**，读的人能看见它是怎么分的，
# 而不是在脚本里藏一张名单。生成类的昇腾适配走 Ascend-SACT / ModelArts 独立配方，
# 不是 vllm-ascend，所以 SACT 也在里头（model.schema.json 的 engine 描述写明了这件事）。
ASCEND_PATTERN = r"昇腾|Ascend|ascend|MindIE|CANN|SACT|\b91[0-9]\b|\b95[0-9]\b"
# 算「这一档有」的平台状态。unknown 不算 —— 它的字面意思就是未查。
_OK_STATUS = {"day0", "supported", "partial"}


def _rung(entries):
    """一组平台条目 → 这一档的三态与依据。

    三种空分得开：没有条目是未录；有条目而全是 none 是确认无（查过了，不支持）。
    """
    if not entries:
        return {"state": "unsurveyed", "entries": [], "engines": {}}
    ok = [e for e in entries if e["status"] in _OK_STATUS]
    engines = {}
    for e in entries:
        engines.setdefault(e["engine"], {}).setdefault(e["status"], 0)
        engines[e["engine"]][e["status"]] += 1
    if ok:
        state = "present"
    elif all(e["status"] == "none" for e in entries):
        state = "confirmed_absent"
    else:
        state = "unsurveyed"
    return {"state": state, "entries": entries, "engines": engines}


def compute(root=ROOT):
    recs = {}
    for p in sorted(glob.glob(os.path.join(root, "data", "models", "**", "*.yaml"),
                              recursive=True)):
        d = yamlio.load(p)
        if isinstance(d, dict) and d.get("id"):
            recs[d["id"]] = d

    derived = {}
    for i in recs:
        dp = os.path.join(root, "data", "derived", f"{i}.json")
        if os.path.exists(dp):
            try:
                derived[i] = json.load(open(dp, encoding="utf-8"))
            except ValueError:
                derived[i] = None

    files = sorted(glob.glob(os.path.join(root, "data", "paradigms", "*.yaml")))
    out = {}
    for p in files:
        doc = yamlio.load(p)
        pid = doc["id"]
        rules = doc.get("match") or []
        hit = sorted(i for i, d in recs.items()
                     if any(C.test(r, d, derived.get(i)) for r in rules))
        fams = sorted({recs[i].get("family") for i in hit if recs[i].get("family")})

        asc, other = [], []
        for i in hit:
            for e in (recs[i].get("platforms") or []):
                row = {"model": i, "engine": e.get("engine") or "?",
                       "hardware": e.get("hardware") or "?",
                       "status": e.get("status") or "unknown"}
                (asc if re.search(ASCEND_PATTERN, f"{row['engine']} {row['hardware']}")
                 else other).append(row)

        prod = doc.get("production") or {}
        out[pid] = {
            "name": doc.get("name"),
            "category": doc.get("category"),
            # 判据立不立得起来，和这一档有没有数据是两件事。合成一个状态会让
            # 「没录」与「录不了」长得一样 —— loop-transformer 是后者，diffusion-lm 是前者。
            "criterion": "expressible" if rules else "inexpressible",
            "l0": {"state": "present" if doc.get("paper") else "unsurveyed"},
            "l1": {"state": "present" if hit else "unsurveyed",
                   "records": hit, "families": fams,
                   "watchlist": len(doc.get("watchlist") or [])},
            "l2": _rung(sorted(other, key=lambda r: (r["model"], r["engine"]))),
            "l3": _rung(sorted(asc, key=lambda r: (r["model"], r["engine"]))),
            "l4": {"state": prod.get("state", "unsurveyed")},
        }

    return {
        "generated": {
            "by": "scripts/derive_paradigms.py",
            "inputs": {"paradigms": [os.path.relpath(f, root) for f in files],
                       "models": len(recs)},
            "note": "无时间戳：产物是输入的纯函数，validate.py 逐字节比对重算结果判过期",
        },
        "ascend_pattern": ASCEND_PATTERN,
        "ok_status": sorted(_OK_STATUS),
        "paradigms": out,
    }


def dumps(doc):
    return json.dumps(doc, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="只对账，过期就非零退出")
    ap.add_argument("--print", dest="to_stdout", action="store_true")
    a = ap.parse_args()

    doc = compute()
    text = dumps(doc)
    path = os.path.join(ROOT, "data", "derived", "paradigms.json")

    if a.to_stdout:
        sys.stdout.write(text)
        return 0
    if a.check:
        if not os.path.exists(path):
            print(f"ERROR {os.path.relpath(path, ROOT)} 不存在，跑一次 derive_paradigms.py")
            return 1
        cur = open(path, encoding="utf-8").read()
        if cur != text:
            print(f"ERROR {os.path.relpath(path, ROOT)} 过期了 —— 记录变了而产物没重算。"
                  "跑 python scripts/derive_paradigms.py")
            return 1
        n = len(doc["paradigms"])
        hit = sum(1 for v in doc["paradigms"].values() if v["l1"]["state"] == "present")
        print(f"paradigms.json 是最新的（{n} 条范式，{hit} 条有 L1 记录）")
        return 0

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(f"写出 {os.path.relpath(path, ROOT)}：{len(doc['paradigms'])} 条范式")
    for pid, v in sorted(doc["paradigms"].items()):
        print(f"  {pid}: L1={v['l1']['state']}({len(v['l1']['records'])} 份) "
              f"L2={v['l2']['state']} L3={v['l3']['state']} 判据={v['criterion']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
