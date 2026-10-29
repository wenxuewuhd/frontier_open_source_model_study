#!/usr/bin/env python3
"""把 fetch_dtypes.py 的实测结果写进模型记录的 precision.quantization。

单独一个脚本而不是让 fetch 直接写记录：取数和改记录分开，取数可以重跑多次，
改记录是一次性的、要能看 diff。

对已经有 per_subsystem 的记录（比如从全量清单手工核过的旗舰），默认只**比对**不覆盖 ——
抽样结果和全量结果不一致时，应该是人去看为什么，而不是让抽样把全量盖掉。
"""
import argparse
import glob
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import yamlio  # noqa: E402


def to_enum(v):
    """schema 的 dtype 是枚举。抽样里出现多个值时记 MIXED，细节留在 facts 里。"""
    return "MIXED" if v and "/" in v else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="data/models")
    ap.add_argument("--facts-dir", default="facts")
    ap.add_argument("--overwrite", action="store_true",
                    help="已有 per_subsystem 时也覆盖。默认只比对并报告差异")
    ap.add_argument("--skip", nargs="*", default=[],
                    help="即使带 --overwrite 也不动的模型 id。用来保住从全量清单核过的记录")
    args = ap.parse_args()

    conflicts = 0

    # 只有量化过的子系统才有「scale」。norm 这类子系统的附属张量是 bias，
    # 不是量化 scale，写进 per_subsystem_scale 会让「归一化 FP32 / FP32」这种读不通的格子出现。
    QUANTIZED = {"FP8_E4M3", "FP8_E5M2", "FP8_E8M0", "FP4", "MXFP4", "NVFP4",
                 "HIF4", "FP6", "INT8", "INT4", "UINT8"}

    for path in sorted(glob.glob(os.path.join(args.models_dir, "**", "*.yaml"), recursive=True)):
        doc = yamlio.load(path)
        mid = doc.get("id")
        fp = os.path.join(args.facts_dir, mid or "", "subsystem_dtypes.json")
        if not mid or not os.path.exists(fp):
            continue
        meas = json.load(open(fp))
        subs = meas.get("subsystems") or {}
        if not subs:
            continue

        per = {k: to_enum(v["dtype"]) for k, v in subs.items() if v.get("dtype")}
        per_scale = {k: to_enum(v["scale_dtype"]) for k, v in subs.items()
                     if v.get("scale_dtype") and per.get(k) in QUANTIZED}
        # 同一子系统里出现多种块布局时用 + 连接，不用 / ——
        # 格子里「权重精度 / scale 精度 / 块」已经用 / 分隔三个字段，
        # 块本身再用 / 就读不出「FP4 / FP8_E8M0 / [1,32] / [32,32]」的后两项都是块
        per_block = {k: "+".join(v["block_share"]) if v.get("block_share") else v["block"]
                     for k, v in subs.items() if v.get("block")}

        q = doc.setdefault("precision", {}).setdefault("quantization", {})
        old = q.get("per_subsystem")
        if old and (mid in args.skip or not args.overwrite):
            diff = {k: (old.get(k), per.get(k)) for k in set(old) | set(per)
                    if old.get(k) != per.get(k)}
            if diff:
                conflicts += 1
                print(f"!! {mid}: 已有 per_subsystem 与抽样不一致，未改动")
                for k, (a, b) in sorted(diff.items()):
                    print(f"     {k:16s} 记录={a}  抽样={b}")
            else:
                # 权重精度一致就不动它，但 scale 精度是新增的一项，补上不算覆盖
                added = False
                if per_block and not q.get("per_subsystem_block"):
                    # 实测 block 是新增项，补上不算覆盖已有的 per_subsystem
                    q["per_subsystem_block"] = per_block
                    doc.setdefault("evidence", {})[
                        "precision.quantization.per_subsystem_block"] = "derived"
                    lines = open(path).read().splitlines(keepends=True)
                    at = next(i for i, l in enumerate(lines)
                              if l.rstrip() == "    per_subsystem:")
                    at += 1
                    while at < len(lines) and lines[at].startswith("      "):
                        at += 1
                    ins = (["    per_subsystem_block:\n"]
                           + [f"      {k}: '{v}'\n" for k, v in per_block.items()])
                    open(path, "w").write("".join(lines[:at] + ins + lines[at:]))
                    added = True
                if per_scale and not q.get("per_subsystem_scale"):
                    q["per_subsystem_scale"] = per_scale
                    q.setdefault("subsystem_method", meas.get("method", ""))
                    ev = doc.setdefault("evidence", {})
                    ev["precision.quantization.per_subsystem_scale"] = "derived"
                    ev["precision.quantization.subsystem_method"] = "derived"
                    # 手工维护的记录要保住原格式，所以按行插入而不是整份重排 yaml。
                    # 插入点是 per_subsystem 的最后一个子项之后 —— 按行走，
                    # 用缩进判断子项边界（早先按字符偏移算，插在了键和它的子项中间）。
                    lines = open(path).read().splitlines(keepends=True)
                    at = next(i for i, l in enumerate(lines)
                              if l.rstrip() == "    per_subsystem:")
                    at += 1
                    while at < len(lines) and lines[at].startswith("      "):
                        at += 1
                    ins = (["    per_subsystem_scale:\n"]
                           + [f"      {k}: {v}\n" for k, v in per_scale.items()]
                           + [f"    subsystem_method: {q['subsystem_method']}\n"])
                    open(path, "w").write("".join(lines[:at] + ins + lines[at:]))
                    added = True
                print(f"== {mid}: 抽样与已有记录一致（{len(per)} 项）"
                      f"{'，补了 scale / block' if added else ''}")
            continue

        q["per_subsystem"] = per
        if per_scale:
            q["per_subsystem_scale"] = per_scale
        if per_block:
            q["per_subsystem_block"] = per_block
        q["subsystem_method"] = meas.get("method", "")
        ev = doc.setdefault("evidence", {})
        ev["precision.quantization.per_subsystem"] = "derived"
        if per_scale:
            ev["precision.quantization.per_subsystem_scale"] = "derived"
        if per_block:
            ev["precision.quantization.per_subsystem_block"] = "derived"
        ev["precision.quantization.subsystem_method"] = "derived"

        head = "".join(l for l in open(path) if l.startswith("#"))
        with open(path, "w") as f:
            f.write(head)
            yaml.safe_dump(doc, f, allow_unicode=True, sort_keys=False, width=96)
        print(f"-> {mid}  ({len(per)} 个子系统)")

    return 1 if conflicts else 0


if __name__ == "__main__":
    sys.exit(main())
