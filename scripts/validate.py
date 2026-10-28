#!/usr/bin/env python3
"""校验：schema 合规、证据覆盖、防泄漏。

三道检查，任一失败就非零退出，供 CI 直接用：

  1. schema   每份 YAML 按 data/schema/*.json 校验
  2. 证据     每个手写叶子字段都要能在 evidence 里找到归属（支持前缀继承）；
              derived 的字段不允许手写；judgement 的字段列出来给人看
  3. 防泄漏   公开仓里不允许出现 visibility: internal 或 source.type: self

第 3 条是公开站与私有数据仓分离的最后一道闸。脱敏做不干净，所以这里不做
字段级脱敏，而是整条记录拒绝。

用法:
  python3 scripts/validate.py                    # 公开模式，含防泄漏
  python3 scripts/validate.py --allow-internal   # 内部 overlay 构建时放行
  python3 scripts/validate.py --list-judgements  # 列出所有判断类字段，供 Gate C
"""
import argparse
import glob
import json
import os
import sys

# evidence 里不需要逐个声明的结构性键
SKIP_KEYS = {"evidence", "sources", "visibility"}
# 这些字段由脚本生成，YAML 里不该出现
GENERATED_ONLY = {"derived"}


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import yamlio  # noqa: E402


def load_yaml(path):
    return yamlio.load(path)


def leaf_paths(obj, prefix=""):
    """枚举所有叶子路径。列表不展开到下标，整个列表算一个叶子归属点。"""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if not prefix and k in SKIP_KEYS:
                continue
            p = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                yield from leaf_paths(v, p)
            else:
                yield p
    else:
        yield prefix


def evidence_for(path, ev):
    """前缀继承：arch 覆盖 arch.attention.type，更深的路径优先。"""
    best, best_len = None, -1
    for k, v in ev.items():
        if path == k or path.startswith(k + "."):
            if len(k) > best_len:
                best, best_len = v, len(k)
    return best


def check_schema(doc, schema, path, errors):
    try:
        import jsonschema
    except ImportError:
        errors.append(f"{path}: 缺 jsonschema，装一下：pip install -r requirements.txt")
        return
    v = jsonschema.Draft202012Validator(schema)
    for e in sorted(v.iter_errors(doc), key=lambda e: list(e.path)):
        loc = ".".join(str(x) for x in e.path) or "(根)"
        errors.append(f"{path}: schema 不合规 @ {loc}: {e.message}")


def check_evidence(doc, path, errors, warnings, judgements):
    ev = doc.get("evidence")
    if not isinstance(ev, dict):
        errors.append(f"{path}: 缺 evidence 块")
        return
    for leaf in leaf_paths(doc):
        root = leaf.split(".")[0]
        if root in GENERATED_ONLY:
            errors.append(f"{path}: {leaf} 属于推导值，不该手写在 YAML 里"
                          f"（它应由 scripts/derive.py 生成到 data/derived/）")
            continue
        tag = evidence_for(leaf, ev)
        if tag is None:
            errors.append(f"{path}: {leaf} 没有证据归属，"
                          f"在 evidence 里加一条（可用前缀，如 {root}: official）")
        elif tag == "judgement":
            judgements.append(f"{path}: {leaf}")
        elif tag == "unknown":
            warnings.append(f"{path}: {leaf} 口径为 unknown，记得补来源或移到 open_questions")
    # 反向检查：evidence 里声明了但文档里没有的路径，通常是改名后忘了同步
    actual = set(leaf_paths(doc))
    for k in ev:
        if not any(a == k or a.startswith(k + ".") for a in actual):
            warnings.append(f"{path}: evidence 里的 '{k}' 在文档里找不到对应字段")


def check_leak(doc, path, errors, allow_internal):
    if allow_internal:
        return
    hits = []

    def walk(o, p=""):
        if isinstance(o, dict):
            # ops_table 是指针：它声明「被指向的表是内部的」，指针本身可公开
            if o.get("visibility") == "internal" and not p.endswith("ops_table"):
                hits.append(f"{p or '(根)'}: visibility=internal")
            src = o.get("source")
            if isinstance(src, dict) and src.get("type") == "self":
                hits.append(f"{p}: source.type=self（自测数据）")
            if isinstance(o.get("basis"), str) and o["basis"] == "measured":
                hits.append(f"{p}: basis=measured（实测数据）")
            for k, v in o.items():
                walk(v, f"{p}.{k}" if p else k)
        elif isinstance(o, list):
            for i, v in enumerate(o):
                walk(v, f"{p}[{i}]")

    walk(doc)
    for h in hits:
        errors.append(f"{path}: 内部数据不能进公开仓 — {h}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--allow-internal", action="store_true",
                    help="内部 overlay 构建时放行 internal 记录")
    ap.add_argument("--list-judgements", action="store_true",
                    help="列出所有判断类字段。Gate C 看这个")
    args = ap.parse_args()

    schema_dir = os.path.join(args.data_dir, "schema")
    model_schema = json.load(open(os.path.join(schema_dir, "model.schema.json")))
    ops_schema_path = os.path.join(schema_dir, "ops-table.schema.json")
    ops_schema = json.load(open(ops_schema_path)) if os.path.exists(ops_schema_path) else None

    errors, warnings, judgements = [], [], []
    models = sorted(glob.glob(os.path.join(args.data_dir, "models", "**", "*.yaml"),
                              recursive=True))
    for p in models:
        doc = load_yaml(p)
        if not isinstance(doc, dict):
            errors.append(f"{p}: 不是一个 YAML 映射")
            continue
        check_schema(doc, model_schema, p, errors)
        check_evidence(doc, p, errors, warnings, judgements)
        check_leak(doc, p, errors, args.allow_internal)

    ops = sorted(glob.glob(os.path.join(args.data_dir, "ops", "**", "*.yaml"), recursive=True))
    for p in ops:
        doc = load_yaml(p)
        if ops_schema:
            check_schema(doc, ops_schema, p, errors)
        check_leak(doc, p, errors, args.allow_internal)

    print(f"校验 {len(models)} 份模型记录, {len(ops)} 份算子表"
          f"{' (放行 internal)' if args.allow_internal else ''}")
    for w in warnings:
        print(f"  warn  {w}")
    for e in errors:
        print(f"  ERROR {e}")

    if args.list_judgements:
        print(f"\n--- 判断类字段 {len(judgements)} 处（Gate C 要人看过）")
        for j in judgements:
            print(f"  {j}")
    else:
        print(f"\n{len(judgements)} 处判断类字段（--list-judgements 看明细）")

    print(f"\n{len(errors)} 个错误, {len(warnings)} 个告警")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
