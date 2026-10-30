#!/usr/bin/env python3
"""列的含义：读 data/schema/columns.json，回答「这个数带什么单位」「空格子是哪一态」。

站点侧要渲染这三态时读同一份文件、实现同一套 op —— 一份定义，两边不会漂移。
两侧的 test() 必须一致，否则同一个空格子在闸口里是「不适用」、在页面上是「未录」。

两个入口：

  column(key)       -> 这一列的定义（单位、实测口径来源、absent 规则）
  absence(key, ...) -> 空格子的三态判定（unsurveyed / not_applicable / confirmed_absent）

**bytes_basis 的取值表与标签不在这里。** 那一套归 scripts/kv_table.py 与站点侧；
在这里再写一份只会变成又一份对不上的拷贝 —— 取值表已经有过三份各自不全的拷贝了。

缺值一律按最弱的那一态（unsurveyed）兜底：说不出来就当没查过，不替记录把话说满。
"""
from __future__ import annotations

import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC_PATH = os.path.join(ROOT, "data", "schema", "columns.json")

_spec = None


def spec() -> dict:
    global _spec
    if _spec is None:
        with open(SPEC_PATH) as fh:
            _spec = json.load(fh)
    return _spec


def columns() -> dict:
    """列定义，去掉 $ 开头的注释键与模板键。"""
    return {k: v for k, v in spec()["columns"].items() if not k.startswith("$")}


def subsystem_template() -> dict:
    return spec()["columns"]["$subsystem"]


def column(key: str) -> dict | None:
    """按 key 取列定义；sub_* 落到逐子系统模板上。"""
    cols = spec()["columns"]
    if key in cols and not key.startswith("$"):
        return cols[key]
    if key.startswith("sub_"):
        return cols["$subsystem"]
    return None


def absence_states() -> dict:
    return {k: v for k, v in spec()["absence_states"].items() if not k.startswith("$")}


# ------------------------------------------------------- 空格子的三态判定

def _dig(obj, dotted: str):
    cur = obj
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def test(rule: dict, rec: dict, derived: dict | None) -> bool:
    """判定一条 absent 规则。op 的集合是固定的小词表，两种语言各实现一遍。

    新加 op 要同时改 columns.ts，闸口会比对两边的 op 列表。
    """
    t = rule.get("test")
    if not t:
        return True  # 没有 test 的规则无条件命中（用于把默认态写在规则位上）
    src = (derived or {}) if t.get("src") == "derived" else rec
    val = _dig(src, t["path"])
    op = t["op"]
    if op == "truthy":
        return bool(val)
    if op == "absent":
        return val is None
    if op == "eq":
        return val in t.get("values", [])
    if op == "in":
        return isinstance(val, list) and any(v in val for v in t.get("values", []))
    if op == "not_in":
        return val is not None and val not in t.get("values", [])
    if op == "matches":
        return bool(isinstance(val, str) and re.search(t["pattern"], val))
    raise KeyError(f"columns.json 用了未实现的 op: {op!r}")


OPS = {"truthy", "absent", "eq", "in", "not_in", "matches"}


def absence(key: str, rec: dict, derived: dict | None = None) -> dict:
    """这个 key 在这份记录上为空时，是三态里的哪一种。

    返回 {state, label, why, is_gap}。说不出来就返回 unsurveyed —— 替记录把话说满
    （比如默认报「不适用」）会把调研缺口渲染成模型事实，那正是要修的那个错。
    """
    col = column(key) or {}
    a = col.get("absent") or {}
    state, why = a.get("state", "unsurveyed"), a.get("why")
    for rule in a.get("rules") or []:
        if test(rule, rec, derived):
            state, why = rule["state"], rule.get("why", why)
            break
    meta = absence_states().get(state) or {}
    return {"state": state, "label": meta.get("label", state),
            "why": why or meta.get("meaning", ""), "is_gap": bool(meta.get("is_gap"))}
