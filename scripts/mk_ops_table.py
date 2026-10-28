#!/usr/bin/env python3
"""渲染算子表：YAML -> xlsx。

列定义沿用既有家规的 21 列，不另造。三个 sheet 对应 schema 的 rows / params /
groups，有 accounting 时再加一张带宽账。

归属决定配色，语义是「要不要干活」：
  existing        灰  已有算子承接，不排期
  generic         蓝  任何同类模型都要，不该占本模型优先级
  model-specific  绿  只有本模型的特殊结构才需要，真正要做的活

用法:
  python3 scripts/mk_ops_table.py data-private/ops/<id>/table.yaml -o out.xlsx
  python3 scripts/mk_ops_table.py tests/fixtures/ops-table.sample.yaml   # 验流程
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import yamlio  # noqa: E402

COLUMNS = [
    ("module", "模块", 18),
    ("op", "算子列表", 30),
    ("kernel", "npu kernel", 34),
    ("op_type", "算子类型", 20),
    ("input_shape", "inputshape", 34),
    ("output_shape", "output shape", 24),
    ("dim", "dim", 10),
    ("shape_range", "shape范围", 34),
    ("input_dtype", "input dtype", 20),
    ("output_dtype", "output dtype", 16),
    ("input_format", "input format", 14),
    ("output_format", "output format", 14),
    ("dtype_next1", "dtype next1", 18),
    ("dtype_next2", "dtype next2", 18),
    ("group", "组号", 8),
    ("ownership", "归属", 14),
    ("_verdict", "核查结论", 12),
    ("_divergence", "实现分叉", 40),
    ("_impl", "实现侧证据", 30),
    ("_ref_impl", "参考实现证据", 30),
    ("note", "说明", 80),
]

FILL = {
    "existing": "D9D9D9",        # 灰：不排期
    "generic": "DCE6F1",         # 蓝：通用
    "model-specific": "D7E4BD",  # 绿：真正要做的
}
OWNERSHIP_CN = {
    "existing": "已有·无需新增",
    "generic": "通用",
    "model-specific": "本模型专属",
}
STATUS_CN = {
    "have": "已有", "extend": "扩展已有", "new": "要新写", "code-only": "纯代码改动",
}


def anchor_str(a):
    if not a:
        return ""
    s = a.get("file", "")
    if a.get("repo"):
        s = f"{a['repo']}:{s}"
    anc = (a.get("anchor") or "").strip().splitlines()
    return f"{s}\n@ {anc[0][:60]}" if anc else s


def flatten(row):
    ev = row.get("evidence") or {}
    out = dict(row)
    out["_verdict"] = ev.get("verdict", "")
    out["_divergence"] = ev.get("divergence", "")
    out["_impl"] = anchor_str(ev.get("impl"))
    out["_ref_impl"] = anchor_str(ev.get("ref_impl"))
    out["ownership"] = OWNERSHIP_CN.get(row.get("ownership", ""), row.get("ownership", ""))
    # 核查留痕并进说明列，不覆盖原值
    rv = row.get("review") or []
    if rv:
        marks = "\n".join(f"【核查 {r['id']}·{r.get('resolution', 'open')}】{r['text']}" for r in rv)
        out["note"] = (out.get("note") or "") + ("\n" if out.get("note") else "") + marks
    return out


def build(doc, path):
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    head_font = Font(bold=True)
    head_fill = PatternFill("solid", fgColor="BFBFBF")
    wrap = Alignment(vertical="top", wrap_text=True)

    wb = Workbook()

    # ---- 算子表
    ws = wb.active
    ws.title = "算子表"
    for c, (_, label, width) in enumerate(COLUMNS, 1):
        cell = ws.cell(1, c, label)
        cell.font, cell.fill, cell.alignment = head_font, head_fill, wrap
        ws.column_dimensions[get_column_letter(c)].width = width
    ws.freeze_panes = "A2"

    rows = doc.get("rows") or []
    for r, raw in enumerate(rows, 2):
        row = flatten(raw)
        fill = FILL.get(raw.get("ownership"))
        for c, (key, _, _) in enumerate(COLUMNS, 1):
            cell = ws.cell(r, c, row.get(key, ""))
            cell.alignment = wrap
            if fill:
                cell.fill = PatternFill("solid", fgColor=fill)

    # ---- 模型参数
    ws2 = wb.create_sheet("模型参数")
    for c, (label, width) in enumerate([("符号 / 条目", 34), ("含义与取值（★ = 容易静默出错）", 120)], 1):
        cell = ws2.cell(1, c, label)
        cell.font, cell.fill, cell.alignment = head_font, head_fill, wrap
        ws2.column_dimensions[get_column_letter(c)].width = width
    ws2.freeze_panes = "A2"
    r = 2
    last_section = None
    for p in doc.get("params") or []:
        if p.get("section") and p["section"] != last_section:
            last_section = p["section"]
            cell = ws2.cell(r, 1, f"—— {last_section} ——")
            cell.font = head_font
            r += 1
        key = ("★ " if p.get("pitfall") else "") + p["key"]
        ws2.cell(r, 1, key).alignment = wrap
        ws2.cell(r, 2, str(p["value"])).alignment = wrap
        r += 1

    # ---- 融合分组
    ws3 = wb.create_sheet("融合分组")
    cols3 = [("组号", 8), ("归属", 16), ("状态", 14), ("融合算子（已有 / 建议）", 46),
             ("收益", 8), ("收益 / 价值说明", 60), ("行数", 7), ("包含算子", 50), ("说明", 90)]
    for c, (label, width) in enumerate(cols3, 1):
        cell = ws3.cell(1, c, label)
        cell.font, cell.fill, cell.alignment = head_font, head_fill, wrap
        ws3.column_dimensions[get_column_letter(c)].width = width
    ws3.freeze_panes = "A2"
    for r, g in enumerate(doc.get("groups") or [], 2):
        members = g.get("members") or []
        vals = [g["id"], OWNERSHIP_CN.get(g["ownership"], g["ownership"]),
                STATUS_CN.get(g.get("status", ""), g.get("status", "")),
                g.get("fused_op", ""), g.get("benefit", ""), g.get("benefit_note", ""),
                len(members), " → ".join(members), g.get("note", "")]
        fill = FILL.get(g.get("ownership"))
        for c, v in enumerate(vals, 1):
            cell = ws3.cell(r, c, v)
            cell.alignment = wrap
            if fill:
                cell.fill = PatternFill("solid", fgColor=fill)

    # 图例：配色的语义比配色本身重要，写在表里省得每次解释
    r = (len(doc.get("groups") or [])) + 3
    ws3.cell(r, 1, "配色 = 要不要干活").font = head_font
    for i, (k, cn) in enumerate(OWNERSHIP_CN.items(), 1):
        n = sum(1 for g in (doc.get("groups") or []) if g.get("ownership") == k)
        rowsn = sum(1 for x in rows if x.get("ownership") == k)
        cell = ws3.cell(r + i, 1, cn)
        cell.fill = PatternFill("solid", fgColor=FILL[k])
        ws3.cell(r + i, 2, f"{n} 组 / {rowsn} 行")

    # ---- 带宽账
    acc = doc.get("accounting") or []
    if acc:
        ws4 = wb.create_sheet("带宽账")
        for c, (label, width) in enumerate(
                [("条目", 34), ("取值", 22), ("依据", 12), ("与官方口径差异", 60), ("可修之处", 50)], 1):
            cell = ws4.cell(1, c, label)
            cell.font, cell.fill, cell.alignment = head_font, head_fill, wrap
            ws4.column_dimensions[get_column_letter(c)].width = width
        for r, a in enumerate(acc, 2):
            for c, v in enumerate([a["item"], str(a["value"]), a["basis"],
                                   a.get("vs_official", ""), a.get("fixable", "")], 1):
                ws4.cell(r, c, v).alignment = wrap

    wb.save(path)


def check(doc):
    """渲染前的自查：分组引用的算子必须存在，组内归属必须一致。"""
    errs = []
    ops = {r.get("op") for r in doc.get("rows") or []}
    dup = len(doc.get("rows") or []) - len(ops)
    if dup > 0:
        errs.append(f"有 {dup} 个重复的算子名，分组引用会指向错的行")
    for g in doc.get("groups") or []:
        for m in g.get("members") or []:
            if m not in ops:
                errs.append(f"分组 {g['id']} 引用了不存在的算子 '{m}'")
        owners = {r.get("ownership") for r in (doc.get("rows") or []) if r.get("op") in (g.get("members") or [])}
        if len(owners) > 1:
            errs.append(f"分组 {g['id']} 内归属不一致：{owners}，配色会自相矛盾")
    grouped = {m for g in doc.get("groups") or [] for m in g.get("members") or []}
    orphan = ops - grouped
    if orphan and doc.get("groups"):
        errs.append(f"{len(orphan)} 个算子没有归入任何分组：{sorted(orphan)[:5]}")
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yaml_path")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--force", action="store_true", help="自查有问题也照样渲染")
    args = ap.parse_args()

    doc = yamlio.load(args.yaml_path)
    errs = check(doc)
    for e in errs:
        print(f"  ERROR {e}", file=sys.stderr)
    if errs and not args.force:
        sys.exit("! 自查未通过，修好再渲染（--force 可跳过）")

    out = args.out or os.path.join(
        os.path.dirname(args.yaml_path) or ".",
        f"{doc.get('model', 'ops')}_decode_算子表.xlsx")
    build(doc, out)
    rows = doc.get("rows") or []
    by = {}
    for r in rows:
        by[r.get("ownership", "?")] = by.get(r.get("ownership", "?"), 0) + 1
    print(f"-> {out}")
    print(f"   {len(rows)} 行, {len(doc.get('groups') or [])} 组, "
          f"{len(doc.get('params') or [])} 条参数")
    print("   归属分布: " + ", ".join(f"{OWNERSHIP_CN.get(k, k)} {v}" for k, v in sorted(by.items())))
    base = doc.get("baseline") or {}
    if base.get("config_origin") == "assembled":
        print(f"   ! 基线是拼装配置: {base.get('config_caveat', '未说明风险')}")


if __name__ == "__main__":
    main()
