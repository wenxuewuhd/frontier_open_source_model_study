#!/usr/bin/env python3
"""校验：schema 合规、证据覆盖、防泄漏、声明与实测对账。

四道检查，前三道任一失败就非零退出，供 CI 直接用：

  1. schema   每份 YAML 按 data/schema/*.json 校验
  2. 证据     每个手写叶子字段都要能在 evidence 里找到归属（支持前缀继承）；
              derived 的字段不允许手写；judgement 的字段列出来给人看
  3. 防泄漏   公开仓里不允许出现 visibility: internal 或 source.type: self
  4. 声明对账 precision.weights_release 里声明的每种精度，都要在实测清单里出现（告警）
  5. 缺值三态 每一个会渲染成空格子的列，都要能说出自己是「没查」「不适用」
              还是「查了但没有」（data/schema/columns.json 的声明；第三态是信息，
              前两态是缺口，渲染成同一个 `—` 就等于把信息和缺口混为一谈）

第 3 条是公开站与私有数据仓分离的最后一道闸。脱敏做不干净，所以这里不做
字段级脱敏，而是整条记录拒绝。

用法:
  python3 scripts/validate.py                    # 公开模式，含防泄漏
  python3 scripts/validate.py --allow-internal   # 内部 overlay 构建时放行
  python3 scripts/validate.py --list-judgements  # 列出所有判断类字段，供 Gate C
  python3 scripts/validate.py --list-gaps        # 列出判为「未录」的格子：调研进度
"""
import argparse
import glob
import json
import os
import sys

# evidence 里不需要逐个声明的结构性键
# 元字段：描述记录本身而非模型事实，不需要证据归属
SKIP_KEYS = {"evidence", "sources", "visibility", "reviewed", "review",
             # 核查元信息，不是关于模型的事实 —— 给它贴证据类别没有意义
             "reviewed_elsewhere"}
# 这些字段由脚本生成，YAML 里不该出现
GENERATED_ONLY = {"derived"}


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import yamlio  # noqa: E402


def load_yaml(path):
    return yamlio.load(path)


def check_release_vs_measured(doc, path, derived_dir, warnings):
    """precision.weights_release 声明的精度，实测清单里有没有。

    参数量那边有 HF 的 safetensors tally 当外部锚，缓存那边有厂商公布的每 token 字节。
    这一道是第三个方向：**拿厂商的发布声明去查本仓的实测**。前两个锚都是「外面的数
    校验本仓算的」，这一个是「本仓实测校验外面声明的」，合起来才闭合。

    比的是 data/derived/<id>.json 的 weight_bytes_by_dtype，不是 facts/ 下的清单 ——
    facts/ 在 gitignore 里，CI 上没有；而 derived 是提交过的。

    **容器情况自己豁免自己，这里不写白名单。** derive.py 在打包成立时已经把容器换成了
    逻辑精度（packing_factor > 1 时 label 取 logical），所以 FP4 装 INT8、INT4 装 INT32、
    MXFP4 装 UINT8 这些在 by_dtype 里本来就显示成 FP4 / INT4 / MXFP4，对得上。
    写死白名单的话，下次出现新的打包格式就会被当成真不一致；而一个真的声明错了的，
    反倒可能被某条白名单捞走。判据交给 packing_factor 是唯一不会随版本腐烂的办法。

    报 warning 不报 error：声明里出现本仓没取过数的那个仓的精度，不一定是写错 ——
    厂商常把量化变体发在独立仓。但两种情况在字段上长得一模一样，所以必须提示。
    weights_release 的口径（要不要带「本仓取没取过数」的标记）定下来之后，
    这一条可以收紧成 error。
    """
    wr = ((doc.get("precision") or {}).get("weights_release")) or []
    mid = doc.get("id")
    if not wr or not mid:
        return
    dp = os.path.join(derived_dir, f"{mid}.json")
    if not os.path.exists(dp):
        return
    try:
        fs = (json.load(open(dp)) or {}).get("from_shapes") or {}
    except (ValueError, OSError):
        return
    by = fs.get("weight_bytes_by_dtype")
    if not by:
        return  # 没有逐 dtype 统计（抽样快照等），这一项无从对账
    measured = {str(k).upper() for k in by}
    missing = [str(x) for x in wr if str(x).upper() not in measured]
    if missing:
        warnings.append(
            f"{path}: precision.weights_release 声明了 {missing}，但实测清单里没有"
            f"（实测 {sorted(measured)}）。要么这些变体在本仓没取过数的独立仓里 —— "
            f"那就在记录里写明是哪个仓、本记录的实测口径只覆盖主仓；"
            f"要么声明本身不对。两种在字段上长得一样，页面会并排显示"
            f"「声明发了 X」和「实测全是 Y」，读者会当成数据错")


def check_absence_contract(errors, warnings):
    """缺值的三态：每一个会渲染成空格子的列，都要能说出自己是哪一种空。

    四道闸全绿的那一周被问出三个问题，三个都不是数值错，全在呈现层。其中一个是
    **`—` 一个符号背着三种含义**：没查 / 不适用 / 查了但没有。45 份记录的
    `precision.kv_cache` 是空的，真相是「厂商没公布，字节按权重 dtype 兜底」——
    那是一条结论，不是缺口，可它和「还没查」渲染成同一个符号。

    最省力的修法是在模板里把 `—` 换成一句话。那只是把问题挪了个地方，下一张新表
    会重犯。所以区分放在 data/schema/columns.json：每一列声明空格子的默认含义，
    加若干条按顺序试的结构判据。这道检查管那份声明本身是完整的 ——
    渲染器照它渲染就无从含糊，而漏声明一列会在这里被拦住，不是等到页面上。

    只查**声明**，不查渲染。渲染侧（站点怎么画这三态）有它自己的归属，
    在这里顺带查会变成两道闸管同一件事，而两道闸迟早会各自漂。
    """
    try:
        from lib import columns
    except ImportError as e:  # pragma: no cover
        errors.append(f"缺 scripts/lib/columns.py（{e}）—— 缺值三态没有定义入口")
        return

    states = set(columns.absence_states())
    if states != {"unsurveyed", "not_applicable", "confirmed_absent"}:
        errors.append(f"columns.json 的 absence_states 不是约定的三态：{sorted(states)}")

    items = list(columns.columns().items()) + [("$subsystem", columns.subsystem_template())]
    for key, col in items:
        a = col.get("absent")
        if not a:
            errors.append(f"columns.json: 列 '{key}' 没有 absent 声明 —— "
                          "它的空格子无从判别是「没查」「不适用」还是「查了但没有」")
            continue
        rules = a.get("rules") or []
        for st, where in ([(a.get("state"), "默认态")]
                          + [(r.get("state"), f"规则 {n}") for n, r in enumerate(rules)]):
            if st not in states:
                errors.append(f"columns.json: 列 '{key}' 的{where} '{st}' 不是三态之一")
        # 「不适用」与「查了但没有」是在替记录下结论，必须说出凭什么；
        # 「未录」是进度陈述，三态表里的通用说明就够了
        if a.get("state") in ("not_applicable", "confirmed_absent") and not a.get("why"):
            errors.append(f"columns.json: 列 '{key}' 默认判为 '{a['state']}' 却没写 why —— "
                          "这是在替记录下结论，要说出凭什么")
        for n, r in enumerate(rules):
            if r.get("state") in ("not_applicable", "confirmed_absent") and not r.get("why"):
                errors.append(f"columns.json: 列 '{key}' 规则 {n} 判为 '{r['state']}' 却没写 why")
            t = r.get("test") or {}
            if t:
                if t.get("op") not in columns.OPS:
                    errors.append(f"columns.json: 列 '{key}' 规则 {n} 的 op "
                                  f"'{t.get('op')}' 没有实现（已实现：{sorted(columns.OPS)}）")
                if not t.get("path"):
                    errors.append(f"columns.json: 列 '{key}' 规则 {n} 的 test 没有 path")
        # 会被读错成另一种量的列必须带单位。元素/token 与字节/token 两张表并排时，
        # 列名离数字十几个字符 —— 1,600（元素）被读成了字节，和另一张表的 890 B 对不上
        if col.get("quantity") and not col.get("unit"):
            errors.append(f"columns.json: 列 '{key}' 声明了 quantity（一种会被读错成"
                          "别的量的物理量）却没有 unit")
        # 实测值要说得出测自哪个仓，否则厂商另发的精度变体在页面上不可见
        if col.get("scope") == "measured" and not col.get("scope_from"):
            errors.append(f"columns.json: 列 '{key}' 声明 scope=measured 却没有 scope_from "
                          "—— 这一格说不出测自哪个仓")


def check_absence_resolves(doc, path, derived_dir, errors, warnings, gaps):
    """这份记录的每一个空格子，按声明判出来是哪一种空。

    判不出来就是 columns.json 漏了一列（上面那道已经拦住）。这里管的是另一半：
    判出来的结果要能复现，而且「未录」要攒成一份清单 —— 它是调研进度，
    该有人看见还剩哪些没查，而不是和「不适用」一起沉在页面的 `—` 里。
    """
    try:
        from lib import columns
    except ImportError:
        return
    mid = doc.get("id")
    if not mid:
        return
    dp = os.path.join(derived_dir, f"{mid}.json")
    derived = None
    if os.path.exists(dp):
        try:
            derived = json.load(open(dp))
        except (ValueError, OSError):
            derived = None

    q = ((doc.get("precision") or {}).get("quantization") or {})
    ps = q.get("per_subsystem") or {}
    subs = [s["key"] for s in json.load(
        open(os.path.join(os.path.dirname(derived_dir), "schema", "subsystems.json")))["subsystems"]]

    for key in list(columns.columns()) + [f"sub_{s}" for s in subs if s not in ps]:
        try:
            a = columns.absence(key, doc, derived)
        except KeyError as e:
            errors.append(f"{path}: 列 '{key}' 的缺值判据跑不起来（{e}）")
            continue
        if a["state"] == "unsurveyed" and key.startswith("sub_"):
            gaps.append((mid, key[4:]))


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


MISSING = object()


def resolve_path(doc, dotted):
    """按点分路径取值，支持 a.b[0].c 这种下标写法。路径不存在返回 MISSING。"""
    cur = doc
    for part in dotted.split("."):
        name, _, idx = part.partition("[")
        if name:
            if not isinstance(cur, dict) or name not in cur:
                return MISSING
            cur = cur[name]
        while idx:
            n, _, rest = idx.partition("]")
            try:
                cur = cur[int(n)]
            except (ValueError, TypeError, IndexError, KeyError):
                return MISSING
            idx = rest.partition("[")[2]
    return cur


def check_review(doc, path, errors, warnings):
    """核查块的自洽性。

    三条规则，都是为了让问题藏不住：
      1. 没有 reviewed 块 -> 告警，以便一眼看出哪些记录还没被核查过
      2. field 必须指向记录里真实存在的路径，否则核查留痕会指向空气
      3. resolution=open 的发现必须同时写进 open_questions —— 否则未解决的问题
         只存在于核查块里，读记录的人看不到
    """
    review = doc.get("review") or []
    if not doc.get("reviewed"):
        if doc.get("reviewed_elsewhere"):
            # 指向别处的核查：这不等于「整份记录已核查」，只是那几个字段核过了。
            # 仍然告警，但说清是部分覆盖 —— 指针本身由 check_review_pointers 核对
            el = doc["reviewed_elsewhere"]
            # 只报指到哪里，不把 scope 整段倒出来 —— 告警是索引，不是正文
            warnings.append(
                f"{path}: 没有自己的 reviewed 块；"
                f"{'/'.join(el.get('findings') or [])} 的核查在 {el.get('record')}，"
                "覆盖范围见 reviewed_elsewhere.scope，其余字段仍未核查")
        else:
            warnings.append(f"{path}: 还没有 reviewed 块，这份记录未经核查")
        if review:
            errors.append(f"{path}: 有 review 发现却没有 reviewed 元信息，无法判断核查范围")
        return

    seen = set()
    open_q = " ".join(doc.get("open_questions") or [])
    for f in review:
        fid = f.get("id", "?")
        if fid in seen:
            errors.append(f"{path}: 核查编号 {fid} 重复")
        seen.add(fid)

        field = f.get("field", "")
        # 已删除的字段无从指向，所以 severity=missing 与被删掉的条目允许指向不存在的路径
        if field and field != "-" and resolve_path(doc, field) is MISSING:
            # 指向「本来就缺失」的字段是合法的：漏字段、已改掉的字段、
            # 以及未解决的缺失项，都无从指向一个现存路径
            if (f.get("severity") != "missing"
                    and f.get("resolution") not in ("corrected", "open")):
                errors.append(f"{path}: 核查 {fid} 指向的字段 '{field}' 在记录里不存在")

        if f.get("resolution") == "open":
            key = field.split("[")[0].split(".")[-1]
            if key and key not in open_q and field not in open_q:
                warnings.append(
                    f"{path}: 核查 {fid}（{field}）未解决，但 open_questions 里没有对应条目 —— "
                    f"未解决的问题不应该只存在于核查块里"
                )
        if f.get("resolution") == "corrected" and not f.get("changed_to"):
            warnings.append(f"{path}: 核查 {fid} 标为已改，但没写改成了什么")


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


def check_review_pointers(docs, errors, warnings):
    """跨记录核对「核查在别处」的指针。

    指针不核对就会悄悄烂掉：对面改了编号、删了条目，或者被指向的记录自己
    从没被核查过 —— 而这份记录上写着「已在别处核过」。那比老实显示未核查更糟，
    因为它让人不再去看。单份记录校验看不到这件事，只能在全部加载完之后比。
    """
    by_id = {doc["id"]: (p, doc) for p, doc in docs.items() if doc.get("id")}
    for path, doc in docs.items():
        el = doc.get("reviewed_elsewhere")
        if not el:
            continue
        tgt = el.get("record")
        if tgt == doc.get("id"):
            errors.append(f"{path}: reviewed_elsewhere 指向自己")
            continue
        if tgt not in by_id:
            errors.append(f"{path}: reviewed_elsewhere 指向的记录 {tgt} 不存在")
            continue
        _, tdoc = by_id[tgt]
        if not tdoc.get("reviewed"):
            errors.append(f"{path}: reviewed_elsewhere 指向 {tgt}，"
                          "但那份记录自己就没有 reviewed 块，指针落在空处")
        have = {f.get("id") for f in (tdoc.get("review") or [])}
        missing = [x for x in (el.get("findings") or []) if x not in have]
        if missing:
            errors.append(f"{path}: reviewed_elsewhere 引用的 {'/'.join(missing)} "
                          f"在 {tgt} 的 review 列表里不存在")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--allow-internal", action="store_true",
                    help="内部 overlay 构建时放行 internal 记录")
    ap.add_argument("--list-judgements", action="store_true",
                    help="列出所有判断类字段。Gate C 看这个")
    ap.add_argument("--list-gaps", action="store_true",
                    help="列出判为「未录」的格子。那是调研进度，不是模型事实")
    args = ap.parse_args()

    schema_dir = os.path.join(args.data_dir, "schema")
    model_schema = json.load(open(os.path.join(schema_dir, "model.schema.json")))
    ops_schema_path = os.path.join(schema_dir, "ops-table.schema.json")
    ops_schema = json.load(open(ops_schema_path)) if os.path.exists(ops_schema_path) else None

    errors, warnings, judgements, gaps = [], [], [], []
    check_absence_contract(errors, warnings)
    models = sorted(glob.glob(os.path.join(args.data_dir, "models", "**", "*.yaml"),
                              recursive=True))
    loaded = {}
    for p in models:
        doc = load_yaml(p)
        if not isinstance(doc, dict):
            errors.append(f"{p}: 不是一个 YAML 映射")
            continue
        loaded[p] = doc
        check_schema(doc, model_schema, p, errors)
        check_evidence(doc, p, errors, warnings, judgements)
        check_review(doc, p, errors, warnings)
        check_leak(doc, p, errors, args.allow_internal)
        check_release_vs_measured(doc, p, os.path.join(args.data_dir, "derived"), warnings)
        check_absence_resolves(doc, p, os.path.join(args.data_dir, "derived"),
                               errors, warnings, gaps)
    check_review_pointers(loaded, errors, warnings)

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

    if gaps:
        by = {}
        for mid, k in gaps:
            by.setdefault(mid, []).append(k)
        if args.list_gaps:
            print(f"\n--- 判为「未录」的逐子系统精度：{len(by)} 份记录")
            for mid, ks in sorted(by.items()):
                print(f"  {mid}: {', '.join(ks)}")
        else:
            print(f"\n{len(by)} 份记录有判为「未录」的逐子系统精度"
                  "（--list-gaps 看明细）—— 调研进度，不是模型事实")

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
