#!/usr/bin/env python3
"""校验：schema 合规、证据覆盖、防泄漏、声明与实测对账。

五道检查，任一失败就非零退出，供 CI 直接用：

  1. schema   每份 YAML 按 data/schema/*.json 校验
  2. 证据     每个手写叶子字段都要能在 evidence 里找到归属（支持前缀继承）；
              derived 的字段不允许手写；judgement 的字段列出来给人看
  3. 防泄漏   公开仓里不允许出现 visibility: internal 或 source.type: self
  4. 声明对账 precision.weights_release 里声明的每种精度，都要在实测清单里出现，
              或者在 weights_release_scope 里交代清楚它在哪个仓 / 装在哪种容器里
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
import re
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


def check_release_vs_measured(doc, path, derived_dir, errors):
    """precision.weights_release 声明的精度，实测清单里有没有。

    参数量那边有 HF 的 safetensors tally 当外部锚，缓存那边有厂商公布的每 token 字节。
    这一道是第三个方向：**拿厂商的发布声明去查本仓的实测**。前两个锚都是「外面的数
    校验本仓算的」，这一个是「本仓实测校验外面声明的」，合起来才闭合。

    比的是 data/derived/<id>.json 的 weight_bytes_by_dtype，不是 facts/ 下的清单 ——
    facts/ 在 gitignore 里，CI 上没有；而 derived 是提交过的。

    **这道闸锚在内侧，所以只守得住一半**（model-audit skill 规则 11）：它查的是
    「记录声明了而本仓清单里没有」，查不出「厂商发了而记录压根没声明」——
    MiniMax-M3 漏掉的 MXFP8 变体就在这个盲区里，声明与实测完全自洽，这里全绿。
    要补那一半，锚得换到外侧（厂商发布清单 / HF 仓列表），是另一种取数。

    **容器情况自己豁免自己，这里不写白名单。** derive.py 在打包成立时已经把容器换成了
    逻辑精度（packing_factor > 1 时 label 取 logical），所以 FP4 装 INT8、INT4 装 INT32、
    MXFP4 装 UINT8 这些在 by_dtype 里本来就显示成 FP4 / INT4 / MXFP4，对得上。
    写死白名单的话，下次出现新的打包格式就会被当成真不一致；而一个真的声明错了的，
    反倒可能被某条白名单捞走。判据交给 packing_factor 是唯一不会随版本腐烂的办法。

    报 warning 不报 error：声明里出现本仓没取过数的那个仓的精度，不一定是写错 ——
    厂商常把量化变体发在独立仓。但两种情况在字段上长得一模一样，所以必须提示。

    口径已经定了：**列厂商发布过的全部变体，每一项带「本仓取没取过数」的标记**
    （precision.weights_release_scope）。40 份带 weights_release 的记录已经全部迁完，
    所以这一条**从告警收紧成错误**：没交代又对不上，现在没有任何正当理由 ——
    要么补 scope 说明在哪个仓（in_precision_table: false）或装在哪种容器里
    （container），要么声明本身不对，两种都得有人动手。

    **这道闸认的是 in_precision_table，不是 manifest_taken。** 两者 2026-10-30 之前
    是同一个 measured 布尔，而它同时背着「取没取过清单」与「进没进精度表」两件事。
    本函数只关心后者：它比的是本记录 derived 的 weight_bytes_by_dtype，所以能为一种
    精度的缺席作正当交代的，只有「它没进这张表」。前者是纯事实，与这道闸无关 ——
    MiniMax-M3 的 MXFP8 清单取过却没进表，在旧字段下只能填 false，字段于是在说反话。

    顺带查一条自相矛盾：in_precision_table 为真而 manifest_taken 为假 ——
    没取过清单不可能有实测值进表，两个键拆开之后这种组合写得出来，所以得挡。
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
    if not missing:
        return

    # weights_release_scope 一旦把某个变体交代清楚了，它就不再是「两种在字段上长得一样」
    # 的那种情况 —— 有标记说明它没进这张精度表（in_precision_table: false），或者说明是
    # 逻辑精度装在别的容器里（container），都算交代过，放行。没标记的那些照旧报。
    #
    # 这样 48 份可以一份一份迁移：每迁一份，它的告警当场消失，不必等全部迁完。
    # **全部迁完之后把下面的 warnings 改成 errors**，这一条就从提示变成闸 ——
    # 届时「没交代又对不上」没有任何正当理由。
    scope = ((doc.get("precision") or {}).get("weights_release_scope")) or {}
    unexplained = []
    for v in missing:
        s = scope.get(v) or scope.get(str(v).upper()) or {}
        if s.get("in_precision_table") is False or s.get("container"):
            continue
        unexplained.append(v)
    if not unexplained:
        return

    errors.append(
        f"{path}: precision.weights_release 声明了 {unexplained}，但实测清单里没有"
        f"（实测 {sorted(measured)}）。要么这些变体在本仓没取过数的独立仓里 —— "
        f"那就在 precision.weights_release_scope 里写明是哪个仓、"
        f"in_precision_table: false（清单取没取过是另一个键 manifest_taken）；"
        f"要么声明本身不对。两种在字段上长得一样，页面会并排显示"
        f"「声明发了 X」和「实测全是 Y」，读者会当成数据错")


def check_scope_coherence(doc, path, errors):
    """weights_release_scope 的两个布尔不能自相矛盾。

    in_precision_table 为真，说的是「这个变体的实测值进了本记录的逐子系统精度表」；
    而实测值只能来自清单。所以 manifest_taken 为假时它不可能为真。

    这条在拆字段（2026-10-30）之前写不出来 —— 那时只有一个 measured 布尔，
    没有第二个键可以和它冲突。拆开换来了表达力，也换来了这种新的错法，
    所以拆的同时就得把它挡住。
    """
    scope = ((doc.get("precision") or {}).get("weights_release_scope")) or {}
    for prec, s in scope.items():
        s = s or {}
        if s.get("in_precision_table") is True and s.get("manifest_taken") is False:
            errors.append(
                f"{path}: precision.weights_release_scope.{prec} 自相矛盾 —— "
                f"in_precision_table: true 说实测值进了逐子系统精度表，"
                f"而 manifest_taken: false 说本仓没取过这个仓的清单。"
                f"没有清单就没有实测值，两者不能同时成立")


def _root(data_dir):
    return os.path.dirname(os.path.abspath(data_dir)) or "."


def check_absence_contract(errors, warnings):
    """缺值的三态：每一个会渲染成空格子的列，都要能说出自己是哪一种空。

    四道闸全绿的那一周被问出三个问题，三个都不是数值错，全在呈现层。其中一个是
    **`—` 一个符号背着三种含义**：没查 / 不适用 / 查了但没有。46 份记录的
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


# 带中文的「标签」：引号里的中文字面量，或 JSX 里 > < 之间的中文文本
CJK_LABEL = re.compile(r"""['"][^'"]*[一-鿿][^'"]*['"]|>[^<>]*[一-鿿][^<>]*<""")
COMPARE = re.compile(r"[!=]==?")
BASIS_VALUES = ["theoretical_floor", "vendor_declared", "declared_dtype",
                "dtype_fallback", "mixed"]
LABEL_SCAN = [os.path.join("site", "src"), "scripts"]
# 读注册表的入口本身要写标签，不在检查范围里
# 标签表的家。basis 的中文标签**不在 columns.json 里** —— 缺值三态和 basis 语义
# 是两件事，混在一张注册表里会让「空格子是哪一种空」和「这个数按什么口径算」
# 共用一套键，所以 basis 那一节是有意从 columns.json 摘掉的。
# 它的家是 site/src/lib/data.ts 的 KV_BASIS_LABEL / KV_BASIS_NOTE（短标签与
# 长解释放在一起，免得出现「标签查表、解释写 ternary」的半吊子状态）。
LABEL_HOME = {os.path.join("scripts", "lib", "columns.py"),
              os.path.join("scripts", "validate.py"),
              os.path.join("site", "src", "lib", "columns.ts"),
              os.path.join("site", "src", "lib", "data.ts")}


def label_table_hit(line):
    """这一行是不是在把某个 basis 取值映射成中文？返回命中的取值名。

    只认手写标签表实际会长的两种样子，别的都放过：

      1. 当成对象键用        `theoretical_floor: '理论下限',`
      2. 拿它做比较再给中文  `basis === 'theoretical_floor' ? '理论下限' : '兜底'`
                             `{kv.bytes_basis === 'dtype_fallback' && <span>（精度兜底）</span>}`

    **按结构找，不按词找。** 「同一行里既有取值名又有中文」会把散文全部误伤：页面上的
    口径说明会写 `<code>theoretical_floor</code> 是查到落盘格式的理论下限`，注释会写
    「原先是 === theoretical_floor ? 理论下限 : …」。散文里怎么写这些概念是行文的事；
    这里要管的是**有没有第二份标签表**。

    反过来，按形状找（认 `const <NAME> = {`）又会漏掉写成三元或两分支 if 的那种 ——
    LayerMap 里那几处正是三元，而且只盖两档，vendor_declared 落进 else 被标成
    「记录没给落盘精度」，说反了。**不是漏标签，是标错。**

    真·basis 相关的行为要放过：`const floor = basis === 'theoretical_floor'`（没有中文）、
    `{basis === 'theoretical_floor' ? <Evidence tag="judgement"/> : ''}`（没有中文标签）
    都不是标签表，照常通过。
    """
    if not CJK_LABEL.search(line):
        return None
    for v in BASIS_VALUES:
        if re.search(r"""(^|[{,\s])['"]?%s['"]?\s*:\s*['"][^'"]*[一-鿿]""" % v, line):
            return v
        if re.search(r"""['"]%s['"]""" % v, line) and COMPARE.search(line):
            return v
    return None


# schema 里声明了、站点上没人读的字段数。**这是个棘轮，不是目标值。**
# 它不该变大：新加字段时要么同时加消费点，要么把这个数改大并在提交信息里说清为什么。
# 当前这 47 个基本是整个 generation 块 —— 视频线的记录里 VAE 规格、音频、
# 文本编码器、采样口径、尺寸约束、checkpoint 矩阵都录了，而页面上只渲染了
# 由它们推出来的负载，规格本身一格没有。
_UNREAD_FIELD_BUDGET = 45


def check_field_consumers(root, errors, warnings):
    """schema 里的字段，站点上得有人读。

    **一天之内撞上四次同一个形状**：字段/产物存在、没有消费点 ——
    `weights_release_scope` 一处不渲染、`bottlenecks.fix` 不走 Prose、
    `generation_workload` 整块没人读、`group_size` 加完没有列。
    四次里三次是五道闸全绿的状态下混过去的，靠人翻页面才发现。

    `check_rendered.py` 查的是「有没有没解开的标记」，不查「这个字段有没有消费点」——
    **闸全绿不等于读者看得见**（这条是 p1-kimi 提的）。

    判据粗：按叶字段名在 `site/src` 下 grep 标识符。会漏（同名标识符撞上就算读过），
    也会误报（经由 `Object.entries` 整块渲染的字段查不到名字）。但它抓的是
    「加了字段就以为完事」这个动作，宁可粗一点也要有。
    所以是**棘轮不是零**：只比对数量有没有变大，不逐条报。
    """
    import re as _re
    sp = os.path.join(root, "data", "schema", "model.schema.json")
    try:
        schema = json.load(open(sp, encoding="utf-8"))
    except (OSError, ValueError):
        return

    def leaves(node, path=""):
        out = []
        if not isinstance(node, dict):
            return out
        props = node.get("properties")
        if props:
            for k, v in props.items():
                pp = f"{path}.{k}" if path else k
                sub = leaves(v, pp)
                out += sub if sub else [(pp, k)]
        elif node.get("type") == "array" and isinstance(node.get("items"), dict):
            out += leaves(node["items"], path)
        return out

    seen = set()
    for dirpath, dirnames, filenames in os.walk(os.path.join(root, "site", "src")):
        dirnames[:] = [d for d in dirnames if d not in ("node_modules", "dist", ".astro")]
        for fn in filenames:
            if not fn.endswith((".ts", ".astro")):
                continue
            try:
                txt = open(os.path.join(dirpath, fn), encoding="utf-8").read()
            except OSError:
                continue
            seen.update(_re.findall(r"[A-Za-z_][A-Za-z0-9_]*", txt))

    watched = ("precision", "generation")
    unread = [pp for pp, k in leaves(schema)
              if pp.startswith(watched) and k not in seen]
    n = len(unread)
    if n > _UNREAD_FIELD_BUDGET:
        errors.append(
            # 不说「新增的是 X」—— 这里拿不到基线名单，只有数量，
            # 说得出口的只有「当前这批是什么」。编一个具体的、可能错的断言，
            # 正是本仓今天反复栽的那个形状
            f"schema 里有 {n} 个字段站点上没人读，超过棘轮 {_UNREAD_FIELD_BUDGET} 个。"
            f"当前没人读的（字母序前 6 个，不一定是你新加的那个）："
            f"{sorted(unread)[:6]}…\n"
            "    **加字段要同时加消费点** —— 否则值填进记录、闸口全绿、页面上看不见。"
            "一天之内这个形状撞过四次。确实不该渲染就把 _UNREAD_FIELD_BUDGET 调大并说明理由")
    elif n < _UNREAD_FIELD_BUDGET:
        warnings.append(
            f"站点没读的 schema 字段降到 {n} 个（棘轮是 {_UNREAD_FIELD_BUDGET}）。"
            "把 _UNREAD_FIELD_BUDGET 改成这个数，棘轮才收得紧")


def check_speedup_baseline(doc, path, errors):
    """报了倍数就必须报基线。

    CLAUDE.md 写着「**没有基准的倍数不能用**，最多进 open_question」——
    此前那条规矩只活在散文里，谁都可能漏。`performance[].speedup` 与
    `baseline.what` 做成一对字段之后，它能被机械校验。

    为什么基线单独成字段而不是写在 note 里（gatea-video 提的）：
    **「有没有基线」是本仓判断一个倍数能不能用的分水岭**，写在散文里闸口查不到，
    而这正是最该查的那一类 —— 厂商材料里的倍数十有八九没说对照的是什么。
    """
    for i, p in enumerate(doc.get("performance") or []):
        if not isinstance(p, dict) or p.get("speedup") is None:
            continue
        what = ((p.get("baseline") or {}).get("what") or "").strip()
        if not what:
            errors.append(
                f"{path}: performance[{i}] 报了 speedup {p['speedup']} 却没有 "
                "baseline.what —— **没有基准的倍数不能用**（CLAUDE.md）。"
                "对照的是什么要写成字段，不是写在 note 里")


def check_family_registry(root, errors):
    """schema 的 family 枚举与站点的 FAMILY_LABEL / FAMILY_ORDER 必须一一对应。

    **漏一个的后果是静默的。** `site/src/pages/families/[family].astro` 的
    getStaticPaths 是 `FAMILY_ORDER.filter(f => fams.has(f))` —— 不在 FAMILY_ORDER
    里的家族不报错、不建页，整个家族页凭空消失，而五道闸全绿。
    视频线第一份记录落地时就这么丢过一次 wan：schema 有 15 个家族，
    data.ts 只有 7 个，模型页建出来了、家族页没有。

    这是「同一份名单在两处各写一遍」的第三个实例（前两个是 basis 标签表与
    kv_table 的漂移守卫）。名单本身没法收口到一处 —— schema 是 JSON、
    站点是 TS，没有共同的运行时；所以改成在这里对账。
    """
    import re as _re
    sp = os.path.join(root, "data", "schema", "model.schema.json")
    dp = os.path.join(root, "site", "src", "lib", "data.ts")
    try:
        schema_fams = set(json.load(open(sp, encoding="utf-8"))
                          ["properties"]["family"]["enum"])
    except (OSError, KeyError, ValueError) as e:
        errors.append(f"data/schema/model.schema.json: 读不到 family 枚举（{e}）")
        return
    try:
        ts = open(dp, encoding="utf-8").read()
    except OSError as e:
        errors.append(f"site/src/lib/data.ts: 读不到（{e}）")
        return

    def _block(name):
        i = ts.find(name)
        if i < 0:
            return None
        j = ts.find("};", i) if "Record<string, string>" in ts[i:i + 80] else ts.find("];", i)
        return ts[i:j] if j > 0 else None

    lab = _block("FAMILY_LABEL")
    order = _block("FAMILY_ORDER")
    if lab is None or order is None:
        errors.append("site/src/lib/data.ts: 找不到 FAMILY_LABEL 或 FAMILY_ORDER")
        return
    # 键可以带引号也可以不带（hunyuan-video 这种带连字符的必须带）
    lab_keys = set(_re.findall(r"^\s*'([a-z0-9-]+)'\s*:", lab, _re.M)) | \
               set(_re.findall(r"^\s*([a-z][a-z0-9]*)\s*:", lab, _re.M))
    order_keys = set(_re.findall(r"'([a-z0-9-]+)'", order))

    for name, got in (("FAMILY_LABEL", lab_keys), ("FAMILY_ORDER", order_keys)):
        miss = schema_fams - got
        extra = got - schema_fams
        if miss:
            errors.append(f"site/src/lib/data.ts 的 {name} 缺 {sorted(miss)} —— "
                          "schema 的 family 枚举有、这里没有。"
                          "FAMILY_ORDER 缺的那个家族**不会报错，只是整个家族页不建**")
        if extra:
            errors.append(f"site/src/lib/data.ts 的 {name} 多出 {sorted(extra)} —— "
                          "schema 的 family 枚举里没有这些值，记录填不进来")


def check_stray_labels(root, errors):
    """第二份标签表：代码里不许把 basis 取值自己映射成中文。

    取值表已经有过多份各自不全的拷贝 —— `vendor_declared` 进了 derive.py 和 schema
    之后，三个用它的地方一个都没跟上，于是全仓证据第三强的那一档在页面上渲染成一个
    不带口径的裸数字。单位和口径必须跟着数字走，而跟不跟得上，取决于有没有人又抄一份。

    **现在报 error**：两份拷贝都已收口到 `site/src/lib/data.ts` 的 `KV_BASIS_LABEL`，
    家在 `LABEL_HOME` 里豁免，所以再出现第三份就是真的抄了一份。
    收口时发现那两份的措辞本来就不一样（模型页「按厂商公布值」vs data.ts「厂商公布」）
    —— 重复的代价不是多写几行，是同一个取值在站内有两种读法。

    已知边界：**逐行判，所以键和中文值分在两行的条目漏得掉**（`theoretical_floor:`
    换行再写值）。不致命 —— 一张表里总有写成一行的条目，表本身还是会被报出来；
    但「这一条没被报」不等于「这一行没问题」。
    """
    for base in LABEL_SCAN:
        for dirpath, dirnames, filenames in os.walk(os.path.join(root, base)):
            dirnames[:] = [d for d in dirnames
                           if d not in ("node_modules", "__pycache__", "dist", ".astro")]
            for fn in sorted(filenames):
                if not fn.endswith((".ts", ".astro", ".py")):
                    continue
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root)
                if rel in LABEL_HOME:
                    continue
                try:
                    lines = open(full, encoding="utf-8").read().split("\n")
                except OSError:
                    continue
                hits = [(n, v) for n, line in enumerate(lines, 1)
                        if (v := label_table_hit(line))]
                if hits:
                    where = "、".join(f"{n}（{v}）" for n, v in hits[:6])
                    more = f" 等 {len(hits)} 处" if len(hits) > 6 else ""
                    errors.append(
                        f"{rel}: 第 {where}{more} 行把 basis 取值自己映射成中文 —— "
                        "又一份标签表。改成从 site/src/lib/data.ts 的 KV_BASIS_LABEL / "
                        "KV_BASIS_NOTE 查（那是 basis 标签唯一的家，不在 columns.json 里）")


# 渲染跟踪字段的页面。新加一张这种表就要加进来 —— 不进来就没人检查它会不会
# 又把三态渲染成同一个 `—`。
TRACKED_TABLES = [os.path.join("site", "src", "pages", "families", "[family].astro")]
DATA_TS = os.path.join("site", "src", "lib", "data.ts")
COLUMNS_TS = os.path.join("site", "src", "lib", "columns.ts")


def check_renderers(root, errors):
    """渲染器不许绕过契约。

    两种绕法都真的出过：
      1. 直接调 f.get(m)，缺值就又变回一个什么都不说的 `—`，三态同形
      2. 按「有没有值」决定整行出不出，于是「查了但没有」那一态连行一起消失 ——
         46 份记录的 KV 精度正是这么在五个家族页上整行不见的，比渲染成 `—` 还糟

    两种语言的 op 词表也在这里比：不一致的话，同一个空格子在闸口里是「不适用」、
    在页面上是「未录」。
    """
    ts = os.path.join(root, COLUMNS_TS)
    if not os.path.exists(ts):
        errors.append(f"缺 {COLUMNS_TS} —— 站点侧没有读契约的入口，渲染器只能继续猜")
    else:
        ts_ops = set(re.findall(r"case '([a-z_]+)':", open(ts, encoding="utf-8").read()))
        from lib import columns
        if ts_ops != columns.OPS:
            errors.append(f"op 词表两侧不一致：python={sorted(columns.OPS)} "
                          f"typescript={sorted(ts_ops)}")

    dt = os.path.join(root, DATA_TS)
    if os.path.exists(dt) and "from './columns'" not in open(dt, encoding="utf-8").read():
        errors.append(f"{DATA_TS} 没有 import ./columns —— 契约没接进渲染路径，"
                      "columns.json 成了摆设，缺值仍然由页面自己猜")

    for rel in TRACKED_TABLES:
        full = os.path.join(root, rel)
        if not os.path.exists(full):
            errors.append(f"找不到 {rel}（改名了就把 TRACKED_TABLES 一起改）")
            continue
        src = open(full, encoding="utf-8").read()
        if re.search(r"\bf\.get\(", src):
            errors.append(f"{rel} 里直接调了 f.get() —— 跟踪字段的格子一律走 "
                          "cell() / <Cell>：直接调的话缺值就又变回一个什么都不说的 `—`，"
                          "而那个符号同时背着「没查」「不适用」「查了但没有」三种意思")
        if "rowIsInformative" not in src:
            errors.append(f"{rel} 没有用 rowIsInformative 判断整行出不出 —— "
                          "按「有没有值」判会把「查了但没有」那一态连行一起藏掉，"
                          "那一态是结论，不是缺口")


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
    check_renderers(_root(args.data_dir), errors)
    check_stray_labels(_root(args.data_dir), errors)
    check_family_registry(_root(args.data_dir), errors)
    check_field_consumers(_root(args.data_dir), errors, warnings)
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
        check_release_vs_measured(doc, p, os.path.join(args.data_dir, "derived"), errors)
        check_speedup_baseline(doc, p, errors)
        check_scope_coherence(doc, p, errors)
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
