#!/usr/bin/env python3
"""回归用例：那三次误读，改之前应当报错、改之后应当通过。

四道闸全绿的那一周被连着问出三个问题，**三个都不是数值错，全在呈现层**。
闸口只跑一次「现在是绿的」不算数 —— 要能证明它当时会报、而且以后再犯还会报。
所以这里把三次误读各写成一对用例：

  负例  把那次误读的条件重新造出来，断言闸口报错
  正例  拿真实的 48 份记录，断言闸口不报，而且判出来的结果正是当时被读错的那个真相

用法:
  .venv/bin/python tests/absence_regression.py          # 全部跑，失败非零退出
  .venv/bin/python tests/absence_regression.py -v       # 连通过的也印出来
"""
from __future__ import annotations

import collections
import copy
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from lib import columns  # noqa: E402
from lib import yamlio  # noqa: E402

import validate  # noqa: E402
import fetch_hf  # noqa: E402
from lib import cfgio  # noqa: E402


# ------------------------------------------------------------ 小工具

PASSED: list = []
FAILED: list = []


def case(name: str):
    def deco(fn):
        try:
            fn()
        except AssertionError as e:
            FAILED.append((name, str(e)))
        except Exception as e:  # noqa: BLE001
            FAILED.append((name, f"{type(e).__name__}: {e}"))
        else:
            PASSED.append(name)
        return fn
    return deco


def contract_errors(spec: dict) -> list:
    """拿一份改过的 columns.json 跑契约检查，返回报出来的错误。"""
    saved = columns._spec
    columns._spec = spec
    try:
        errors, warnings = [], []
        validate.check_absence_contract(errors, warnings)
        return errors
    finally:
        columns._spec = saved


def load_spec() -> dict:
    return copy.deepcopy(columns.spec())


def records() -> dict:
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "data", "models", "*", "*.yaml"))):
        d = yamlio.load(p) or {}
        if d.get("id"):
            out[d["id"]] = d
    return out


def derived(mid: str):
    p = os.path.join(ROOT, "data", "derived", f"{mid}.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except (ValueError, OSError):
        return None


RECS = records()


def release_errors(doc) -> list:
    """40 份迁完之后这一条已经从告警收紧成错误，所以收的是 errors。"""
    e = []
    validate.check_release_vs_measured(
        doc, doc.get("id", "?"), os.path.join(ROOT, "data", "derived"), e)
    return e


# ============================================================ 误读一：单位
#
# 族系旗舰表以「元素/token」为主键、DeepSeek 精度表以「字节/token」为主键，
# 两张表并排，列名离数字十几个字符。DeepSeek-V4.1-Flash 的 1,600（元素）
# 被读成字节，与另一张表的 890 B 对不上，看起来像数据错了。两个数都对：
# 1,600 元素 × 0.556 B/元素 = 890 B，是两个口径。

@case("一·负例：声明了 quantity 却没有 unit -> 报错")
def _():
    spec = load_spec()
    spec["columns"]["kv_elems"].pop("unit", None)
    errs = contract_errors(spec)
    assert any("kv_elems" in e and "unit" in e for e in errs), \
        f"没有报出 kv_elems 缺单位，实际报了：{errs}"


@case("一·正例：会被读错成别的量的列都带着单位")
def _():
    for key, col in columns.columns().items():
        if col.get("quantity"):
            assert col.get("unit"), f"列 {key} 声明了 quantity 却没有 unit"


@case("一·正例：元素列与字节列的单位不同 —— 这正是当时被混掉的那个区别")
def _():
    e = columns.column("kv_elems")
    b = columns.column("kv_bytes")
    assert e.get("unit") and b.get("unit"), "元素列与字节列都要有单位"
    assert e["unit"] != b["unit"], \
        f"两列的单位不能相同（{e['unit']!r} vs {b['unit']!r}）—— 相同就分不出是哪种量"


# ============================================================ 误读二：三态
#
# `—` 一个符号背着三种含义：没查 / 不适用 / 查了但没有。
# 46 份记录的 precision.kv_cache 是空的，真相是「厂商没公布，字节按权重 dtype
# 兜底」—— 那是一条结论，不是缺口。而页面上它和「还没查」长得一样。

@case("二·负例：一列没有 absent 声明 -> 报错")
def _():
    spec = load_spec()
    spec["columns"]["kv_dtype"].pop("absent", None)
    errs = contract_errors(spec)
    assert any("kv_dtype" in e for e in errs), f"没有报出 kv_dtype 缺 absent：{errs}"


@case("二·负例：判「不适用」却不说凭什么 -> 报错")
def _():
    spec = load_spec()
    spec["columns"]["kv_dtype"]["absent"] = {"state": "not_applicable"}
    errs = contract_errors(spec)
    assert any("why" in e for e in errs), f"没有报出缺 why：{errs}"


@case("二·负例：规则用了没实现的 op -> 报错")
def _():
    spec = load_spec()
    spec["columns"]["kv_dtype"]["absent"]["rules"] = [
        {"state": "confirmed_absent", "why": "x",
         "test": {"src": "rec", "op": "regex_match_somehow", "path": "a.b"}}]
    errs = contract_errors(spec)
    assert any("op" in e for e in errs), f"没有报出未实现的 op：{errs}"


@case("二·负例：态写成三态之外的值 -> 报错")
def _():
    spec = load_spec()
    spec["columns"]["kv_dtype"]["absent"]["state"] = "missing"
    errs = contract_errors(spec)
    assert any("三态" in e for e in errs), f"没有报出非法的态：{errs}"


@case("二·正例：真实记录的契约是完整的")
def _():
    errors, warnings = [], []
    validate.check_absence_contract(errors, warnings)
    assert not errors, f"契约检查报错：{errors}"


@case("二·正例：空着的 KV 精度按 kind 分成「未公布」与「不适用」，没有「未录」")
def _():
    # **这条用例断言的是规则，不是人口普查。**
    # 原先写死「46 份」「47 份」，于是每进一份生成类记录就红一次 ——
    # 红得有道理（分布确实变了），但每次都只能把数字加一，用例什么也没守住。
    # 现在按 kind 推期望值：
    #   生成类空着 = 每步重算、根本不建 KV，没有这笔账      → not_applicable
    #   自回归空着 = 厂商没公布落盘精度，字节按权重 dtype 兜底 → confirmed_absent
    # **两者都不是 unsurveyed**，unsurveyed 才是「本仓还没查」。
    blank = [i for i, d in RECS.items() if not ((d.get("precision") or {}).get("kv_cache"))]
    assert blank, "一份空着的都没有，判据多半失效了"
    for mid in blank:
        is_gen = RECS[mid].get("kind") in ("video-gen", "image-gen")
        want = "not_applicable" if is_gen else "confirmed_absent"
        a = columns.absence("kv_dtype", RECS[mid], derived(mid))
        assert a["state"] == want, \
            f"{mid}（kind={RECS[mid].get('kind')}）判成了 {a['state']}，应当是 {want}"
        assert not a["is_gap"], f"{mid}：这两态都是信息，不该算成缺口"
        assert a["why"], f"{mid}：判了却说不出凭什么"


@case("二·正例：契约判出来的两种说法与 kind 一一对应")
def _():
    # 同上，不数总数 —— 只查「标签 ↔ kind」这个对应关系没有串。
    # 站点侧 data.ts 的 kvCachePrecision 委托给同一份契约，所以这里过了那边就一致。
    pairs = collections.Counter()
    for mid, d in RECS.items():
        if (d.get("precision") or {}).get("kv_cache"):
            continue
        kind = "生成类" if d.get("kind") in ("video-gen", "image-gen") else "自回归"
        pairs[(kind, columns.absence("kv_dtype", d, derived(mid))["label"])] += 1
    bad = [k for k in pairs if k not in (("生成类", "不适用"), ("自回归", "未公布"))]
    assert not bad, f"出现了不该有的组合：{bad}（全部：{dict(pairs)}）"
    assert pairs[("自回归", "未公布")] > 0 and pairs[("生成类", "不适用")] > 0, \
        f"两种组合要各有至少一份，否则这条用例没验到对应关系：{dict(pairs)}"


@case("二·正例：三态都真的用得上 —— 没有一态是摆设")
def _():
    seen = set()
    subs = [s["key"] for s in json.load(open(
        os.path.join(ROOT, "data", "schema", "subsystems.json")))["subsystems"]]
    for mid, d in RECS.items():
        dv = derived(mid)
        ps = ((d.get("precision") or {}).get("quantization") or {}).get("per_subsystem") or {}
        for key in list(columns.columns()) + [f"sub_{s}" for s in subs if s not in ps]:
            seen.add(columns.absence(key, d, dv)["state"])
    assert seen == {"unsurveyed", "not_applicable", "confirmed_absent"}, \
        f"只用上了 {sorted(seen)} —— 没用上的那一态要么判据写错了，要么不该存在"


@case("二·正例：结构判据分得开「不适用」与「未录」")
def _():
    # 非 latent 注意力没有 KV 低秩压缩，是结构上不适用；
    # MLA / hybrid 有这个结构而没填，才是没查
    for mid, d in RECS.items():
        if ((d.get("arch") or {}).get("attention") or {}).get("kv_lora_rank") is not None:
            continue
        t = ((d.get("arch") or {}).get("attention") or {}).get("type")
        a = columns.absence("kv_lora", d, derived(mid))
        want = "unsurveyed" if t in ("MLA", "hybrid", "MQA-latent") else "not_applicable"
        assert a["state"] == want, f"{mid}（{t}）的 KV latent 秩判成了 {a['state']}，应当是 {want}"


# ============================================================ 误读三：取数口径
#
# 精度表只显示本仓取数那个仓，厂商的多精度独立发布不可见。MiniMax-M3 确实是
# BF16，而厂商另发的 MXFP8 变体本仓未录 —— 表上看着像 M2 都低精度了而 M3
# 莫名倒退。

@case("三·负例：实测列不说测自哪里 -> 报错")
def _():
    spec = load_spec()
    spec["columns"]["$subsystem"].pop("scope_from", None)
    errs = contract_errors(spec)
    assert any("scope_from" in e for e in errs), f"没有报出缺 scope_from：{errs}"


@case("三·正例：有实测精度的记录都说得出测自哪个仓")
def _():
    tpl = columns.subsystem_template()
    path = tpl.get("scope_from")
    assert path, "逐子系统实测列没有 scope_from"
    for mid, d in RECS.items():
        if not ((d.get("precision") or {}).get("quantization") or {}).get("per_subsystem"):
            continue
        cur = d
        for part in path.split("."):
            cur = cur.get(part) if isinstance(cur, dict) else None
        assert cur, f"{mid} 有实测精度，却说不出测自哪个仓（{path} 是空的）"


@case("三·正例：「声明了而清单里没有」的记录，每一份都得带交代")
def _():
    # **这条断言规则，不是名单。** 原先写死四个 id，于是每进一份这类记录就红一次
    # （今天已经第三条这样的用例了：KV 那两条同病）。名单会变，规律不会 ——
    # 凡是摘掉 weights_release_scope 就报错的，必须本来就带着 scope 交代。
    #
    # 这一类长什么样：厂商另发了量化变体、本仓没把它的实测值并进精度表。
    # 典型是 minimax-m3（MiniMaxAI/MiniMax-M3-MXFP8 并行维护，本仓取过清单
    # 但没进 per_subsystem 表）与三份通义变体。LTX 两份进来之后是六份。
    # 闸口照旧放行（它们有 scope），这条查的是更底下那一层：
    # 摘掉交代之后还认不认得出来。
    #
    # **2026-10-31 起「交代」有两种。** 精度页的取数口径从「主仓那一份权重」扩到
    # 「厂商官方发布过的最低精度」，于是一个变体可以不靠 scope 的布尔、而靠
    # precision.quantization.official_variants 里的**实测值**为自己的缺席作证
    # （那份缺席只是「主仓的 derived 直方图里没有」，而它本来就不该有）。
    # 所以这条用例两样都摘：摘完还报错的，才是真正「只能靠 scope 交代」的那些。
    need, missing = [], []
    for mid, d in RECS.items():
        bare = copy.deepcopy(d)
        bare.get("precision", {}).pop("weights_release_scope", None)
        (bare.get("precision", {}).get("quantization") or {}).pop("official_variants", None)
        if not release_errors(bare):
            continue
        need.append(mid)
        if not (d.get("precision") or {}).get("weights_release_scope"):
            missing.append(mid)
    assert need, "一份都没有，说明 release_errors 失效了 —— 这条用例没验到东西"
    assert not missing, (
        f"这几份「声明了而清单里没有」却没带 weights_release_scope 交代：{sorted(missing)}")
    # 带了交代的，每一档都要说清是「取没取过清单」还是「进没进精度表」
    for mid in need:
        for prec, sc in (RECS[mid]["precision"]["weights_release_scope"]).items():
            assert "manifest_taken" in sc and "in_precision_table" in sc, \
                f"{mid} 的 {prec} 档缺两个布尔之一 —— 一个布尔装不下两件事"


@case("三·负例：把两种交代都摘掉，那一份立刻报错")
def _():
    # 这四份是全仓真正「声明了而主仓清单里没有」的，交代摘掉就该拦住。
    # **两种交代都要摘**：scope 的布尔，和 official_variants 里的实测值 ——
    # 后者是 2026-10-31 新增的第二种，只摘前一种的话这条用例会在
    # qwen3.5-27b 上静默失效（它已有逐仓实测，自己就能为缺席作证）
    for mid in ("minimax-m3", "qwen3.5-27b", "qwen3.5-397b-a17b", "qwen3.8-2.4t-a95b"):
        d = copy.deepcopy(RECS[mid])
        d["precision"].pop("weights_release_scope", None)
        (d["precision"].get("quantization") or {}).pop("official_variants", None)
        assert release_errors(d), f"{mid} 摘掉两种交代之后应当报错"


@case("三·正例：official_variants 的实测值本身就是交代，不必再靠 scope 的布尔")
def _():
    """口径扩到「厂商官方发布过的最低精度」之后新增的一条。

    一个变体在厂商另发的独立仓里，本仓取过那个仓的清单并把逐子系统实测落了库 ——
    那时它从主仓的 derived 直方图里缺席是**正常的**，不是缺口：主仓根本没有那个精度。
    所以 official_variants 里有同 dtype 的一条、且那一条的 per_subsystem 里真的
    出现了这个精度时，闸口放行，不要求 scope 再说一遍。
    """
    d = copy.deepcopy(RECS["qwen3.5-27b"])
    d["precision"].pop("weights_release_scope", None)
    assert not release_errors(d), \
        f"逐仓实测在 official_variants 里，这就是交代：{release_errors(d)}"


@case("三·负例：official_variants 列了一档却没测出那个精度 —— 不算交代")
def _():
    """**判据是实测值，不是「有这么一条」。**

    光把 dtype 写进 official_variants、而那一条的 per_subsystem 里一格都不是这个精度，
    等于换了个地方说谎 —— 和「标 in_precision_table: true 却查无实据」是同一种错法，
    所以要用同一种方式挡住。
    """
    d = copy.deepcopy(RECS["qwen3.5-27b"])
    d["precision"].pop("weights_release_scope", None)
    for v in d["precision"]["quantization"]["official_variants"]:
        # 精度全改成 BF16：条目还在、dtype 还写着 INT4 / FP8，但一格实测都对不上
        v["per_subsystem"] = {k: "BF16" for k in (v.get("per_subsystem") or {})}
    assert release_errors(d), "条目在而实测值不在，必须继续报"


@case("三·正例：40 份迁完之后，真实记录一条都不报")
def _():
    for mid, d in RECS.items():
        assert not release_errors(d), f"{mid} 还在报：{release_errors(d)}"


@case("三·负例：只交代一半，剩下那半照旧报")
def _():
    d = copy.deepcopy(RECS["qwen3.5-27b"])
    # 这一条查的是 scope 这套机制本身，所以把另一种交代摘掉 ——
    # 留着的话两档都被 official_variants 的实测值放行，用例就验不到东西了
    (d["precision"].get("quantization") or {}).pop("official_variants", None)
    d["precision"]["weights_release_scope"] = {
        "FP8_E4M3": {"manifest_taken": False, "in_precision_table": False,
                     "repo": "Qwen/Qwen3.5-27B-FP8"},
    }
    after = release_errors(d)
    assert after, "还有一种没交代，应当继续报"
    assert "INT4" in after[0] and "FP8_E4M3" not in after[0], \
        f"应当只报剩下的 INT4：{after}"


@case("三·正例：in_precision_table: true 不能当成交代 —— 那是在说谎")
def _():
    # 声明了、derived 的 by_dtype 里没有、却标 in_precision_table: true：
    # 这是自相矛盾，不该被放行
    d = copy.deepcopy(RECS["qwen3.8-2.4t-a95b"])
    d["precision"]["weights_release_scope"] = {
        "FP8_E4M3": {"manifest_taken": True, "in_precision_table": True},
    }
    assert release_errors(d), "标 in_precision_table: true 却查无实据，必须继续报"


@case("三·负例：拆字段之后，manifest_taken 单独不足以当交代")
def _():
    """这条是拆字段（2026-10-30）的核心回归。

    旧字段下「取过清单」与「进了精度表」是同一个布尔，所以这两件事分不开。
    拆开之后要保证闸口认的仍然是后者 —— 一个变体清单取过、但实测值没并进
    逐子系统精度表，依旧要能用 in_precision_table: false 交代；
    反过来只写 manifest_taken 而不写 in_precision_table，不能蒙混过关。
    """
    d = copy.deepcopy(RECS["qwen3.8-2.4t-a95b"])
    # 只说取过清单，不说进没进表 —— 闸口应当照旧报
    d["precision"]["weights_release_scope"] = {"FP8_E4M3": {"manifest_taken": True}}
    assert release_errors(d), "只写 manifest_taken 不构成交代，必须继续报"
    # 补上「没进表」才算交代清楚
    d["precision"]["weights_release_scope"] = {
        "FP8_E4M3": {"manifest_taken": True, "in_precision_table": False},
    }
    assert not release_errors(d), "manifest_taken: true + in_precision_table: false 是正当交代"


@case("三·负例：两个布尔自相矛盾要被挡住")
def _():
    """没取过清单不可能有实测值进表。

    这种错法在拆字段之前写不出来（只有一个布尔，没有第二个键可以和它冲突）——
    拆开换来表达力，也换来新的错法，所以拆的同时就得把它挡住。
    """
    import validate as V
    d = copy.deepcopy(RECS["minimax-m3"])
    d["precision"]["weights_release_scope"]["MXFP8"] = {
        "manifest_taken": False, "in_precision_table": True,
    }
    errs = []
    V.check_scope_coherence(d, "<test>", errs)
    assert errs and "自相矛盾" in errs[0], f"应当报自相矛盾，实际 {errs}"
    # 真实记录里一条都不该有
    for mid, rec in RECS.items():
        e = []
        V.check_scope_coherence(rec, mid, e)
        assert not e, f"{mid} 报了：{e}"


@case("三·正例：四份「逻辑精度 vs 容器」不被误报")
def _():
    # FP4 装 INT8、8 个 INT4 一个 I32、MXFP4 一字节两个 —— 这是正确记法。
    # derive.py 在 packing_factor > 1 时已经把容器换成逻辑精度，所以对得上
    for mid in ("deepseek-v41-flash", "kimi-k2-thinking", "kimi-k2.5", "kimi-k3"):
        d = RECS.get(mid)
        assert d, f"{mid} 不在记录里"
        bare = copy.deepcopy(d)
        bare.get("precision", {}).pop("weights_release_scope", None)
        assert not release_errors(bare), f"{mid} 被误报了（且不是靠 scope 兜的）"


# ================================================== 标签表：按结构找不按词找
#
# 取值表有过多份各自不全的拷贝：vendor_declared 进了 derive.py 和 schema 之后，
# 三个用它的地方一个都没跟上，于是全仓证据第三强的那一档在页面上渲染成一个不带
# 口径的裸数字。判据要能抓到三元写法（按形状找 `const <NAME> = {` 抓不到），
# 又不能误伤散文（按词找会把口径说明全部误伤）。

@case("标签·正例：对象键式的标签表抓得到")
def _():
    assert validate.label_table_hit("  theoretical_floor: '理论下限',") == "theoretical_floor"


@case("标签·正例：三元式的抓得到 —— 按形状找正是漏在这里")
def _():
    line = ("{lw.bytes_basis === 'theoretical_floor' ? '，理论下限' "
            ": '，按模型 dtype 兜底'}")
    assert validate.label_table_hit(line) == "theoretical_floor"


@case("标签·正例：JSX 文本式的抓得到")
def _():
    line = ("{kv?.bytes_basis === 'dtype_fallback' && "
            "<span class=\"muted\">（精度兜底）</span>}")
    assert validate.label_table_hit(line) == "dtype_fallback"


@case("标签·负例：真·basis 行为不是标签表，要放过")
def _():
    # main 的 LayerMap 里这两处是行为不是标签：R31 交叉引用、Evidence judgement 标记，
    # 都只对 theoretical_floor 成立。放过它们是这条判据能不能用的前提
    for line in (
        "const floor = basis === 'theoretical_floor';",
        "{r.s.basis === 'theoretical_floor' ? <> <Evidence tag=\"judgement\" /></> : ''}</td>",
        "          {floor ? <>（缺口见核查区 <b>R31</b>）</> : ''}。</>}",
    ):
        assert validate.label_table_hit(line) is None, f"误伤了行为：{line}"


@case("标签·负例：散文提到取值名不是标签表，要放过")
def _():
    for line in (
        "    字节由落盘精度决定（<code>theoretical_floor</code> 是查到落盘格式的理论下限，",
        "                {/* 查表，不写 ternary：原先是「=== theoretical_floor ? 理论下限」，",
        '    """整体 basis 取记录里写的那个；没写就看各分量是否一致，不一致记 mixed。"""',
    ):
        assert validate.label_table_hit(line) is None, f"误伤了散文：{line}"


@case("标签·反例：全仓现在一份多余的标签表都没有")
def _():
    # 这条用例的期望值改过三次，每次都是真实状态变了：
    #   3 处 → derive.py 的 CLI 改成直接印 bytes_basis 原值，不再是标签表
    #   2 处 → [id].astro 的那份删掉改查 KV_BASIS_LABEL；data.ts 是家，进 LABEL_HOME
    #   0 处 → 现在
    # **预期跟着真实状态走，不是把修复回退去迁就用例。**
    # 收口时才发现那两份的措辞本来就不一样（「按厂商公布值」vs「厂商公布」）——
    # 重复的代价不是多写几行，是同一个取值在站内有两种读法。
    errs = []
    validate.check_stray_labels(ROOT, errs)
    assert errs == [], f"又冒出标签表：{sorted(x.split(':')[0] for x in errs)}"


@case("标签·正例：真多出一份时这道闸会红 —— 没见它红过的闸不算闸")
def _():
    # 上一条是反例（全仓干净）。反例单独立着是空的：如果 check_stray_labels 哪天
    # 坏成永远返回空，它照样通过。所以这里放一个探针文件进扫描范围再撤掉。
    probe = os.path.join(ROOT, "site", "src", "lib", "__regression_probe.ts")
    try:
        with open(probe, "w", encoding="utf-8") as f:
            f.write("const X: Record<string,string> = { vendor_declared: '厂商公布' };\n")
        errs = []
        validate.check_stray_labels(ROOT, errs)
        hit = [x for x in errs if "__regression_probe" in x]
        assert hit, f"探针没被抓到，这道闸是哑的：{errs}"
    finally:
        if os.path.exists(probe):
            os.remove(probe)


# ================================ 读 config：声明可能嵌在 text_config 下
#
# Kimi 的四份 VLM（k2.5 / k2.6 / k2.7-code / k3）把整个 `quantization_config`
# 嵌在 `text_config` 下，顶层只剩 `dtype: bfloat16`。仓里曾有四个消费点两套读法：
# derive.py 拍平后取，fetch_dtypes / extract_layers / scaffold 只读顶层。
# 后果不是报错 —— 是逐子系统精度表那几格渲染成「未录」，而 config 明写着
# group_size 32。**「未录」让人以为去查就行，没人会想到判据本身读错了地方。**

@case("读 config·正例：嵌在 text_config 下的 quantization_config 要找得到")
def _():
    import json as _json
    # **判据不能依赖 facts/** —— 它在 gitignore 里、不随分支走，worktree 与新克隆
    # 都没有。我第一版写成「一份都读不到就失败」，于是在任何干净 checkout 上都红。
    # 逻辑本身由下面那条合成用例守；这里只在 facts 真的在的时候做实物校验。
    want = {"kimi-k3", "kimi-k2.5", "kimi-k2.6", "kimi-k2.7-code"}
    for mid in want:
        fp = os.path.join(ROOT, "facts", mid, "cfg.json")
        if not os.path.exists(fp):
            continue
        cfg = _json.load(open(fp, encoding="utf-8"))
        assert cfgio.quant_site(cfg) == "text_config", \
            f"{mid} 的声明位置变了：{cfgio.quant_site(cfg)}"
        q = cfgio.quant_cfg(cfg)
        assert q, f"{mid}：嵌套的 quantization_config 没读到"
        grp = next(iter((q.get("config_groups") or {}).values()), {}).get("weights", {})
        assert grp.get("group_size") == 32, f"{mid}: group_size={grp.get('group_size')}"
        assert grp.get("num_bits") == 4, f"{mid}: num_bits={grp.get('num_bits')}"


@case("读 config·正例：合成的嵌套 config 也要读到（不依赖 facts/）")
def _():
    # 上一条要 facts/ 才验得了实物，而 facts/ 在 gitignore 里。这条守逻辑，
    # 任何 checkout 上都跑得了 —— 把 quant_cfg 改回只读顶层，它会红。
    cfg = {"dtype": "bfloat16",
           "text_config": {"quantization_config": {
               "format": "mxfp4-pack-quantized",
               "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 32}}}}}}
    assert cfgio.quant_site(cfg) == "text_config"
    q = cfgio.quant_cfg(cfg)
    assert q.get("format") == "mxfp4-pack-quantized"
    assert q["config_groups"]["group_0"]["weights"]["group_size"] == 32


@case("读 config·反例：顶层声明的照常读得到，位置记成 root")
def _():
    cfg = {"quantization_config": {"format": "pack-quantized"}, "text_config": {}}
    assert cfgio.quant_site(cfg) == "root"
    assert cfgio.quant_cfg(cfg)["format"] == "pack-quantized"


@case("读 config·反例：两处都没有时返回 {} 而不是 None")
def _():
    # 调用方一律 .get()，返回 None 会炸在很远的地方
    assert cfgio.quant_cfg({"dtype": "bfloat16"}) == {}
    assert cfgio.quant_site({"dtype": "bfloat16"}) == "none"


@case("读 config·正例：text_config 拍平后层数别名认 num_layers")
def _():
    # 扩散主干的 config 把层数叫 num_layers，不认的后果是 Gate B 的层数核对静默 skip
    assert cfgio.flatten({"num_layers": 30})["num_hidden_layers"] == 30
    # 两个键都有时不覆盖（glm-4-9b 是这种，两处值相同）
    assert cfgio.flatten({"num_hidden_layers": 40, "num_layers": 40})["num_hidden_layers"] == 40


# ============================================ 倍数必须带基线
#
# CLAUDE.md 写着「没有基准的倍数不能用，最多进 open_question」。那条规矩此前只活在
# 散文里 —— 而厂商材料里的倍数十有八九没说对照的是什么，正是最该查的一类。
# `performance[].speedup` 与 `baseline.what` 做成一对之后它能被机械校验。

@case("性能·正例：报了 speedup 没有 baseline.what 要被拦住")
def _():
    doc = {"performance": [{"source": {"type": "official"}, "hardware": "H800",
                            "engine": "x", "speedup": 1.87}]}
    errs = []
    validate.check_speedup_baseline(doc, "probe.yaml", errs)
    assert errs and "baseline.what" in errs[0], f"没拦住：{errs}"


@case("性能·反例：带了基线就放行")
def _():
    doc = {"performance": [{"source": {"type": "official"}, "hardware": "H800",
                            "engine": "x", "speedup": 1.87,
                            "baseline": {"what": "FlashAttention-3 稠密注意力"}}]}
    errs = []
    validate.check_speedup_baseline(doc, "probe.yaml", errs)
    assert not errs, errs


@case("性能·反例：没报 speedup 的条目不查基线")
def _():
    # 只给绝对值不给倍数是合法的 —— 这条用例防的是把闸口写成「凡 performance 都要 baseline」
    doc = {"performance": [{"source": {"type": "official"}, "hardware": "H800",
                            "engine": "x", "seconds_per_step": 2.9475}]}
    errs = []
    validate.check_speedup_baseline(doc, "probe.yaml", errs)
    assert not errs, errs


@case("性能·正例：全仓真有带基线的条目，判据没有空转")
def _():
    hit = [(mid, p) for mid, d in RECS.items() for p in (d.get("performance") or [])
           if p.get("speedup") is not None]
    assert hit, "全仓一条带 speedup 的 performance 都没有 —— 这组用例没验到真数据"
    for mid, p in hit:
        assert (p.get("baseline") or {}).get("what"), f"{mid} 报了倍数却没基线"


# ======================================== 取数：别把几个模型合并成一个不存在的模型
#
# Lightricks/LTX-2 的仓里 `ltx-2-19b-distilled.safetensors`（BF16）与
# `-distilled-fp8.safetensors`（F8_E4M3）**张量名完全相同、dtype 不同**。
# 原来的 collect_shapes 是 `out[name] = ...`，后读到的盖掉先读到的，而分片是并发完成的
# —— 哪些张量是 BF16、哪些是 FP8 取决于线程完成顺序。实测这一个仓就有 **7,132 个** 这样的名字。
# 更坏的一种是 LTX-2.3：8 个顶层 safetensors 是 8 个不同模型，而 dev 与 distilled 的
# **形状和 dtype 完全一样**，连冲突都报不出来。所以要两道：选分片时拒绝歧义，合并时查冲突。

@case("取数·反例：有 index.json 的分片仓照常全读")
def _():
    files = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors",
             "model.safetensors.index.json", "config.json"]
    got = fetch_hf.pick_shards(files, [])
    assert len(got) == 2, f"分片仓不该被拦：{got}"


@case("取数·正例：无 index 且顶层多个 safetensors 时拒绝，不猜")
def _():
    # LTX-2.3 的真实形状：8 个独立模型，没有任何 index
    files = ["ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-distilled.safetensors",
             "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"]
    try:
        fetch_hf.pick_shards(files, [])
    except SystemExit as e:
        assert "--include" in str(e), f"拒绝了但没告诉人怎么办：{e}"
        return
    raise AssertionError("没拦住 —— 合并这三个会产出一个不存在的模型")


@case("取数·反例：--include 点名之后放行，且只放行点到的")
def _():
    files = ["ltx-2.3-22b-dev.safetensors", "ltx-2.3-22b-distilled.safetensors"]
    got = fetch_hf.pick_shards(files, ["ltx-2.3-22b-dev.safetensors"])
    assert got == ["ltx-2.3-22b-dev.safetensors"], got


@case("取数·正例：--include 一个都没匹配上要报错，不能静默取空")
def _():
    # 空清单会让 derive.py 把参数量静默算成 0，所以这里必须响
    try:
        fetch_hf.pick_shards(["a.safetensors", "idx.safetensors.index.json"], ["nope-*.safetensors"])
    except SystemExit:
        return
    raise AssertionError("没匹配上却没报错 —— 下游会得到参数量 0")


@case("取数·反例：单个顶层 safetensors 无 index 不算歧义")
def _():
    got = fetch_hf.pick_shards(["model.safetensors", "config.json"], [])
    assert got == ["model.safetensors"], got


# ============================================================ 跑

def main() -> int:
    verbose = "-v" in sys.argv
    print(f"回归用例：{len(PASSED) + len(FAILED)} 条")
    if verbose:
        for n in PASSED:
            print(f"  ok    {n}")
    for n, why in FAILED:
        print(f"  FAIL  {n}\n        {why}")
    print(f"\n{len(PASSED)} 通过, {len(FAILED)} 失败")
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
