#!/usr/bin/env python3
"""回归用例：那三次误读，改之前应当报错、改之后应当通过。

四道闸全绿的那一周被连着问出三个问题，**三个都不是数值错，全在呈现层**。
闸口只跑一次「现在是绿的」不算数 —— 要能证明它当时会报、而且以后再犯还会报。
所以这里把三次误读各写成一对用例：

  负例  把那次误读的条件重新造出来，断言闸口报错
  正例  拿真实的 47 份记录，断言闸口不报，而且判出来的结果正是当时被读错的那个真相

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
# 45 份记录的 precision.kv_cache 是空的，真相是「厂商没公布，字节按权重 dtype
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


@case("二·正例：45 份空着的 KV 精度判成「查了但没有」，不是「未录」")
def _():
    blank = [i for i, d in RECS.items() if not ((d.get("precision") or {}).get("kv_cache"))]
    assert len(blank) == 45, f"空着的应当是 45 份，实际 {len(blank)}"
    for mid in blank:
        a = columns.absence("kv_dtype", RECS[mid], derived(mid))
        assert a["state"] == "confirmed_absent", \
            f"{mid} 的 KV 精度判成了 {a['state']}，而真相是厂商没公布、字节按权重 dtype 兜底"
        assert not a["is_gap"], f"{mid}：这一态是信息，不该算成缺口"
        assert a["why"], f"{mid}：判成「查了但没有」却说不出凭什么"


@case("二·正例：契约判出来的和站点侧手写的 kvCachePrecision 一致")
def _():
    # 站点侧 data.ts 的 kvCachePrecision 是手写的、只管 KV 这一行，注释里写着
    # 「当前 47 份里 45 份走『未公布』、0 份走『不适用』」。契约要能复现同一个结论 ——
    # 复现不了就说明两边在说不同的话，而这条线存在的理由就是消灭这种口径不一致。
    seen = collections.Counter()
    for mid, d in RECS.items():
        if (d.get("precision") or {}).get("kv_cache"):
            continue
        seen[columns.absence("kv_dtype", d, derived(mid))["label"]] += 1
    assert seen == collections.Counter({"未公布": 45}), \
        f"契约判出来是 {dict(seen)}，站点侧手写的结论是 45 份「未公布」、0 份「不适用」"


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


@case("三·正例：全仓真正「声明了而清单里没有」的那四份")
def _():
    got = []
    for mid, d in RECS.items():
        bare = copy.deepcopy(d)
        bare.get("precision", {}).pop("weights_release_scope", None)
        if release_errors(bare):
            got.append(mid)
    # 原先是三份通义。minimax-m3 是后加的第四份：厂商与主仓并行维护
    # MiniMaxAI/MiniMax-M3-MXFP8（2026-10-30 查 HF API 确认存在，31 个分片、
    # 下载 48,817），而本仓只取过主仓的清单。补进 weights_release 之后它
    # 当然会落进这张名单 —— **这正是对的**：声明了、没测过、所以必须带
    # measured: false 交代清楚。闸口照旧放行（它有 scope），这条用例查的是
    # 更底下那一层：摘掉交代之后还认不认得出来。
    want = ["minimax-m3", "qwen3.5-27b", "qwen3.5-397b-a17b", "qwen3.8-2.4t-a95b"]
    assert sorted(got) == want, f"实际是 {sorted(got)}，应当是 {want}"


@case("三·负例：把 scope 摘掉，那一份立刻报错")
def _():
    # 这四份是全仓真正「声明了而清单里没有」的，交代摘掉就该拦住
    for mid in ("minimax-m3", "qwen3.5-27b", "qwen3.5-397b-a17b", "qwen3.8-2.4t-a95b"):
        d = copy.deepcopy(RECS[mid])
        d["precision"].pop("weights_release_scope", None)
        assert release_errors(d), f"{mid} 摘掉 scope 之后应当报错"


@case("三·正例：40 份迁完之后，真实记录一条都不报")
def _():
    for mid, d in RECS.items():
        assert not release_errors(d), f"{mid} 还在报：{release_errors(d)}"


@case("三·负例：只交代一半，剩下那半照旧报")
def _():
    d = copy.deepcopy(RECS["qwen3.5-27b"])
    d["precision"]["weights_release_scope"] = {
        "FP8_E4M3": {"measured": False, "repo": "Qwen/Qwen3.5-27B-FP8"},
    }
    after = release_errors(d)
    assert after, "还有一种没交代，应当继续报"
    assert "INT4" in after[0] and "FP8_E4M3" not in after[0], \
        f"应当只报剩下的 INT4：{after}"


@case("三·正例：measured: true 不能当成交代 —— 那是在说谎")
def _():
    # 声明了、实测清单里没有、却标 measured: true：这是自相矛盾，不该被放行
    d = copy.deepcopy(RECS["qwen3.8-2.4t-a95b"])
    d["precision"]["weights_release_scope"] = {"FP8_E4M3": {"measured": True}}
    assert release_errors(d), "标 measured: true 却查无实据，必须继续报"


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


@case("标签·正例：现在剩下的两处都被报出来")
def _():
    w = []
    validate.check_stray_labels(ROOT, w)
    got = sorted(x.split(":")[0] for x in w)
    # 原先是三处，含 scripts/derive.py 的 CLI 口径标签（只盖 5 个取值里的 2 个）。
    # 那一处没有补成五个标签 —— 那样仓里会多出第四份 basis 标签表，正是这道闸在数的
    # 东西。改成直接印 bytes_basis 原值，于是它不再是一张标签表，自然退出名单。
    # 这条用例因此从 3 改成 2：**预期跟着真实状态走，不是把修复回退去迁就用例。**
    # 剩下两处要等站点侧的 columns.ts 落地才能消。
    want = ["site/src/lib/data.ts", "site/src/pages/models/[id].astro"]
    assert got == want, f"实际报的是 {got}，应当是 {want}"


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
