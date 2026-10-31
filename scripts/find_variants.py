#!/usr/bin/env python3
"""外侧锚：逐记录去厂商的 HF 组织下找「这个模型的官方量化仓」。

**这是本仓唯一一个锚在外侧的检查**（model-audit 规则 11）。其余几道闸 ——
参数量对 HF 的 tally、缓存字节对厂商公布值、weights_release 对权重清单 ——
全都拿「本仓实测」当锚，所以只查得出**内侧多出来的东西**（记录声明了而清单里没有），
查不出**外面有而里面没有的**（厂商发了而记录压根没声明）。MiniMax-M3 漏掉 MXFP8、
通义 13 份记录画出一面 BF16 的墙，都在那个盲区里，而且**闸全绿**。

所以这个脚本不校验任何东西，它只**列**：主仓同名下还有哪些带量化后缀的官方仓。
判据是「同一个 org + 主仓名加后缀」，故意收得很紧：

  认  Qwen/Qwen3.5-27B-FP8、zai-org/GLM-4.5-FP8、Qwen/Qwen3.5-27B-GPTQ-Int4
  不认 别的 org（社区量化不在来源阶梯上，见 CLAUDE.md「第三方转换只能作旁证」）
  不认 主仓名不是前缀的（Qwen3.5-27B-Instruct-FP8 这种另一条支线）

**宁可漏不可误**：漏了由人补，误了会把一个社区仓写进「厂商官方发布」。
所以输出永远要人过一遍，不要拿它直接落库。

用法:
  .venv/bin/python scripts/find_variants.py                 # 全仓 57 份
  .venv/bin/python scripts/find_variants.py --id qwen3.5-27b
  .venv/bin/python scripts/find_variants.py --json out.json # 给批量取数用
"""
import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import yamlio  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API = "https://huggingface.co/api/models"
UA = {"User-Agent": "frontier-model-study/0.1"}

# 后缀 -> 这一档的声明精度。顺序即「位宽从低到高」，用来挑最低的那一个。
# **名字只决定候选，不决定结论** —— 落盘精度一律由 variant_precision.py 逐张量读出来，
# 这张表只负责「值不值得去取清单」。所以这里宁可把 AWQ 和 GPTQ 都列上，
# 哪个真的更低由实测说了算。
SUFFIX_DTYPE = [
    ("-NVFP4", "NVFP4"),
    ("-MXFP4", "MXFP4"),
    ("-FP4", "FP4"),
    ("-GPTQ-Int4", "INT4"),
    ("-AWQ", "INT4"),
    ("-Int4", "INT4"),
    ("-W4A16", "INT4"),
    ("-GPTQ-Int8", "INT8"),
    ("-Int8", "INT8"),
    ("-W8A8", "INT8"),
    ("-MXFP8", "MXFP8"),
    ("-FP8", "FP8_E4M3"),
]
# 不是量化变体的后缀，撞上也跳过 —— 它们是另一条支线或另一种产物
SKIP = ("-GGUF", "-MLX", "-ONNX", "-AWQ-GGUF", "-Base", "-Instruct", "-Chat",
        "-Thinking", "-Preview", "-Distill")


def get_json(url, timeout=60):
    req = urllib.request.Request(url, headers=dict(UA))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def org_models(org, cache):
    """这个 org 下的全部仓名。一次拉完缓存起来 —— 一份记录一次查询太浪费，
    而同一个 org 下常有十几份记录。"""
    if org in cache:
        return cache[org]
    out, page = [], 0
    while True:
        url = f"{API}?author={urllib.parse.quote(org)}&limit=1000&skip={page * 1000}"
        try:
            batch = get_json(url)
        except (urllib.error.URLError, ValueError) as e:
            print(f"!! 拉 {org} 失败：{e}", file=sys.stderr)
            break
        if not batch:
            break
        out.extend(m.get("id", "") for m in batch)
        if len(batch) < 1000:
            break
        page += 1
        time.sleep(0.2)
    cache[org] = out
    return out


def variants_for(repo, names):
    """主仓 repo 的官方量化变体，以及「量化的是兄弟 checkpoint」那一类。

    返回 (variants, siblings)：
      variants  仓名正好是 `<主仓>-<量化后缀>` —— 同一份权重的另一种存法，可以落库
      siblings  仓名是 `<主仓>-<别的什么>-<量化后缀>` —— **量化的不是这一份权重**

    第二类必须单列而不是一并收进去。通义早几代的量化仓挂在 Chat / Instruct 档下
    （`Qwen/Qwen1.5-72B-Chat-GPTQ-Int4`），而本仓那几份记录跟的是 **Base**
    （`Qwen/Qwen1.5-72B`）—— 两者结构相同、参数量相同，于是
    `check_variants` 的逐位对账**会通过**，错误完全隐形。
    这正是本仓最怕的那种形状：闸绿了，而表上写着「这一代官方发过 Int4」，
    说的却是另一个 checkpoint 的事。所以只列不收，由人决定怎么记。
    """
    hits, sibs = [], []
    for name in names:
        if not name.startswith(repo + "-"):
            continue
        tail = name[len(repo):]
        if any(tail.endswith(s) for s in SKIP):
            continue
        for sfx, dt in SUFFIX_DTYPE:
            if tail.lower() == sfx.lower():
                hits.append((name, dt, sfx))
                break
            if tail.lower().endswith(sfx.lower()):
                sibs.append((name, dt, sfx))
                break
    order = {dt: i for i, (_s, dt) in enumerate(SUFFIX_DTYPE)}
    return (sorted(hits, key=lambda h: order.get(h[1], 99)),
            sorted(sibs, key=lambda h: order.get(h[1], 99)))


def _strings(node, where):
    """枚举一棵子树里的所有字符串及其路径。review[] 由调用方自己不传。"""
    if isinstance(node, str):
        yield where, node
    elif isinstance(node, dict):
        for k, v in node.items():
            yield from _strings(v, f"{where}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _strings(v, f"{where}[{i}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", help="只查这一份记录")
    ap.add_argument("--data-dir", default=os.path.join(ROOT, "data"))
    ap.add_argument("--json", help="把结果写成 JSON，给批量取数用")
    args = ap.parse_args()

    cache, rows, docs = {}, [], []
    paths = sorted(glob.glob(os.path.join(args.data_dir, "models", "**", "*.yaml"),
                             recursive=True))
    for p in paths:
        doc = yamlio.load(p)
        if not isinstance(doc, dict):
            continue
        mid = doc.get("id")
        if args.id and mid != args.id:
            continue
        repo = (doc.get("hf") or {}).get("repo")
        if not repo or "/" not in repo:
            continue
        org = repo.split("/", 1)[0]
        vs, sibs = variants_for(repo, org_models(org, cache))
        declared = [str(x).upper()
                    for x in ((doc.get("precision") or {}).get("weights_release") or [])]
        recorded = {str(v.get("dtype")).upper()
                    for v in (((doc.get("precision") or {}).get("quantization") or {})
                              .get("official_variants") or [])}
        docs.append(doc)
        rows.append({"id": mid, "repo": repo,
                     "variants": [{"repo": r, "dtype": d, "suffix": s} for r, d, s in vs],
                     "siblings": [{"repo": r, "dtype": d} for r, d, _s in sibs],
                     "declared": declared, "recorded": sorted(recorded)})

    todo = [r for r in rows if r["variants"]
            and any(v["dtype"] not in r["recorded"] for v in r["variants"])]
    print(f"{len(rows)} 份记录，{sum(1 for r in rows if r['variants'])} 份有官方量化仓，"
          f"{len(todo)} 份还没落库")
    for r in rows:
        if not r["variants"]:
            continue
        miss = [v for v in r["variants"] if v["dtype"] not in r["recorded"]]
        mark = "TODO" if miss else "  ok"
        print(f"{mark}  {r['id']:<24} {r['repo']}")
        for v in r["variants"]:
            flag = "+" if v["dtype"] not in r["recorded"] else " "
            dec = "" if v["dtype"] in r["declared"] else "   ← weights_release 里没有"
            print(f"      {flag} {v['dtype']:<10} {v['repo']}{dec}")
    sib_rows = [r for r in rows if r["siblings"] and not r["variants"]]
    if sib_rows:
        print(f"\n--- 另有 {len(sib_rows)} 份：官方量化仓挂在**兄弟 checkpoint** 下，"
              f"量化的不是本记录跟的这一份权重 ---")
        print("    结构相同、参数量相同，所以逐位对账会通过 —— 收进来错误完全隐形。")
        print("    只列不收，怎么记由人定。")
        for r in sib_rows:
            print(f"    {r['id']:<22} 本记录跟 {r['repo']}")
            for v in r["siblings"]:
                print(f"        {v['dtype']:<10} {v['repo']}")
    # ── 翻完布尔别忘了散文 ──────────────────────────────────
    # **这一段是「列」不是「闸」，有意的。** 它靠关键词找嫌疑句，而关键词判不出
    # 一句话是否过期 —— 做成 CI 闸就会每次红、每次该忽略，然后人学会不读告警
    # （主会话的 check_pending_claims 刚栽过这个形状）。放在取数脚本里按需跑，
    # 看见了自己判，不占闸口的信用。
    #
    # 为什么需要它：翻 weights_release_scope 的布尔是字段操作，而同一件事还写在
    # open_questions / lineage.*.benefit / notes 的散文里 —— 工具按字段走，扫不到。
    # 本轮栽过两次：ltx-2.3 的 lineage benefit 整段说反（v2-ltx 发现），
    # 以及五份记录的 open_questions 还写着「本仓没有取它的清单」。
    # **review[] 不扫**：那里的同一句话是当时的记录，就该停在当时。
    stale = re.compile(r"没取过|没有取过|没有取它|连清单都没取|未取数|没取数"
                       r"|没进精度表|没有并进|两个布尔都是假|没有实测依据")
    suspects = []
    for r, doc in zip(rows, docs):
        if not (((doc.get("precision") or {}).get("quantization") or {})
                .get("official_variants")):
            continue
        for field in ("open_questions", "notes", "lineage"):
            for where, text in _strings(doc.get(field), field):
                if stale.search(text):
                    suspects.append((r["id"], where, stale.search(text).group(0)))
    if suspects:
        print(f"\n--- {len(suspects)} 处散文可能跟不上布尔（**列，不是闸**：关键词判不出"
              f"一句话是否过期，自己看）---")
        for mid, where, hit in suspects:
            print(f"    {mid:22} {where}: 「{hit}」")

    if args.json:
        json.dump(rows, open(args.json, "w"), ensure_ascii=False, indent=1)
        print(f"\n写了 {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
