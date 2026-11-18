#!/usr/bin/env python3
"""双周刷新：哪些记录的上游变了、变的是权重、结构，还是模板一类权重外的附属文件。

对每个要追的 HF 仓库，比本地 facts/<id>/fetched.json 钉住的 rev 与 HF 当前 sha。
sha 没变就到此为止（一次 API 请求）；变了再取新 sha 下的 index.json 张量名与 config.json，
和本地比：

  unchanged      sha 相同（本地只钉了短 sha 时按前缀相同，见下）
  sidecar-only   sha 变了，张量名集合、config（除元字段外）与 safetensors tally 都相同 ——
                 改的是聊天模板 / 分词器配置 / README 这类权重外的附属文件，不必重取
  weights-only   名字与 config 相同而 tally 变了 —— **换了一版权重**。记录里所有实测自
                 权重的数（参数量、逐子系统精度、缓存宽度）都可能已过期，要重取清单重跑
                 derive。这是本脚本最该报的一条
  config-changed 张量名相同、config 有字段变了 —— 结构可能没变，但要人看是哪些字段
  names-changed  张量名集合变了 —— 结构变了，要重新取数、重走审计
  no-baseline    本地没有可比的名字清单（见下），只能报 sha 变了
  not-fetched    名册里有、facts/ 下没有（待录的旗舰）
  error          取数失败

**为什么要比 tally**：`remote_names()` 读的是 index.json 里 `weight_map` 的**键**，
不看形状、不看 dtype、不看文件 size、不看 LFS oid。所以「名字集合同 + config 同」
这一步**没有任何关于权重内容的证据**，而它原本直接产出 `weights-only`：通义 11-07
那次十个仓只改了 `chat_template.jinja` 与 `tokenizer_config.json`（权重 0/10、config 0/10），
与「厂商真换一版权重而名字和 config 不变」在这个判据下输出一模一样 ——
最该报的那件事和最无害的那件事同一个标签。`safetensors.total` 把两者分开，
而且不多花一次请求：HF 逐仓元数据里已经带这个字段，本地存在 facts/<id>/repoinfo.json。
两侧有一侧取不到 tally（9 份 facts/ 的 repoinfo 没有这个字段，多是 .pt 仓或零差异档）
时**按 weights-only 报并在 why 里写明无法区分** —— 宁可多报一次，不把「判不出来」
混进「什么都不用做」。

**短 sha**：`fetched.json` 的 rev 是取数时 `--rev` 原样记下的，facts/ 里有几份钉的是
12 位短 sha，与 HF 的 40 位直接比**永远不等**，于是跳过 unchanged 一路落进「sha 变了」
那几档（MiniMax 三份就是这么长期假阳的）。`sha_equal()` 按较短的一方取前缀比，
并且长度不等时一律把这件事写进 why —— 下一份短 sha 不会再静默。

**本地名字清单的来源**：`names.json` 优先，没有时才用 `shapes.json` 的键。
零差异版本是 `--names-only` 取的，它的 `shapes.json` 是空字典 —— 拿空的 shapes 去比，
会把「本地没有清单」读成「本地一个张量都没有」，于是报出整套张量都是新增的。
两者都为空时报 no-baseline，不比。

**这道检查锚在哪一侧**（model-audit skill 规则 11）：基线在**内侧** —— 本地 `facts/`
的取数快照（rev / 名字清单 / cfg.json / repoinfo 的 tally），比的是 HF 当前 head。
所以它查得出**上游动了什么**，查不出**我们当初漏取了什么**：名册与记录之外的仓
（厂商另发的量化变体、同族的其它档）压根不进比对，记录声明了而清单里没有的东西
也不是它的事（那是 validate 的 `weights_release` 那道闸，方向同样是单向的）。

**它绿了还不能说明什么**（规则 13）：
  - `unchanged` 只说明 HF 的 head sha 与我们钉的 rev 相同，**不说明记录里的数对** ——
    取数当时读错、算错、填错，sha 一个字都不会变。这个脚本全绿不是记录正确的证据
  - `sidecar-only` 只说明名字集合、config 与 tally 相同，**不说明逐张量数值没变** ——
    同形同 dtype 换一版权重，tally 分毫不动。要钉死得比 LFS oid 或逐文件 blobId
    （`?blobs=true`），那是另一次取数，不是这道闸的延伸

只读：不写 facts/，不改记录。要更新就照输出里的命令跑 fetch_hf.py。

用法:
  python3 scripts/refresh_check.py               # 记录 + 名册
  python3 scripts/refresh_check.py --orphans     # 另查 facts/ 下没有记录的目录（已删的零差异版本）
  python3 scripts/refresh_check.py --json        # 机器可读
退出码：0 无变化或只有 sidecar-only；1 有记录需要处理（换了权重 / 结构 / config，或取数失败）
"""
import argparse
import glob
import json
import os
import sys
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_hf import API, RESOLVE, _get, _get_json, _token_header  # noqa: E402
from lib import yamlio  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTS = os.path.join(ROOT, "facts")

# config 里与结构无关的元字段。只差这些的两份 config 视为相同（R1 与 V3 就只差 transformers_version）
CONFIG_META = {"transformers_version", "_name_or_path"}

# 需要人处理的状态。**weights-only 在里面**：它现在的意思是 tally 变了、上游换了一版权重，
# 于是记录里所有实测自权重的数都可能已过期 —— 那是最该报的一条，不是「什么都不用做」。
# 唯一「sha 变了而什么都不用做」的是 sidecar-only（模板 / 分词器 / README）。
ACTIONABLE = {"weights-only", "config-changed", "names-changed", "no-baseline", "error"}


def targets(orphans: bool, facts: str = FACTS) -> list:
    """(id, repo, 来源)。记录优先；名册里的旗舰没有记录时也列出来。"""
    out, seen = [], set()
    for path in sorted(glob.glob(os.path.join(ROOT, "data", "models", "*", "*.yaml"))):
        doc = yamlio.load(path)
        repo = (doc.get("hf") or {}).get("repo")
        if repo:
            out.append((doc["id"], repo, "record"))
            seen.add(doc["id"])
    tracking = yamlio.load(os.path.join(ROOT, "data", "tracking.yaml"))
    for fam in tracking.get("families") or []:
        mid = fam.get("model_id")
        if mid and mid in seen:
            continue
        out.append((mid or fam["flagship"], fam["hf_repo"], "tracking"))
        if mid:
            seen.add(mid)
    if orphans:
        for d in sorted(glob.glob(os.path.join(facts, "*", "fetched.json"))):
            mid = os.path.basename(os.path.dirname(d))
            if mid not in seen:
                out.append((mid, json.load(open(d))["repo"], "orphan"))
    return out


def _read(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def local_names(fdir: str):
    """(名字集合, 来源文件) 或 (None, 原因)。空清单当作没有，不当作零个张量。"""
    names = _read(os.path.join(fdir, "names.json"))
    if names:
        return set(names), "names.json"
    shapes = _read(os.path.join(fdir, "shapes.json"))
    if shapes:
        return set(shapes), "shapes.json"
    return None, "names.json 不存在、shapes.json 为空"


def remote_names(repo: str, rev: str, files: list):
    idx = next((f for f in files if f.endswith(".safetensors.index.json")), None)
    if not idx:
        return None
    doc = json.loads(_get(RESOLVE.format(repo=repo, rev=rev, path=idx), _token_header()))
    return set(doc.get("weight_map") or {})


def config_diff(old: dict, new: dict) -> list:
    keys = (set(old) | set(new)) - CONFIG_META
    return sorted(k for k in keys if old.get(k) != new.get(k))


def tally(info: dict):
    """HF 元数据里的 safetensors 元素总数，取不到返回 None。

    本地 facts/<id>/repoinfo.json 存的是同一个端点的同一个字段，所以两侧同口径。
    它只是元素数之和：同形同 dtype 换一版权重它不动（见模块 docstring 末尾）。
    """
    total = ((info or {}).get("safetensors") or {}).get("total")
    return total if isinstance(total, int) else None


def sha_equal(local: str, remote: str):
    """(是不是同一个提交, 口径说明或 None)。

    长度不等时按较短的一方取前缀比 —— `fetched.json` 的 rev 是取数时 `--rev` 原样记下的，
    facts/ 里有几份是 12 位短 sha。**长度不等这件事一定要报出去**（第二个返回值），
    否则「短 sha 与 40 位 sha 恒不等」会静默落进某个「sha 变了」的状态，
    而那正是 MiniMax 三份长期假阳的成因。
    """
    if not local or not remote:
        return False, f"没法比 sha：本地 {local!r}、远端 {remote!r}"
    if len(local) == len(remote):
        return local == remote, None
    short, long_ = sorted((local, remote), key=len)
    same = long_.startswith(short)
    return same, (f"本地 rev {len(local)} 位、远端 sha {len(remote)} 位，"
                  f"按前 {len(short)} 位比对{'相同' if same else '不同'}"
                  "；建议把 fetched.json 的 rev 补成全长")


def _note(r: dict, text: str) -> dict:
    """往 r['why'] 追加一句，不覆盖已经写进去的（短 sha 那句得活到最后）。"""
    r["why"] = f"{r['why']}；{text}" if r.get("why") else text
    return r


def check(mid: str, repo: str, facts: str = FACTS) -> dict:
    fdir = os.path.join(facts, mid)
    fetched = _read(os.path.join(fdir, "fetched.json"))
    r = {"id": mid, "repo": repo}
    try:
        info = _get_json(API + repo)
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        return {**r, "status": "error", "why": f"取 HF 元数据失败: {e}"}
    sha = info.get("sha")
    r["remote_sha"] = sha
    if not fetched:
        return {**r, "status": "not-fetched", "why": "facts/ 下没有取数记录"}
    r["local_rev"] = fetched["rev"]
    if fetched.get("repo") != repo:
        # 记录指向的仓与取数时的仓不同，比 sha 没有意义
        return {**r, "status": "error",
                "why": f"facts 取自 {fetched.get('repo')}，记录写的是 {repo}"}
    same_commit, rev_note = sha_equal(fetched["rev"], sha)
    if rev_note:
        _note(r, rev_note)
    if same_commit:
        return {**r, "status": "unchanged"}

    r["last_modified"] = info.get("lastModified")
    files = [s["rfilename"] for s in info.get("siblings", [])]
    try:
        new_names = remote_names(repo, sha, files)
        new_cfg = (json.loads(_get(RESOLVE.format(repo=repo, rev=sha, path="config.json"),
                                   _token_header()))
                   if "config.json" in files else {})
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        return {**r, "status": "error", "why": f"取新 sha 下的清单失败: {e}"}

    r["config_changed"] = config_diff(_read(os.path.join(fdir, "cfg.json")) or {}, new_cfg)
    old_names, src = local_names(fdir)
    if old_names is None or new_names is None:
        why = src if old_names is None else "新 sha 下没有 safetensors index"
        return {**r, "status": "no-baseline", "why": f"sha 变了，但没法比张量名：{why}"}
    r["names_source"] = src
    added, removed = sorted(new_names - old_names), sorted(old_names - new_names)
    if added or removed:
        return {**r, "status": "names-changed", "added": added, "removed": removed}
    if r["config_changed"]:
        return {**r, "status": "config-changed"}
    # 名字与 config 都相同 —— 到这一步还没有一个字节的权重证据，tally 是分水岭
    r["local_tally"] = tally(_read(os.path.join(fdir, "repoinfo.json")))
    r["remote_tally"] = tally(info)
    if r["local_tally"] is None or r["remote_tally"] is None:
        side = "本地 repoinfo.json" if r["local_tally"] is None else "HF 元数据"
        return {**_note(r, f"{side} 没有 safetensors.total，分不清是换了权重还是只改了"
                           "附属文件 —— 按换权重报"), "status": "weights-only"}
    if r["local_tally"] == r["remote_tally"]:
        return {**_note(r, f"safetensors tally 未变（{r['local_tally']:,}），"
                           "变的是权重外的附属文件"), "status": "sidecar-only"}
    return {**_note(r, f"safetensors tally {r['local_tally']:,} -> {r['remote_tally']:,}"),
            "status": "weights-only"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orphans", action="store_true",
                    help="另查 facts/ 下没有记录的目录")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--facts-dir", default=FACTS, help="默认 facts/；换一个目录可以拿假基线验各条路径")
    args = ap.parse_args()

    results = [dict(check(mid, repo, args.facts_dir), source=src)
               for mid, repo, src in targets(args.orphans, args.facts_dir)]
    if args.json:
        json.dump(results, sys.stdout, ensure_ascii=False, indent=1)
        print()
    else:
        for r in results:
            line = f"{r['status']:15} {r['id']:26} {r['repo']}"
            if r.get("local_rev") and r["status"] != "unchanged":
                line += f"  {r['local_rev'][:12]} -> {(r.get('remote_sha') or '?')[:12]}"
            print(line)
            if r.get("why"):
                print(f"{'':16}{r['why']}")
            if r.get("config_changed"):
                print(f"{'':16}config 变了的字段: {', '.join(r['config_changed'])}")
            for k, label in (("added", "+"), ("removed", "-")):
                if r.get(k):
                    print(f"{'':16}{label}{len(r[k])} 个张量名，如 {', '.join(r[k][:3])}")
            if r["status"] in ACTIONABLE - {"error"}:
                print(f"{'':16}重新取数: .venv/bin/python scripts/fetch_hf.py {r['repo']} "
                      f"--id {r['id']} --rev {r['remote_sha']}")
    sys.exit(1 if any(r["status"] in ACTIONABLE for r in results) else 0)


if __name__ == "__main__":
    main()
