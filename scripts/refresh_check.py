#!/usr/bin/env python3
"""双周刷新：哪些记录的上游变了、变的是权重还是结构。

对每个要追的 HF 仓库，比本地 facts/<id>/fetched.json 钉住的 rev 与 HF 当前 sha。
sha 没变就到此为止（一次 API 请求）；变了再取新 sha 下的 index.json 张量名与 config.json，
和本地比：

  unchanged      sha 相同
  weights-only   sha 变了，张量名集合与 config（除元字段外）都相同 —— 权重刷新，
                 不必重走算子评估。这正是 README 里「先比张量名集合」那条判据
  config-changed 张量名相同、config 有字段变了 —— 结构可能没变，但要人看是哪些字段
  names-changed  张量名集合变了 —— 结构变了，要重新取数、重走审计
  no-baseline    本地没有可比的名字清单（见下），只能报 sha 变了
  not-fetched    名册里有、facts/ 下没有（待录的旗舰）
  error          取数失败

**本地名字清单的来源**：`names.json` 优先，没有时才用 `shapes.json` 的键。
零差异版本是 `--names-only` 取的，它的 `shapes.json` 是空字典 —— 拿空的 shapes 去比，
会把「本地没有清单」读成「本地一个张量都没有」，于是报出整套张量都是新增的。
两者都为空时报 no-baseline，不比。

只读：不写 facts/，不改记录。要更新就照输出里的命令跑 fetch_hf.py。

用法:
  python3 scripts/refresh_check.py               # 记录 + 名册
  python3 scripts/refresh_check.py --orphans     # 另查 facts/ 下没有记录的目录（已删的零差异版本）
  python3 scripts/refresh_check.py --json        # 机器可读
退出码：0 无变化；1 有记录需要处理（weights-only 之外的变化，或取数失败）
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

# 需要人处理的状态。weights-only 不在里面：它的结论就是「什么都不用做」
ACTIONABLE = {"config-changed", "names-changed", "no-baseline", "error"}


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
    if sha == fetched["rev"]:
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
    return {**r, "status": "config-changed" if r["config_changed"] else "weights-only"}


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
