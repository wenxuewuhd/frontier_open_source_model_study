#!/usr/bin/env python3
"""只取「产出 KV 的那几类张量」的宽度与层覆盖，用来算 KV 缓存容量。

KV 容量的乘式是「每条目元素数 × 条目数 × 每元素字节」。第一个因子只能从权重里量 ——
config 里的 head_dim / index_head_dim 都不是缓存宽度（见下面第一、二条坑），
所以这里把产出 KV 的张量挑出来读头。

为什么不用 fetch_hf.py 的全量 shapes.json：万亿模型几百个分片、每个分片两次 Range
请求。要回答的只是几个宽度，没必要读全量。但也不能只读一个代表 ——

  * **层覆盖**（哪些层产出 KV）从 index.json 的 weight_map 就能答，不花额外请求；
  * **宽度必须逐层量**。每 token 的账是「宽度 × 源层数」，只读一层然后假设其余相同，
    等于把一个未验的因子乘进结论。所以这里读全部命中的张量，并检查同类内宽度
    是否一致；不一致就报出来，因为那时「宽度 × 层数」这个乘式本身就不成立。

三个坑，都在这个脚本的产出上踩过：

  * **index K 是 MQA 共享的。** 缓存宽度是产出张量 indexer.wk 的输出维，不是
    index_n_heads × index_head_dim —— 按后者算过一次，高估 32 倍。
  * **compressor.wkv 和 attn.wkv 不是一回事。** 前者产出全局压缩 KV，随上下文增长；
    后者产出滑窗局部 KV，每序列固定 sliding_window 个位置。把后者当成全局 KV
    逐 token 计费，是另一次 5.2 倍高估的一部分。所以两者分成两类，不合并正则 ——
    合并了就分不出哪个宽度属于哪个缓存。
  * **这份数据是定向抓取，不是清单。** 绝不写进 shapes.json，也不能拿去算参数量：
    缺张量的清单会让参数量静默少算，而空文件和「真的没有」在代码里长得一模一样。
    所以一类都没命中时这里报错退出，不写空结果。

条目宽度取张量的 shape[0]（safetensors 里 Linear 权重是 [out, in]）。不是二维的
张量这个假设就不成立，此时宽度记 None，会让同类「宽度一致」判定失败并报出来。

用法:
  python3 scripts/fetch_kv_widths.py --id deepseek-v41-flash \\
      --repo deepseek-ai/DeepSeek-V4.1-Flash

  # 别的族系张量名不同，按需追加（可重复）
  python3 scripts/fetch_kv_widths.py --id glm-5.3 --repo ... \\
      --pattern kv_b='\\.attn\\.kv_b_proj\\.weight$'
"""
import argparse
import json
import os
import re
import sys
import urllib.error
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_hf import (API, RESOLVE, _get, _get_json, _token_header,  # noqa: E402
                      _write_atomic, safetensors_header)

# 产出 KV 的几类张量。名字是 DeepSeek V4 一代的；别的族系用 --pattern 追加。
# 顺序即优先级：一个张量名归第一个匹配上的类。
CLASSES = {
    "global_kv": (r"\.attn\.compressor\.wkv\.weight$",
                  "全局压缩 KV 的条目；条目数随上下文增长，按 compress_ratio 折算到每 token"),
    "index_k": (r"\.attn\.indexer\.wk\.weight$",
                "index K 条目；在 index 头之间 MQA 共享，宽度就是这个张量的输出维"),
    "local_window": (r"\.attn\.wkv\.weight$",
                     "滑窗局部 KV；每序列固定 sliding_window 个位置，不随上下文增长"),
}

_LAYER = re.compile(r"(?:^|\.)(layers|mtp)\.(\d+)\.")


def label(name: str) -> str:
    """张量名 → 层标签。主干 layers.N 记 L<N>，MTP 块 mtp.N 记 MTP<N>。"""
    m = _LAYER.search(name)
    if not m:
        return name
    return ("MTP" if m.group(1) == "mtp" else "L") + m.group(2)


def _lsort(lbl: str):
    """主干在前、MTP 在后，各自按层号数值排 —— 字典序会把 L10 排到 L2 前面。"""
    digits = re.sub(r"\D", "", lbl)
    return (lbl.startswith("MTP"), int(digits) if digits else -1, lbl)


def classify_kv(name: str, compiled: list):
    for key, rx, _ in compiled:
        if rx.search(name):
            return key
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True, help="本仓模型 id，决定 facts 子目录名")
    ap.add_argument("--repo", required=True, help="HF 仓库，如 deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--facts-dir", default="facts")
    ap.add_argument("--rev", default=None, help="指定 commit；默认用仓库当前 sha，保证可复现")
    ap.add_argument("--pattern", action="append", default=[], metavar="KEY=REGEX",
                    help="追加一类张量；可重复")
    ap.add_argument("--workers", type=int, default=8, help="并发读分片头的线程数")
    args = ap.parse_args()

    compiled = [(k, re.compile(p), note) for k, (p, note) in CLASSES.items()]
    for spec in args.pattern:
        if "=" not in spec:
            print(f"  ! --pattern 要写成 KEY=REGEX，收到 {spec!r}", file=sys.stderr)
            return 2
        k, _, p = spec.partition("=")
        compiled.append((k, re.compile(p), "由 --pattern 追加"))

    out_dir = os.path.join(args.facts_dir, args.id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"==> {args.repo}", file=sys.stderr)
    info = _get_json(API + args.repo)
    rev = args.rev or info.get("sha") or "main"
    files = [s["rfilename"] for s in info.get("siblings", [])]

    idx = next((f for f in files if f.endswith(".safetensors.index.json")), None)
    if not idx:
        print(f"  ! {args.id}: 没有 safetensors index，定向取数要靠 weight_map 定位分片，"
              f"这里取不了。只能走 fetch_hf.py 读全量头。", file=sys.stderr)
        return 1
    wm = json.loads(_get(RESOLVE.format(repo=args.repo, rev=rev, path=idx),
                         _token_header())).get("weight_map") or {}
    if not wm:
        print(f"  ! {args.id}: weight_map 是空的，不写文件", file=sys.stderr)
        return 1

    hit = defaultdict(list)
    for name in wm:
        k = classify_kv(name, compiled)
        if k:
            hit[k].append(name)

    for key, _, _ in compiled:
        if not hit.get(key):
            print(f"  ! {args.id}: {key} 一个张量都没命中 —— 要么这一代没有这个部件，"
                  f"要么张量名变了。两种情况在结果里长得一样，别当成「宽度为 0」。",
                  file=sys.stderr)
    if not hit:
        print(f"  ! {args.id}: 所有类都为空，不写文件（空文件会被下游当成实测结果）",
              file=sys.stderr)
        return 1

    want = sorted(n for names in hit.values() for n in names)
    want_set = set(want)
    shards = sorted({wm[n] for n in want})
    total_shards = len({*wm.values()})
    print(f"  {args.id}: {len(wm)} 个张量名，命中 {len(want)} 个，"
          f"读 {len(shards)} 个分片头（全量是 {total_shards} 个）", file=sys.stderr)

    seen, failed = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(safetensors_header, args.repo, rev, p): p for p in shards}
        for fut in as_completed(futs):
            path = futs[fut]
            try:
                head = fut.result()
            except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
                failed.append(path)
                print(f"  ! {path} 读头失败: {e}", file=sys.stderr)
                continue
            for name, meta in head.items():
                if name != "__metadata__" and name in want_set:
                    seen[name] = {"dtype": meta.get("dtype"), "shape": meta.get("shape")}

    classes = {}
    for key, _, note in compiled:
        names = sorted(hit.get(key, []))
        if not names:
            continue
        got = {n: seen[n] for n in names if n in seen}
        widths = defaultdict(list)
        for n, m in got.items():
            shp = m.get("shape") or []
            widths[shp[0] if len(shp) == 2 else None].append(label(n))
        dts = defaultdict(int)
        for m in got.values():
            dts[m.get("dtype")] += 1
        uniform = len(widths) == 1 and None not in widths
        entry = next(iter(widths)) if uniform else None
        classes[key] = {
            "entry_elems": entry,
            "uniform": uniform,
            "layers": sorted((label(n) for n in names), key=_lsort),
            "tensors_matched": len(names),
            "tensors_read": len(got),
            "dtypes": dict(dts),
            "note": note,
        }
        if not uniform:
            classes[key]["widths"] = {
                str(w): sorted(ls, key=_lsort) for w, ls in widths.items()
            }
            print(f"  ! {args.id}: {key} 的条目宽度逐层不一致 —— "
                  f"每 token 的账不能按「宽度 × 层数」算。"
                  + "，".join(f"{w}：{len(ls)} 层" for w, ls in widths.items()),
                  file=sys.stderr)
        if len(got) < len(names):
            classes[key]["missing"] = sorted(set(names) - set(got))

    doc = {
        "repo": args.repo,
        "rev": rev,
        "method": (f"按 model.safetensors.index.json 的 weight_map 定位分片，"
                   f"只读命中分片的头（Range 请求），取产出 KV 的 {len(classes)} 类张量；"
                   f"读了 {len(shards)}/{total_shards} 个分片，不是全量清单"),
        "caveat": "定向抓取：只能回答 KV 条目宽度与层覆盖，不能用于参数量统计，"
                  "也不要并入 shapes.json",
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "failed_shards": failed,
        "classes": classes,
        "tensors": {n: seen[n] for n in want if n in seen},
    }
    _write_atomic(os.path.join(out_dir, "kv_widths.json"), doc)

    for key, c in classes.items():
        w = c["entry_elems"]
        print(f"    {key:14s} 条目 {(str(w) if w is not None else '不一致'):>8} 元素  "
              f"{len(c['layers']):>2} 层  {'/'.join(d for d in c['dtypes'] if d)}",
              file=sys.stderr)
    print(f"==> {out_dir}/kv_widths.json  rev={rev[:12]}", file=sys.stderr)
    if failed:
        print(f"! {len(failed)} 个分片读失败，宽度可能缺层 —— 缺层会让「宽度 × 层数」"
              f"算少。重跑以补全。", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
