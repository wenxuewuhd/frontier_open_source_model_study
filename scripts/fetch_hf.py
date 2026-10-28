#!/usr/bin/env python3
"""取数：把一个 HF 仓库的事实抓到 facts/<id>/ 下。

产出四个文件，与既有样例的 facts/ 保持一致：
  cfg.json       config.json 原文
  shapes.json    逐 shard 读 safetensors 头得到的张量清单（名字 -> shape/dtype）
  repoinfo.json  HF 元数据（sha / lastModified / tags / downloads）
  fetched.json   取数记录：时间、sha、每个文件的来源 URL

只读 safetensors 的头，不下权重 —— 一个 600B 模型也只有几 MB 的网络流量。

用法:
  python3 scripts/fetch_hf.py deepseek-ai/DeepSeek-V4.1-Flash --id deepseek-v41-flash
  python3 scripts/fetch_hf.py <repo> --id <id> --facts-dir facts   # 默认 facts/
"""
import argparse
import json
import os
import struct
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

API = "https://huggingface.co/api/models/"
RESOLVE = "https://huggingface.co/{repo}/resolve/{rev}/{path}"
UA = {"User-Agent": "frontier-model-study/0.1"}


def _get(url: str, headers=None, timeout=60) -> bytes:
    h = dict(UA)
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _get_json(url: str):
    return json.loads(_get(url))


def _token_header():
    """有 token 就带上，没有也能取公开仓。"""
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def safetensors_header(repo: str, rev: str, path: str) -> dict:
    """只取 safetensors 的头。

    格式：前 8 字节是小端 u64 头长度，紧接着是该长度的 JSON。
    用 Range 请求两次，避免拉整个 shard。
    """
    url = RESOLVE.format(repo=repo, rev=rev, path=path)
    hdr = dict(_token_header())
    hdr["Range"] = "bytes=0-7"
    raw = _get(url, hdr)
    if len(raw) < 8:
        raise RuntimeError(f"{path}: 头长度字段读不全，拿到 {len(raw)} 字节")
    n = struct.unpack("<Q", raw[:8])[0]
    hdr["Range"] = f"bytes=8-{8 + n - 1}"
    return json.loads(_get(url, hdr))


def collect_shapes(repo: str, rev: str, files: list) -> dict:
    """遍历所有 safetensors shard，汇总张量清单。"""
    shards = sorted(f for f in files if f.endswith(".safetensors"))
    if not shards:
        print("  ! 仓库里没有 .safetensors，shapes 留空", file=sys.stderr)
        return {}
    out = {}
    for i, path in enumerate(shards, 1):
        print(f"  [{i}/{len(shards)}] {path}", file=sys.stderr)
        try:
            head = safetensors_header(repo, rev, path)
        except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
            print(f"  ! {path} 读头失败: {e}", file=sys.stderr)
            continue
        for name, meta in head.items():
            if name == "__metadata__":
                continue
            out[name] = {"shape": meta.get("shape"), "dtype": meta.get("dtype")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="HF 仓库，如 deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--id", required=True, help="本仓模型 id，决定 facts 子目录名")
    ap.add_argument("--facts-dir", default="facts")
    ap.add_argument("--rev", default=None, help="指定 commit；默认用仓库当前 sha，保证可复现")
    ap.add_argument("--skip-shapes", action="store_true", help="只取 config 与元数据")
    args = ap.parse_args()

    out_dir = os.path.join(args.facts_dir, args.id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"==> {args.repo}", file=sys.stderr)
    info = _get_json(API + args.repo)
    rev = args.rev or info.get("sha") or "main"
    files = [s["rfilename"] for s in info.get("siblings", [])]

    cfg = {}
    if "config.json" in files:
        cfg = json.loads(_get(RESOLVE.format(repo=args.repo, rev=rev, path="config.json"),
                              _token_header()))
    else:
        print("  ! 没有 config.json", file=sys.stderr)

    shapes = {} if args.skip_shapes else collect_shapes(args.repo, rev, files)

    for name, obj in (("repoinfo.json", info), ("cfg.json", cfg), ("shapes.json", shapes)):
        with open(os.path.join(out_dir, name), "w") as f:
            json.dump(obj, f, ensure_ascii=False, indent=1)

    record = {
        "repo": args.repo,
        "rev": rev,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tensor_count": len(shapes),
        "shard_count": sum(1 for f in files if f.endswith(".safetensors")),
        "config_url": RESOLVE.format(repo=args.repo, rev=rev, path="config.json"),
    }
    with open(os.path.join(out_dir, "fetched.json"), "w") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)

    print(f"==> {out_dir}: config {'有' if cfg else '无'}, {len(shapes)} 个张量, rev={rev[:12]}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
