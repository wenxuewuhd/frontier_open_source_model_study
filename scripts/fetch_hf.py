#!/usr/bin/env python3
"""取数：把一个 HF 仓库的事实抓到 facts/<id>/ 下。

产出的文件，与既有样例的 facts/ 保持一致：
  cfg.json       **顶层** config.json 原文。没有顶层 config 的仓是 {} —— 含义不变，
                 48 份语言模型记录全靠它，不要改成别的形状
  cfg_components.json  组件目录下的 config（`transformer/` `vae/` `audio_vae/` …）
                 与 model_index.json。生成类仓的配置散在组件里，顶层那份可能根本不存在
  shapes.json    逐 shard 读 safetensors 头得到的张量清单（名字 -> shape/dtype）
  shard_map.json 每个张量名来自哪个分片。**合并是否可信全看这个**
  repoinfo.json  HF 元数据（sha / lastModified / tags / downloads）
  fetched.json   取数记录：时间、sha、每个文件的来源 URL

**生成类仓有四种布局**（2026-10-30 实扫 12 个视频仓）：
  顶层 config 就够     wan2.1-t2v-14b / wan2.2-ti2v-5b / hunyuanvideo-13b / hunyuanvideo-1.5
  model_index + 组件   minimax-h3（17 份 config）/ ltx-2 / cogvideox-1.5-5b / stepvideo-t2v-30b
  无 model_index 有组件 wan2.2-t2v-a14b / magi-2-preview
  完全没有 config      ltx-2.3 / sana-video-2.0-5b
所以不能假设「生成类 = Diffusers 布局 = 没有顶层 config」，四种都得认。

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
from concurrent.futures import ThreadPoolExecutor, as_completed
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


def pick_shards(files: list, include: list) -> list:
    """选出要读的 safetensors，并在「合并出一个不存在的模型」之前拦住。

    原来这里是 `sorted(f for f in files if f.endswith(".safetensors"))` —— 把仓里
    **所有** safetensors 当成同一个模型的分片。分片仓确实如此，但不是所有仓都是分片仓：

      Lightricks/LTX-2.3 的 8 个 .safetensors 是 8 个**不同的模型**
      （`ltx-2.3-22b-dev` / `-distilled` / 两个 LoRA / spatial-upscaler …），
      不是一个模型的 8 片。

    而 dev 和 distilled 的**形状完全一样**，只是权重值不同 —— 所以靠「形状冲突」
    检测不出来，合并出来的是一个不存在的模型，而且每个数看着都合理。

    判据按结构不按名字：**有 index.json 就按它的 weight_map 走**（分片仓的权威来源）；
    没有 index 而顶层有多个 .safetensors，就是歧义，**拒绝猜**，让人用 --include 点名。
    """
    st = sorted(f for f in files if f.endswith(".safetensors"))
    if include:
        import fnmatch
        hit = [f for f in st if any(fnmatch.fnmatch(f, g) for g in include)]
        if not hit:
            raise SystemExit(f"! --include {include} 一个 .safetensors 都没匹配上。仓里有：\n  "
                             + "\n  ".join(st))
        return hit
    idx = [f for f in files if f.endswith(".safetensors.index.json")]
    if idx:
        return st
    top = [f for f in st if "/" not in f]
    if len(top) > 1:
        raise SystemExit(
            "! 没有 index.json，而顶层有多个 .safetensors —— 无法判断它们是一个模型的分片\n"
            "  还是几个独立模型。合并会静默产出一个不存在的模型（形状相同时连冲突都报不出来）。\n"
            "  用 --include 点名要哪些，例如 --include 'ltx-2.3-22b-dev.safetensors'。\n"
            "  仓里有：\n  " + "\n  ".join(st))
    return st


def collect_shapes(repo: str, rev: str, files: list, workers: int = 8, include: list = None):
    """遍历选中的 safetensors shard，汇总张量清单。

    万亿参数模型有几百个分片（每个分片两次 Range 请求），串行要十几分钟，
    所以并发取。并发数保持适中，不要把 HF 打爆。
    任何一个分片读失败都要报出来并记进 failed —— 清单缺分片会让参数量
    静默少算，derive.py 的完整性检查靠层索引连续性兜底，但这里先报一次。

    **同名张量不再静默覆盖。** 原来是 `out[name] = ...`，后读到的盖掉先读到的，
    而分片是并发完成的，所以盖掉谁还不确定 —— 同一个仓两次取数可能得到不同清单。
    现在：形状与 dtype 相同记一条 duplicate（多 index 仓里同一份权重出现在两个
    index 里是正常的），不同则直接报错退出，那是两个不同的张量撞了名字。
    """
    shards = pick_shards(files, include or [])
    if not shards:
        print("  ! 仓库里没有 .safetensors，shapes 留空", file=sys.stderr)
        print("    注意：空清单会让 derive.py 把参数量静默算成 0，而不是报错。", file=sys.stderr)
        return {}, [], [], {}

    out, failed, dupes, shard_map, done = {}, [], [], {}, 0
    conflicts = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(safetensors_header, repo, rev, p): p for p in shards}
        for fut in as_completed(futs):
            path = futs[fut]
            done += 1
            try:
                head = fut.result()
            except (urllib.error.HTTPError, urllib.error.URLError, RuntimeError) as e:
                failed.append(path)
                print(f"  ! [{done}/{len(shards)}] {path} 读头失败: {e}", file=sys.stderr)
                continue
            for name, meta in head.items():
                if name == "__metadata__":
                    continue
                cur = {"shape": meta.get("shape"), "dtype": meta.get("dtype")}
                if name in out:
                    if out[name] == cur:
                        dupes.append({"tensor": name, "shards": [shard_map[name], path]})
                    else:
                        conflicts.append({"tensor": name,
                                          "a": {"shard": shard_map[name], **out[name]},
                                          "b": {"shard": path, **cur}})
                    continue
                out[name] = cur
                shard_map[name] = path
            if done % 20 == 0 or done == len(shards):
                print(f"  [{done}/{len(shards)}] {len(out)} 个张量", file=sys.stderr)

    if conflicts:
        print(f"! {len(conflicts)} 个张量名在两个分片里形状或 dtype 不同 —— "
              "这不是分片仓，合并出来的清单不对应任何真实模型。前 5 条：", file=sys.stderr)
        for c in conflicts[:5]:
            print(f"    {c['tensor']}\n      {c['a']['shard']}: {c['a']['shape']} {c['a']['dtype']}"
                  f"\n      {c['b']['shard']}: {c['b']['shape']} {c['b']['dtype']}", file=sys.stderr)
        raise SystemExit("  用 --include 把范围收到单个模型上再取。")
    if dupes:
        print(f"  {len(dupes)} 个张量在多个分片里有相同副本（形状与 dtype 一致），"
              "已记进 shard_map 的 duplicates", file=sys.stderr)
    return out, failed, dupes, shard_map


# tokenizer / 预处理器的 config 不是模型配置，别混进来 —— 它们属于文本编码器那一侧，
# 而文本编码器在本仓是单独一份记录或一个共用组件。
_NOT_MODEL_CFG = ("tokenizer_config.json", "preprocessor_config.json",
                  "feature_extractor_config.json", "generation_config.json")


def fetch_configs(repo: str, rev: str, files: list):
    """取顶层 config 与各组件 config。返回 (root_cfg, components)。

    为什么不是「有没有顶层 config.json」一个判断就够：实扫 12 个视频仓，布局有四种
    （见模块头注释）。`wan2.2-t2v-a14b` 没有顶层 config 但有两份组件 config；
    `hunyuanvideo-1.5` 顶层有、组件下另有 16 份；`ltx-2.3` 一份都没有。

    **root_cfg 的含义保持不变** —— 它就是顶层 config.json 原文，没有就是 {}。
    48 份语言模型记录的 cfg.json 全是这个形状，改了会全体失配。组件 config 另存一个文件。
    """
    hdr = _token_header()

    def _try(path):
        try:
            return json.loads(_get(RESOLVE.format(repo=repo, rev=rev, path=path), hdr))
        except (urllib.error.HTTPError, urllib.error.URLError, ValueError) as e:
            print(f"  ! {path} 取不到: {e}", file=sys.stderr)
            return None

    root = {}
    if "config.json" in files:
        root = _try("config.json") or {}
    else:
        print("  ! 没有顶层 config.json", file=sys.stderr)

    comps = {}
    if "model_index.json" in files:
        mi = _try("model_index.json")
        if mi is not None:
            comps["_model_index"] = mi

    for f in sorted(files):
        if not f.endswith("config.json") or f == "config.json":
            continue
        if any(f.endswith(x) for x in _NOT_MODEL_CFG):
            continue
        if "/" not in f:
            continue
        d = _try(f)
        if d is not None:
            comps[f.rsplit("/", 1)[0]] = d

    if comps:
        names = [k for k in comps if k != "_model_index"]
        print(f"  组件 config {len(names)} 份: {', '.join(names[:8])}"
              + (" …" if len(names) > 8 else ""), file=sys.stderr)
    elif not root:
        print("  ! 顶层与组件都没有 config —— 这一份的 Gate B 只剩清单一路，"
              "记录里要写明「config 侧不存在」而不是「未取」", file=sys.stderr)
    return root, comps


def collect_names(repo: str, rev: str, files: list) -> dict:
    """只取张量名清单，不取形状。

    safetensors 的 index.json 里有 weight_map（张量名 -> 分片），一次请求就能拿全。
    代价是没有 shape/dtype，所以参数量算不了；但「某个张量存不存在」这类结构问题
    它就能答 —— 而这类问题恰恰是跨版本比对最常问的（有没有 KV 上投影、
    有没有 attn_sink、indexer 落在哪些层）。

    万亿模型跑一次 collect_shapes 要读几百个分片头，十几个历史版本就是上千次请求。
    结构问题用这个模式问，参数量问题才值得去读全量头。
    """
    idx = next((f for f in files if f.endswith(".safetensors.index.json")), None)
    if not idx:
        print("  ! 没有 safetensors index，名字清单留空", file=sys.stderr)
        return {}
    doc = json.loads(_get(RESOLVE.format(repo=repo, rev=rev, path=idx), _token_header()))
    wm = doc.get("weight_map") or {}
    print(f"  {len(wm)} 个张量名（来自 {idx}）", file=sys.stderr)
    return sorted(wm)


def _write_atomic(path: str, obj) -> None:
    """先写临时文件再 rename。

    万亿模型的 shapes.json 有几十 MB，直接覆写时如果中断或被并发读到，
    留下的是半个 JSON —— 下游解析失败还算好的，更坏的是被当成「清单不完整」处理。
    rename 在同一文件系统内是原子的，读到的要么是旧的完整文件要么是新的。
    """
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="HF 仓库，如 deepseek-ai/DeepSeek-V4.1-Flash")
    ap.add_argument("--id", required=True, help="本仓模型 id，决定 facts 子目录名")
    ap.add_argument("--facts-dir", default="facts")
    ap.add_argument("--rev", default=None, help="指定 commit；默认用仓库当前 sha，保证可复现")
    ap.add_argument("--skip-shapes", action="store_true", help="只取 config 与元数据")
    ap.add_argument("--names-only", action="store_true",
                    help="从 index.json 取张量名清单（无形状，算不了参数量，但能答结构问题）")
    ap.add_argument("--workers", type=int, default=8, help="并发读分片头的线程数")
    ap.add_argument("--include", action="append", default=[], metavar="GLOB",
                    help="只读匹配的 .safetensors（可多次）。没有 index.json 而顶层有多个 "
                         "safetensors 时必须给 —— 那种仓里它们往往是几个独立模型，"
                         "合并会产出一个不存在的模型")
    args = ap.parse_args()

    out_dir = os.path.join(args.facts_dir, args.id)
    os.makedirs(out_dir, exist_ok=True)

    print(f"==> {args.repo}", file=sys.stderr)
    info = _get_json(API + args.repo)
    rev = args.rev or info.get("sha") or "main"
    files = [s["rfilename"] for s in info.get("siblings", [])]

    cfg, comps = fetch_configs(args.repo, rev, files)

    names = []
    dupes, shard_map = [], {}
    if args.names_only or args.skip_shapes:
        shapes, failed = {}, []
        if args.names_only:
            names = collect_names(args.repo, rev, files)
    else:
        shapes, failed, dupes, shard_map = collect_shapes(
            args.repo, rev, files, args.workers, args.include)

    for name, obj in (("repoinfo.json", info), ("cfg.json", cfg), ("shapes.json", shapes)):
        _write_atomic(os.path.join(out_dir, name), obj)
    # 组件 config 与分片归属另存。不并进 cfg.json / shapes.json —— 那两个的形状
    # 是 48 份已有记录的契约，加字段等于让全体失配。
    if comps:
        _write_atomic(os.path.join(out_dir, "cfg_components.json"), comps)
    if shard_map:
        _write_atomic(os.path.join(out_dir, "shard_map.json"),
                      {"by_tensor": shard_map, "duplicates": dupes})
    # 名字清单单独存。不混进 shapes.json：形状为空的清单会让 derive.py
    # 把参数量静默算成 0，而不是报「清单不完整」。
    if names:
        _write_atomic(os.path.join(out_dir, "names.json"), names)

    record = {
        "repo": args.repo,
        "rev": rev,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "tensor_count": len(shapes),
        "name_count": len(names),
        "shard_count": sum(1 for f in files if f.endswith(".safetensors")),
        "config_url": RESOLVE.format(repo=args.repo, rev=rev, path="config.json"),
        "has_root_config": bool(cfg),
        "component_configs": sorted(k for k in comps if k != "_model_index"),
        "has_model_index": "_model_index" in comps,
        "include_globs": args.include,
        "duplicate_tensors": len(dupes),
        "failed_shards": failed,
    }
    _write_atomic(os.path.join(out_dir, "fetched.json"), record)

    ncomp = len([k for k in comps if k != "_model_index"])
    print(f"==> {out_dir}: 顶层 config {'有' if cfg else '无'}, 组件 config {ncomp} 份, "
          f"{len(shapes)} 个张量, rev={rev[:12]}", file=sys.stderr)
    if not cfg and not comps:
        print("! 这个仓没有任何 config —— Gate B 的双路对账只剩清单一路。"
              "记录里要写明「config 侧不存在」，不要写成「未取」。", file=sys.stderr)
    if failed:
        print(f"! {len(failed)} 个分片读失败，清单不完整，参数量会少算。重跑以补全。",
              file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
