#!/usr/bin/env python3
"""从完整权重清单提取逐层实现：每层有哪些算子组、每个张量的形状与精度、实测量化块、
每 token 缓存贡献。产出 data/derived/<id>.layers.json，站点模型页据此画逐层图。

为什么要逐层而不是一份「代表层」：DeepSeek V4 一代起主干层不同构。产出全局 KV 的
源层、只重打分的层、带 Engram 的层、压缩器带不带门控 —— 这些都只在权重清单里看得出，
拿任意一层当代表去排算子表，其余几类层的算子就漏了。

三条线：

  * 层的角色一律从权重判定（有没有 compressor.wkv / indexer.wk / engram.*），
    config 里的同名列表只拿来比对。config 和权重不符时以权重为准，不符处记进 checks。
  * 实测量化块 = 权重形状 ÷ scale 形状，INT8 容器先把末维还原成 FP4 的逻辑宽度。
    不抄 config 的 weight_block_size。
  * 缓存贡献只把权重能证明的部分记成实测：条目宽度来自产出张量的输出维。
    一个条目覆盖几个 token 取决于 compress_ratios，那是 config 值，
    按它折算出的每 token 值单独标 basis，不冒充实测。

只吃完整清单。零差异孪生版的 shapes.json 是空的 —— 空文件和「真的没有」长得一样，
所以这里遇到空清单直接退出，不产出一份全是空层的文件。

用法:
  python3 scripts/extract_layers.py --id deepseek-v41-flash
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from math import prod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import dtypes  # noqa: E402
from lib.subsystems import classify  # noqa: E402
from fetch_dtypes import CONTAINERS, blk_dim  # noqa: E402

# 层内算子组，顺序即一层里的执行顺序，也是图上的行序。
# 正则作用在去掉 layers.N. / mtp.N. 前缀之后的名字上，先匹配先得。
# 和 subsystems.py 的分工：子系统回答「这块权重算谁的账、什么精度」，
# 算子组回答「这一层要跑哪些算子」—— 同属归一化的 kv_norm 和 compressor.norm
# 在算子表里是不同位置的两个算子，所以这里按位置分，不按子系统分。
OP_GROUPS = [
    ("hc_attn", "mHC 混合 · 注意力前", r"^hc_attn_"),
    ("attn_norm", "attn_norm", r"^attn_norm\."),
    ("engram", "Engram 查表与投影", r"^engram\."),
    ("q", "Q 路径", r"^attn\.(wq_a|q_norm|wq_b)\."),
    ("kv_local", "滑窗 KV", r"^attn\.(wkv|kv_norm)\."),
    ("indexer", "稀疏索引器", r"^attn\.indexer\."),
    ("compressor", "KV 压缩器", r"^attn\.compressor\."),
    ("sink", "attention sink", r"^attn\.attn_sink$"),
    ("o", "输出投影", r"^attn\.(wo_a|wo_b)\."),
    ("hc_ffn", "mHC 混合 · FFN 前", r"^hc_ffn_"),
    ("ffn_norm", "ffn_norm", r"^ffn_norm\."),
    ("router", "路由器", r"^ffn\.gate\."),
    ("shared_expert", "共享专家", r"^ffn\.shared_experts\."),
    ("expert", "路由专家", r"^ffn\.experts\."),
    ("mtp_glue", "MTP 拼接", r"^(main_proj|main_norm|enorm|hnorm|eh_proj)\."),
    # 只在最后一个 MTP 块上：DSpark 的 markov / confidence 头与草稿侧末端 norm
    ("draft_head", "DSpark 头", r"^(markov_head|confidence_head|norm)\."),
]
_OP_RX = [(k, re.compile(rx)) for k, _, rx in OP_GROUPS]

_SCALE = re.compile(r"\.(scale|weight_scale_inv|scale_inv)$")
_BLOCK = re.compile(r"^(layers|mtp)\.(\d+)\.(.*)$")
_EXPERT = re.compile(r"\.experts\.(\d+)\.")


def op_group(local: str) -> str:
    for k, rx in _OP_RX:
        if rx.search(local):
            return k
    return "other"


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def collect(shapes: dict):
    """按 (主干/MTP, 块号) 归拢张量；专家折叠成 experts.E.*，并核对 384 份是否同形同精度。"""
    blocks = defaultdict(lambda: defaultdict(list))
    for name, meta in shapes.items():
        m = _BLOCK.match(name)
        if not m:
            continue
        kind = "main" if m.group(1) == "layers" else "mtp"
        local = _EXPERT.sub(".experts.E.", "." + m.group(3))[1:]
        blocks[(kind, int(m.group(2)))][local].append((name, meta))
    return blocks


def tensor_rows(block: dict, fp4: bool, mtp_from: int):
    """一层的张量清单：权重行带上同名 scale，算出逻辑形状与实测块。"""
    rows, issues = [], []
    for local in sorted(block):
        if _SCALE.search(local):
            continue
        items = block[local]
        name, meta = items[0]
        variants = {(tuple(m["shape"]), m["dtype"]) for _, m in items}
        if len(variants) > 1:
            issues.append(f"{local}: {len(items)} 份副本形状或精度不一致 {sorted(variants)}")
        shape = list(meta["shape"])
        dt = dtypes.norm(meta["dtype"])
        logical = list(shape)
        logical_dt = dt
        if meta["dtype"] in CONTAINERS and fp4:
            logical[-1] *= 2
            logical_dt = "FP4"
        base = re.sub(r"\.weight$", "", local)
        sc = None
        for suffix in (".scale", ".weight_scale_inv", ".scale_inv"):
            if base + suffix in block:
                sc = block[base + suffix][0][1]
                break
        row = {
            "tensor": local,
            "group": op_group(local),
            "subsystem": classify(name, mtp_from),
            "count": len(items),
            "shape": shape,
            "dtype": dt,
        }
        if logical != shape or logical_dt != dt:
            row["logical_shape"] = logical
            row["logical_dtype"] = logical_dt
        nbytes = prod(shape) * dtypes.nbytes(dt)
        if sc:
            row["scale_shape"] = sc["shape"]
            row["scale_dtype"] = dtypes.norm(sc["dtype"])
            if len(sc["shape"]) == len(logical) and all(sc["shape"]):
                row["block"] = "[" + ",".join(blk_dim(a, s) for a, s in zip(logical, sc["shape"])) + "]"
            nbytes += prod(sc["shape"]) * dtypes.nbytes(sc["dtype"])
        row["bytes_each"] = int(nbytes)
        rows.append(row)
    # 没有对应权重的 scale 不该存在，出现了就是命名约定变了
    for local in block:
        if _SCALE.search(local):
            base = _SCALE.sub("", local)
            if base + ".weight" not in block and base not in block:
                issues.append(f"{local}: 找不到对应的权重张量")
    return rows, issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--facts-dir", default="facts")
    ap.add_argument("--out-dir", default="data/derived")
    ap.add_argument("--kv-bytes", type=int, default=2,
                    help="缓存元素字节数，默认 BF16。只影响字节换算，不影响元素数")
    args = ap.parse_args()

    fdir = os.path.join(args.facts_dir, args.id)
    sp, cp = os.path.join(fdir, "shapes.json"), os.path.join(fdir, "cfg.json")
    shapes = json.load(open(sp))
    if not shapes:
        sys.exit(f"{sp} 是空的。零差异孪生版故意不拉全量清单，逐层图要对它的孪生版生成")
    cfg = json.load(open(cp))
    t = cfg.get("text_config") or cfg
    n_main = t["num_hidden_layers"]
    fp4 = str((cfg.get("quantization_config") or {}).get("expert_dtype", "")).lower() == "fp4"
    ratios = t.get("compress_ratios") or []
    window = t.get("sliding_window")

    blocks = collect(shapes)
    layers, classes, sig_to_class, issues = [], [], {}, []

    for (kind, idx) in sorted(blocks, key=lambda k: (k[0] != "main", k[1])):
        block = blocks[(kind, idx)]
        rows, iss = tensor_rows(block, fp4, n_main)
        issues += [f"{kind}.{idx} {i}" for i in iss]
        has = set(block)
        flat_idx = idx if kind == "main" else n_main + idx

        roles = []
        if "attn.compressor.wkv.weight" in has:
            roles.append("kv_source")
            if "attn.compressor.wgate.weight" not in has:
                roles.append("compressor_ungated")
        if "attn.indexer.wk.weight" in has or "attn.indexer.compressor.wkv.weight" in has:
            roles.append("index_source")
        elif "attn.indexer.wq_b.weight" in has:
            roles.append("index_rescore")
        if any(k.startswith("engram.") for k in has):
            roles.append("engram")

        def out_dim(local):
            return block[local][0][1]["shape"][0] if local in block else None

        ratio = ratios[flat_idx] if flat_idx < len(ratios) else None
        cache = {}
        lw = out_dim("attn.wkv.weight")
        if lw and window:
            cache["local_window"] = {
                "entry_elems": lw, "window": window,
                "bytes_per_sequence": lw * window * args.kv_bytes,
            }
        for key, src in (("global_kv", "attn.compressor.wkv.weight"),
                         ("index_k", "attn.indexer.wk.weight")):
            w = out_dim(src)
            if not w:
                continue
            c = {"entry_elems": w, "source": src, "compress_ratio": ratio}
            if ratio:
                c["per_token_elems"] = w / ratio
                c["per_token_bytes"] = w / ratio * args.kv_bytes
            cache[key] = c

        groups = sorted({r["group"] for r in rows}, key=lambda g: [k for k, *_ in OP_GROUPS + [("other",)]].index(g))
        bytes_by_group = defaultdict(int)
        for r in rows:
            bytes_by_group[r["group"]] += r["bytes_each"] * r["count"]
        # decode 每 token 实际搬运：路由专家只读 top_k 份，其余组全读。
        # MTP 块里的专家池与 top_k 是 DSpark 自己的（128 选 3），不是主干的 384 选 6
        if kind == "mtp" and t.get("dspark_n_routed_experts"):
            pool, k = t["dspark_n_routed_experts"], t.get("dspark_num_experts_per_tok")
        else:
            pool, k = t.get("n_routed_experts"), t.get("num_experts_per_tok")
        decode_by_group = dict(bytes_by_group)
        n_seen = max((r["count"] for r in rows if r["group"] == "expert"), default=0)
        if n_seen and k:
            if pool and pool != n_seen:
                issues.append(f"{kind}.{idx} 专家数：config {pool}，权重 {n_seen}")
            decode_by_group["expert"] = bytes_by_group["expert"] * k // n_seen
        # Engram 嵌入表每 token 只 gather 几行：每个 n-gram 阶（2..max）× 每个哈希头一行。
        # 按存储算会把一张 90 GiB 的表记成每 token 全读；engram.wkv 投影则是稠密全读，不折算
        emb = next((r for r in rows if r["tensor"] == "engram.embed.weight"), None)
        if emb and t.get("engram_max_ngram_size") and t.get("engram_n_heads"):
            rows_read = (t["engram_max_ngram_size"] - 1) * t["engram_n_heads"]
            decode_by_group["engram"] += rows_read * emb["bytes_each"] // emb["shape"][0] - emb["bytes_each"]

        sig = json.dumps([(r["tensor"], r["shape"], r["dtype"], r.get("scale_shape")) for r in rows])
        if sig not in sig_to_class:
            sig_to_class[sig] = len(classes)
            classes.append({"key": f"C{len(classes)}", "layers": [], "roles": roles, "tensors": rows})
        cls = classes[sig_to_class[sig]]
        cls["layers"].append(f"{'L' if kind == 'main' else 'MTP'}{idx}")

        layers.append({
            "kind": kind, "index": idx, "label": f"{'L' if kind == 'main' else 'MTP'}{idx}",
            "class": cls["key"], "roles": roles, "compress_ratio": ratio,
            "groups": groups, "bytes_by_group": dict(bytes_by_group),
            "bytes_total": sum(bytes_by_group.values()),
            "decode_bytes_by_group": decode_by_group,
            "decode_bytes_total": sum(decode_by_group.values()),
            "experts_active": f"{k}/{n_seen}" if n_seen and k else None,
            "cache": cache,
        })

    main_layers = [l for l in layers if l["kind"] == "main"]
    if len(main_layers) != n_main:
        issues.append(f"主干层数：config {n_main}，权重 {len(main_layers)}")

    # config 只做比对。角色名对应的 config 字段
    def by_role(role):
        return [l["index"] for l in main_layers if role in l["roles"]]
    checks = []
    for role, key in (("kv_source", "kv_source_layer_ids"),
                      ("index_source", "index_source_layer_ids"),
                      ("engram", "engram_layer_ids")):
        if key not in t:
            continue
        w, c = by_role(role), list(t[key])
        if role == "index_source":
            # config 把产出 index K 的层和只重打分的层并成一份列表，比对时并起来看
            w_all = sorted(by_role("index_source") + by_role("index_rescore"))
            checks.append({"item": key, "config": c, "weights": w_all,
                           "status": "ok" if w_all == c else "MISMATCH",
                           "note": f"config 一份列表；权重分两类：产出 index K {w}，"
                                   f"只重打分 {by_role('index_rescore')}。按 config 当 8 层都产出会把 indexer 缓存算成两倍"})
        else:
            checks.append({"item": key, "config": c, "weights": w,
                           "status": "ok" if w == c else "MISMATCH"})
    if ratios:
        checks.append({"item": "compress_ratios 长度", "config": len(ratios),
                       "weights": len(layers),
                       "status": "ok" if len(ratios) == len(layers) else "MISMATCH",
                       "note": "末尾几项对应 MTP 块"})
    ungated = by_role("compressor_ungated")
    if ungated:
        checks.append({"item": "压缩器无门控的源层", "config": [f"L{i}:{ratios[i]}" for i in ungated],
                       "weights": [f"L{i}" for i in ungated],
                       "status": "ok" if all(ratios[i] == 1 for i in ungated) else "MISMATCH",
                       "note": "权重列是缺 compressor.wgate 的源层，config 列是这些层的 compress_ratio；应恰好都是 1"})

    totals = {}
    for key in ("global_kv", "index_k"):
        src = [l for l in main_layers if key in l["cache"]]
        if not src:
            continue
        totals[key] = {
            "layers": [l["index"] for l in src],
            "entry_elems_sum": sum(l["cache"][key]["entry_elems"] for l in src),
            "per_token_elems_sum": sum(l["cache"][key].get("per_token_elems", 0) for l in src),
            "note": "entry_elems_sum 把每个条目当成一个 token；per_token_elems_sum 按 compress_ratio 折算",
        }
    lw = [l["cache"]["local_window"]["bytes_per_sequence"] for l in main_layers if "local_window" in l["cache"]]
    if lw:
        totals["local_window_bytes_per_sequence"] = sum(lw)

    extra = {}
    if t.get("dspark_target_layer_ids"):
        extra["dspark_target_layers"] = t["dspark_target_layer_ids"]
    if t.get("candidate_source_layer_id") is not None:
        extra["candidate_source_layer"] = t["candidate_source_layer_id"]

    out = {
        "id": args.id,
        "generated_by": "scripts/extract_layers.py",
        "generated_at": datetime.date.today().isoformat(),
        "inputs": {"shapes": sp, "shapes_sha256_16": sha(sp), "cfg": cp, "cfg_sha256_16": sha(cp),
                   "tensors": len(shapes)},
        "op_groups": [{"key": k, "label": lb} for k, lb, _ in OP_GROUPS],
        "kv_elem_bytes": args.kv_bytes,
        "bytes_semantics": "bytes_by_group 是存储（含 scale，全部专家）；decode_bytes_by_group 是每 token "
                           "搬运，路由专家按 top_k / 专家数折算，其余组全读",
        "cache_basis": {
            "entry_elems": "产出张量的输出维，权重实测",
            "per_token_elems": "entry_elems ÷ compress_ratio。每条目覆盖 compress_ratio 个 token 是推断，"
                               "不是实测 —— 依据是比值为 1 的源层恰好没有 compressor.wgate（只有 1 个 token "
                               "时无须加权汇聚），以及 indexer.wk 的输入维等于压缩器输出维，即 index K "
                               "也按压缩条目算",
            "local_window": "attn.wkv 输出维（实测）× sliding_window（config），每序列固定，不随上下文增长",
        },
        "config_only_roles": extra,
        "layers": layers,
        "classes": classes,
        "cache_totals": totals,
        "checks": checks,
        "issues": issues,
    }
    os.makedirs(args.out_dir, exist_ok=True)
    op = os.path.join(args.out_dir, f"{args.id}.layers.json")
    json.dump(out, open(op, "w"), ensure_ascii=False, indent=1)
    print(f"写入 {op}：{len(main_layers)} 主干层 + {len(layers) - len(main_layers)} MTP 块，"
          f"{len(classes)} 类结构")
    for c in classes:
        print(f"  {c['key']} {','.join(c['roles']) or '—':32s} {' '.join(c['layers'])}")
    for c in checks:
        print(f"  [{c['status']}] {c['item']}: config={c['config']} weights={c['weights']}")
    for k, v in totals.items():
        print(f"  {k}: {v}")
    for i in issues:
        print(f"  !! {i}")


if __name__ == "__main__":
    main()
