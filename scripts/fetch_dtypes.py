#!/usr/bin/env python3
"""按子系统测权重精度，不读整份权重清单。

要回答「这个模型的专家是什么精度、注意力是什么精度」，不需要全部张量的形状 ——
每个子系统拿一个代表张量就够。所以这里：

  1. 读 model.safetensors.index.json 的 weight_map（张量名 → 分片），一次请求
  2. 按 lib/subsystems.py 归类，为每个子系统挑代表张量
  3. 求覆盖这些代表张量的**最小分片集**，只读这几个分片的头

万亿模型全量读头是几百个分片；这样通常 3–6 个就够。

两条必须守住的线：

  * 结果写 subsystem_dtypes.json，**绝不写进 shapes.json**。它是抽样，不是清单，
    拿去算参数量会静默少算 —— derive.py 的完整性检查靠层索引连续性兜底，
    但抽样数据根本不该进到那一步。
  * 亚字节权重按容器类型存放（FP4 两个一字节塞进 INT8）。光看 safetensors 的 dtype
    只能看到容器 INT8，报出来就错了。所以容器类型要结合 config 的 expert_dtype
    判定，并把「容器 / 逻辑」两个值都记下来。

用法:
  python3 scripts/fetch_dtypes.py --id ds-v3 --repo deepseek-ai/DeepSeek-V3
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict
from math import prod

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import cfgio  # noqa: E402
from lib import dtypes  # noqa: E402
from lib.subsystems import ORDER, classify  # noqa: E402
from fetch_hf import API, RESOLVE, _get, _get_json, _token_header, safetensors_header  # noqa: E402

CONTAINERS = {"I8", "U8", "INT8", "UINT8"}

# 占比低于此值就不认定为代表精度，记 MIXED
_DOMINANT = 0.6


# 只匹配量化 scale。**不要把 bias 放进来**：bias 是参数不是 scale，
# 混进去会让没量化的子系统也报出一个 scale 精度 —— router 的权重是 BF16、
# 根本没有 scale 张量，却因为 ffn.gate.bias 是 F32 而被报成 scale=FP32，
# 下游照这个去给 router 分配一块 FP32 scale 就错了。
# （`_amax` 这类 TensorRT 风格的名字是 `*.weight_amax`，真要支持得另写，
#  写成 `._amax$` 匹配不上，所以这里不留假防线。）
_SCALE = re.compile(r"\.(scale|weight_scale_inv|scale_inv)$", re.I)


def blk_dim(a: int, s: int) -> str:
    """由「这一维长度 a」和「scale 在这一维的条数 s」反推 block 跨度。

    整除时就是 a/s。不整除说明末块不满（scale 条数是 ceil(a/b)），
    此时 b 不唯一，只知道落在 (a/s, a/(s-1)] 区间内 —— 取区间里最小的 2 的幂作为
    代表值并标 ceil。DeepSeek V3 的 kv_a_proj_with_mqa 输出维 576 就是这种情况：
    576 = 4×128 + 64，scale 有 5 条。按「block 一律整除」写的反量化实现会在这里错位。
    """
    if a % s == 0:
        return str(a // s)
    lo, hi = a / s, a / (s - 1) if s > 1 else a
    b = 1
    while b * 2 <= hi:
        b *= 2
    return f"{b}ceil" if lo < b <= hi else f"~{a / s:.1f}"


def representatives(weight_map: dict, per_sub: int = 3, mtp_from: int = None) -> dict:
    """每个子系统挑代表张量，权重与 scale 分开挑。

    两类必须分开，否则结果会错得很安静：同一个专家既有 w1.weight（FP4 容器）
    又有 w1.scale（FP8_E8M0）。早先的版本只按「名字短」挑，于是 `.scale` 赢了
    `.weight`，整条 V4 线的专家精度被报成 FP8_E8M0 —— 报的是 scale 的精度。

    权重里再偏好名字短的（层号小、不带额外后缀），通常就是主权重。
    """
    w, s = defaultdict(list), defaultdict(list)
    for name in weight_map:
        (s if _SCALE.search(name) else w)[classify(name, mtp_from)].append(name)
    out = {}
    for sub in set(w) | set(s):
        for bucket, kind in ((w, "weight"), (s, "scale")):
            names = sorted(bucket.get(sub, []), key=lambda n: (len(n), n))
            if names:
                out.setdefault(sub, {})[kind] = names[:per_sub]
    return out


def minimal_shards(reps: dict, weight_map: dict) -> list:
    return sorted({weight_map[n] for kinds in reps.values()
                   for names in kinds.values() for n in names})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--facts-dir", default="facts")
    ap.add_argument("--rev", default=None)
    ap.add_argument("--per-subsystem", type=int, default=6,
                    help="每个子系统挑几个代表张量用于选分片。统计用读到的全部张量")
    args = ap.parse_args()

    out_dir = os.path.join(args.facts_dir, args.id)
    os.makedirs(out_dir, exist_ok=True)

    info = _get_json(API + args.repo)
    rev = args.rev or info.get("sha") or "main"
    files = [s["rfilename"] for s in info.get("siblings", [])]
    idx = next((f for f in files if f.endswith(".safetensors.index.json")), None)
    if not idx:
        print(f"  ! {args.id}: 没有 safetensors index，跳过", file=sys.stderr)
        return 1
    wm = json.loads(_get(RESOLVE.format(repo=args.repo, rev=rev, path=idx),
                         _token_header())).get("weight_map") or {}

    # 先读 config，拿到 num_hidden_layers 用于把 MTP 层从主干里分出来
    cfg_early = os.path.join(out_dir, "cfg.json")
    _cfg = json.load(open(cfg_early)) if os.path.exists(cfg_early) else {}
    _t = _cfg.get("text_config") or _cfg
    mtp_from = _t.get("num_hidden_layers")

    reps = representatives(wm, args.per_subsystem, mtp_from)
    shards = minimal_shards(reps, wm)
    print(f"  {args.id}: {len(wm)} 个张量 / {len(reps)} 个子系统，"
          f"读 {len(shards)} 个分片（全量是 {len({*wm.values()})} 个）", file=sys.stderr)

    seen = {}
    for p in shards:
        for name, meta in safetensors_header(args.repo, rev, p).items():
            if name != "__metadata__":
                seen[name] = meta

    # config 声明的专家精度，用来判定容器里装的是什么
    cfg, t = _cfg, _t
    # 顶层与 text_config 下都找 —— Kimi 的四份 VLM 把整块嵌在 text_config 下，
    # 只读顶层会让逐子系统精度表那几格渲染成「未录」，而 config 明写着 group_size 32
    q = cfgio.quant_cfg(cfg)
    declared_expert = (t.get("expert_dtype") or q.get("expert_dtype") or "").upper() or None

    # 子系统的代表精度 = **按元素数加权的多数**，在读完分片之后统计。
    #
    # 换过两版都不对：按名字长度挑会选中 `.scale`（比 `.weight` 短）；改成挑最大的单个
    # 张量，又会被同一子系统里的异类带偏 —— V3.2 的 indexer 有 FP32 的 k_norm，
    # 单看最大的那个会把整个「归一化」报成 FP32，而那一代绝大多数 norm 是 BF16。
    #
    # 按元素数占比统计既能让带宽主体说话（专家权重压倒性多数），又不会被少数异类翻盘。
    # 占比不足 _DOMINANT 时记 MIXED，细节留在 samples 里。
    def profile(sub, aux):
        cands = [(n, m) for n, m in seen.items()
                 if classify(n, mtp_from) == sub and bool(_SCALE.search(n)) == aux]
        by_dtype = defaultdict(float)
        for n, m in cands:
            by_dtype[m.get("dtype")] += prod(m.get("shape") or [0])
        total = sum(by_dtype.values())
        if not total:
            return None, [], {}
        ranked = sorted(by_dtype.items(), key=lambda kv: -kv[1])
        share = {d: v / total for d, v in ranked}
        top = ranked[0][0] if share[ranked[0][0]] >= _DOMINANT else "/".join(d for d, _ in ranked)
        cands.sort(key=lambda kv: -prod(kv[1].get("shape") or [0]))
        obs = [{"tensor": n, "dtype": m.get("dtype"), "shape": m.get("shape")}
               for n, m in cands[:args.per_subsystem]]
        return top, obs, {d: round(s, 3) for d, s in share.items()}

    # 实测 block = 权重形状 ÷ 同名 scale 形状，FP4 先还原打包。
    #
    # 必须实测而不是抄 config 的 weight_block_size：V4 一代声明 [128,128]，
    # 但 FP4 路由专家实际是 [1,32]（逐输出行、沿 K 每 32 个一个 scale）。
    # 按声明值去反量化专家，会期望 scale 行数是权重行数的 1/128，差 128 倍，
    # 权重被静默解错 —— 这类错误 config 自己看不出来，只有除一下才知道。
    def blocks(sub):
        out = {}
        for n, m in seen.items():
            if classify(n, mtp_from) != sub or not _SCALE.search(n):
                continue
            base = _SCALE.sub("", n)
            w = seen.get(base + ".weight") or seen.get(base)
            if not w or not w.get("shape") or not m.get("shape"):
                continue
            ws, ss = list(w["shape"]), m["shape"]
            if len(ws) != len(ss):
                continue
            if w.get("dtype") in CONTAINERS and declared_expert:
                ws[-1] *= 8 // dtypes.BITS.get(declared_expert, 8)
            if any(b == 0 for b in ss):
                continue
            shape = "[" + ",".join(blk_dim(a, b) for a, b in zip(ws, ss)) + "]"
            out[shape] = out.get(shape, 0) + prod(ws)
        if not out:
            return None, {}
        total = sum(out.values())
        ranked = sorted(out.items(), key=lambda kv: -kv[1])
        return "+".join(s for s, _ in ranked), {s: round(v / total, 3) for s, v in ranked}

    result = {}
    for sub in ORDER:
        if sub not in reps:
            continue
        w_top, w_obs, w_share = profile(sub, aux=False)
        s_top, s_obs, s_share = profile(sub, aux=True)
        blk, blk_share = blocks(sub)
        if not w_top:
            continue
        logical, note = w_top, None
        if any(d in CONTAINERS for d in w_share) and declared_expert and sub in ("expert", "draft"):
            logical = declared_expert
            per_byte = 8 // dtypes.BITS.get(declared_expert, 8)
            note = (f"safetensors 的 dtype 是容器 "
                    f"{'/'.join(d for d in w_share if d in CONTAINERS)}，"
                    f"按 {per_byte} 个一字节打包存放，numel 是容器数不是参数数。"
                    f"**位宽是实测的**（容器末维正好是逻辑维的 1/{per_byte}，"
                    f"同层未打包的共享专家给出逻辑维作对照）；"
                    f"**具体是 {declared_expert} 而不是同位宽的整型，只有 config 的 "
                    f"expert_dtype={declared_expert.lower()} 一个来源** —— "
                    f"形状和 dtype 都区分不了 4 bit 浮点与 4 bit 整型")
        result[sub] = {
            "dtype": dtypes.norm(logical) if "/" not in logical else logical,
            "dtype_share": w_share,
            "scale_dtype": (dtypes.norm(s_top) if s_top and "/" not in s_top else s_top),
            "scale_dtype_share": s_share,
            "block": blk,
            "block_share": blk_share,
            "samples": w_obs + s_obs,
        }
        if note:
            result[sub]["packing_note"] = note

    doc = {"repo": args.repo, "rev": rev, "method": "抽样：每个子系统 "
           f"{args.per_subsystem} 个代表张量，共读 {len(shards)} 个分片头",
           "caveat": "抽样数据，只能回答精度，不能用于参数量统计",
           "subsystems": result}
    with open(os.path.join(out_dir, "subsystem_dtypes.json"), "w") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
    for sub, v in result.items():
        sc = f" scale={v['scale_dtype']}" if v.get("scale_dtype") else ""
        bk = f" block={v['block']}" if v.get("block") else ""
        print(f"    {sub:16s} {v['dtype']:10s}{sc:20s}{bk:24s} {v['samples'][0]['tensor']}",
              file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
