#!/usr/bin/env python3
"""从完整权重清单提取逐层实现：每层有哪些算子组、每个张量的形状与精度、实测量化块、
每 token 缓存贡献。产出 data/derived/<id>.layers.json，站点模型页据此画逐层图。

为什么要逐层而不是一份「代表层」：DeepSeek V4 一代起主干层不同构。产出全局 KV 的
源层、只重打分的层、带 Engram 的层、压缩器带不带门控 —— 这些都只在权重清单里看得出，
拿任意一层当代表去排算子表，其余几类层的算子就漏了。V4.1 之外的五个旗舰更极端：
Kimi-K3 的 93 层里 69 层是 KDA、24 层是 MLA，Qwen3.8 的 92 层里 69 层是线性注意力，
StepFun 的 92 层按 3:1 分滑窗与全局 —— 「代表层」在这几份里根本不存在。

三条线：

  * 层的角色一律从权重判定（有没有产出全局 KV / index K 的投影、有没有 engram.*），
    config 里的同名列表只拿来比对。config 和权重不符时以权重为准，不符处记进 checks。
  * 实测量化块 = 权重形状 ÷ scale 形状，INT8 / UINT8 容器先把末维还原成低位格式的
    逻辑宽度。不抄 config 的 weight_block_size。
  * 缓存贡献只把权重能证明的部分记成实测：条目宽度来自产出张量的输出维。
    一个条目覆盖 compress_ratios 个 token（时序池化），比值本身是 config 值，
    折算出的每 token 值单独标 basis，不冒充实测。
  * 字节 = 元素数 × 落盘精度。精度读记录里的 precision.kv_cache_components，
    来源和元素数不同，按分量各自记格式与口径；读不到才按 --kv-bytes 兜底。
    按落盘精度算出的是理论下限，不含分页与环页粒度。
  * 递归式注意力层（线性注意力 / KDA / GDN）**不产生每 token KV**，状态每序列固定。
    它们单独记一个 recurrent_state 分量，不折成每 token 值 —— 那两个量不是一回事。

六个族系的张量词汇差异很大（见 OP_GROUPS 与 KV_SOURCES 的注释）。词汇表留在代码里、
不搬进 data/schema：算子组是**代码的分类规则**，每一条的价值一半在正则、一半在它旁边
那段「为什么锚在这里」的案子，JSON 装不下注释。data/schema/subsystems.json 只装展示
顺序与中文名，正则连同案子留在 subsystems.py —— 这里沿用同一个分工。

只吃完整清单。零差异孪生版的 shapes.json 是空的 —— 空文件和「真的没有」长得一样，
所以这里遇到空清单直接退出，不产出一份全是空层的文件。**提取不到层、或层数与 config
不符，同样是退出码非 0 且不写文件**：本仓栽过一次「闸口全绿而产物三千多处出错」
（STATUS.md 坑 #4），空产出走成功路径是同一个毛病的小号版本。

用法:
  python3 scripts/extract_layers.py --id deepseek-v41-flash
  python3 scripts/extract_layers.py --id glm-5.3 --facts-dir /abs/path/to/facts
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import sys
from collections import Counter, defaultdict
from math import prod

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib import cfgio  # noqa: E402
from lib import dtypes  # noqa: E402
from lib.subsystems import classify, is_tower  # noqa: E402
from fetch_dtypes import CONTAINERS, blk_dim  # noqa: E402

# 层内算子组，顺序即一层里的执行顺序，也是图上的行序。
# 正则作用在去掉栈前缀与 layers.N. / mtp.N. 之后的局部名上，先匹配先得。
# 和 subsystems.py 的分工：子系统回答「这块权重算谁的账、什么精度」，
# 算子组回答「这一层要跑哪些算子」—— 同属归一化的 kv_norm 和 compressor.norm
# 在算子表里是不同位置的两个算子，所以这里按位置分，不按子系统分。
#
# 一张表装六族。每一条都钉在具体拼写上，不放成模糊匹配：放松的代价是把语义不同的
# 东西并进一格（GLM 的 kv_a_proj_with_mqa 是 MLA 潜向量、DeepSeek 的 attn.wkv 是滑窗
# KV，两者都能被 `kv` 咬到，并进一格之后图上就分不出哪层有随上下文增长的缓存了）。
OP_GROUPS = [
    ("hc_attn", "mHC 混合 · 注意力前", r"^hc_attn_"),
    # attn_norm：DeepSeek 叫 attn_norm，其余五族沿用 HF 的 input_layernorm
    ("attn_norm", "attn_norm", r"^(attn_norm|input_layernorm)\."),
    # Kimi-K3 的 Attention Residuals：每层一组 [1, hidden] 系数加一个 norm
    ("attn_res", "注意力残差系数", r"^self_attention_res_(norm|proj)\."),
    ("engram", "Engram 查表与投影", r"^engram\."),
    # Q 路径三种写法：DeepSeek 的 wq_a/wq_b、GLM 与 Kimi MLA 的 q_a_proj/q_b_proj、
    # MiniMax / StepFun / Qwen 全注意力层的 q_proj。Qwen 的 q_proj 输出维是 2×头×维
    # （一半是 query、一半是输出门，config 的 attn_output_gate），同在这一组
    ("q", "Q 路径", r"^attn\.(wq_a|q_norm|wq_b)\."
                    r"|^self_attn\.(q_a_proj|q_a_layernorm|q_b_proj|q_proj|q_norm)\."),
    # MLA 潜向量：一个 kv_a_proj_with_mqa 输出维里同时装压缩 KV 与 rope 尾段，
    # 它才是随上下文增长的那份缓存；kv_b_proj 是解压回头维度的权重，不进缓存
    ("kv_latent", "MLA 潜向量 KV", r"^self_attn\.(kv_a_proj_with_mqa|kv_a_layernorm|kv_b_proj)\."),
    ("kv_local", "滑窗 KV", r"^attn\.(wkv|kv_norm)\."),
    # K / V 分开落盘的族系（MiniMax MSA、StepFun GQA、Qwen 全注意力层）。
    # 同一个 self_attn.k_proj 在 Kimi 的 KDA 层里是递归输入，不是缓存 ——
    # 那边由 RECURRENT_GROUPS 先截走，判据是结构不是拼写，见 is_recurrent()
    ("kv", "K / V 投影", r"^self_attn\.(k_proj|k_norm|v_proj|v_norm)\."),
    # 线性注意力 / KDA / GDN 的递归状态与输入投影。Qwen 的整支挂在 linear_attn. 下，
    # Kimi 的挂在 self_attn. 下（靠 RECURRENT_GROUPS 分流）。
    # out_proj 与 in_proj_z 两个否定前瞻是有意的：它们分别排到下面的 o 与 attn_gate，
    # 和 Kimi 的 o_proj / g_proj、StepFun 的 g_proj 同在一行 —— 跨族系看同一行才有意义，
    # 而递归层的输出投影在算子表里就是一个普通 GEMM，放进「状态」那行会读错位置
    ("linear_attn", "线性注意力状态", r"^linear_attn\.(?!(out_proj|in_proj_z)\.)"),
    # 稀疏索引器四种写法：DeepSeek attn.indexer.*、GLM self_attn.indexer.*、
    # MiniMax index_[qk]_proj / index_[qk]_norm、StepFun sparse_indexer_* 与 ssmax_s
    # （config 的 sparse_indexer_ssmax_s_granularity=q_head，它是索引器的打分缩放）
    ("indexer", "稀疏索引器", r"^attn\.indexer\.|^self_attn\.indexer\."
                              r"|^self_attn\.index_[qk]_(proj|norm)\."
                              r"|^self_attn\.sparse_indexer_|^self_attn\.ssmax_s$"),
    ("compressor", "KV 压缩器", r"^attn\.compressor\."),
    ("sink", "attention sink", r"^attn\.attn_sink$"),
    # 注意力输出门：Kimi 两类层都有（config mla_use_output_gate），
    # StepFun 是逐头一维（use_head_wise_attn_gate，g_proj [64, hidden]）
    ("attn_gate", "注意力输出门", r"^self_attn\.g_proj\.|^linear_attn\.in_proj_z\."),
    ("o", "输出投影", r"^attn\.(wo_a|wo_b)\.|^self_attn\.o_proj\.|^linear_attn\.out_proj\."),
    ("hc_ffn", "mHC 混合 · FFN 前", r"^hc_ffn_"),
    ("ffn_norm", "ffn_norm", r"^(ffn_norm|post_attention_layernorm)\."),
    ("ffn_res", "FFN 残差系数", r"^mlp_res_(norm|proj)\."),
    # 路由器：gate 后面必须跟点号，否则会把 StepFun 的 moe.gate_proj（专家权重，
    # 堆叠存储 [352, …]）当成路由器 —— 两者只差 _proj 三个字符
    ("router", "路由器", r"^(ffn|mlp|moe|block_sparse_moe)\.gate\."
                         r"|^moe\.router_bias$|^block_sparse_moe\.e_score_correction_bias$"),
    # 共享专家四种写法：DeepSeek / GLM / MiniMax / Kimi 的 shared_experts、
    # Qwen 的 shared_expert 与 shared_expert_gate、StepFun 的 share_expert（单数无 d）
    ("shared_expert", "共享专家", r"^(ffn|mlp|block_sparse_moe)\.shared_experts\."
                                  r"|^mlp\.shared_expert(_gate)?[._]|^share_expert\."),
    # Kimi-K3 的 latent MoE：路由专家共用的升降维投影与 norm，每层一份、不随 top_k 折算。
    # 并进 expert 会被 top_k/专家数 除掉（896 选 16，偏低 56 倍），所以单列
    ("expert_latent", "路由专家潜空间投影",
     r"^block_sparse_moe\.routed_expert_(up_proj|down_proj|norm)\."),
    # 路由专家两种存法：分片（每专家一个张量，名字里带 experts.N.）与堆叠
    # （专家维在张量内部，Qwen mlp.experts.gate_up_proj [512, …]、StepFun moe.gate_proj
    # [352, …]）。堆叠的份数看不出来，decode 折算另走一路，见 main() 里的 n_seen
    ("expert", "路由专家", r"^ffn\.experts\.|^(mlp|block_sparse_moe)\.experts[._]"
                           r"|^moe\.(gate_proj|up_proj|down_proj)\."),
    # 稠密 FFN：前 first_k_dense_replace 层与（StepFun）末几层不走 MoE。
    # 必须排在 expert / shared_expert 之后，锚死三个投影名，别用裸 mlp
    ("dense_ffn", "稠密 FFN", r"^mlp\.(gate_proj|up_proj|down_proj)\."),
    # MTP 拼接：DeepSeek main_proj/main_norm、V3 一代的 enorm/hnorm/eh_proj、
    # GLM 与 StepFun 的 shared_head。Qwen 的 fc / norm / pre_fc_norm_* 挂在 mtp. 栈顶、
    # 不属于任何一块，由 collect() 结构性地归到这一组，不靠名字 —— 靠名字的话
    # 它那个裸 norm.weight 会落进下面的 draft_head，在 Qwen 页上标成「DSpark 头」
    ("mtp_glue", "MTP 拼接", r"^(main_proj|main_norm|enorm|hnorm|eh_proj)\."
                             r"|^(transformer\.)?shared_head\."),
    # 只在最后一个 MTP 块上：DSpark 的 markov / confidence 头与草稿侧末端 norm
    ("draft_head", "DSpark 头", r"^(markov_head|confidence_head|norm)\."),
]
_OP_RX = [(k, re.compile(rx)) for k, _, rx in OP_GROUPS]

# 递归层专用，优先于 OP_GROUPS。Kimi 的 KDA 层把递归输入叫 self_attn.q_proj /
# k_proj / v_proj —— 和 MiniMax 普通 MSA 的 q_proj 一模一样的拼写，两种语义。
# 本仓三次栽在「用名字当锚」上（STATUS.md 坑 #2），这里的分流判据是结构：
# 本层有没有因果深度卷积与衰减参数，见 is_recurrent()
RECURRENT_GROUPS = [
    ("linear_attn", r"^self_attn\.(q|k|v)_(proj|conv1d)\."
                    r"|^self_attn\.(A_log|dt_bias)$"
                    r"|^self_attn\.(f_a_proj|f_b_proj|b_proj|o_norm)\."),
]
_REC_RX = [(k, re.compile(rx)) for k, rx in RECURRENT_GROUPS]

# 缓存分量的产出张量候选。宽度一律取该张量的**输出维**（shape[0]），不是 config 的
# 头数乘积 —— 后者在 MQA 共享的 index K 上高估 32 倍（model-audit 缓存账第 2 问）。
# 每个分量按组试，第一组整组齐备就用它；多张量的组合宽度相加（K 与 V 分开落盘的族系）。
# 这张表只回答「谁产出」。「这一层产的是全局 KV 还是窗口 KV」是另一个问题，
# 由层类型决定，见 main() 里的 is_global / is_sliding
KV_SOURCES = {
    "global_kv": [
        ("attn.compressor.wkv.weight",),            # DeepSeek V4：压缩后的全局 KV
        ("self_attn.kv_a_proj_with_mqa.weight",),    # GLM / Kimi MLA 层：潜向量（含 rope 尾段）
        ("self_attn.k_proj.weight", "self_attn.v_proj.weight"),  # MiniMax / StepFun / Qwen
    ],
    "index_k": [
        ("attn.indexer.wk.weight",),                 # DeepSeek
        ("attn.indexer.compressor.wkv.weight",),     # DeepSeek：indexer 自带压缩器的一代
        ("self_attn.indexer.wk.weight",),            # GLM
        ("self_attn.index_k_proj.weight",),          # MiniMax MSA
        ("self_attn.sparse_indexer_k.weight",),      # StepFun
    ],
    "local_window": [
        ("attn.wkv.weight",),                        # DeepSeek：全部主干层都带滑窗
        ("self_attn.k_proj.weight", "self_attn.v_proj.weight"),  # StepFun 的滑窗层
    ],
}

# 专家池与 top_k 的字段名，各族不同。顺序即优先级
_POOL_KEYS = ("n_routed_experts", "num_local_experts", "moe_num_experts", "num_experts")
_TOPK_KEYS = ("num_experts_per_tok", "moe_top_k", "num_experts_per_token")

_SCALE = re.compile(r"\.(scale|weight_scale|weight_scale_inv|scale_inv)$")
_SCALE_SUFFIXES = (".scale", ".weight_scale", ".weight_scale_inv", ".scale_inv")
# 权重张量的末段：compressed-tensors 把打包权重叫 weight_packed（Kimi 的 MXFP4 专家）
_WEIGHT_TAIL = re.compile(r"\.(weight|weight_packed)$")
# 块定位：前缀 + layers.N. / mtp.N.。不锚行首（六族主干前缀各不相同），
# 但前缀整条留着交给调用方判栈 —— 放成模糊匹配会把视觉塔的层号咬进来
_BLOCK = re.compile(r"^(.*?)((?:layers|mtp)\.)(\d+)\.(.*)$")
# 草稿栈前缀下、不在 layers.N. 里的张量（Qwen 的 mtp.fc / mtp.norm / mtp.pre_fc_norm_*）
_LOOSE = re.compile(r"^(mtp|draft|dspark|eagle)\.(.*)$")
_DRAFT_STACK = re.compile(r"\b(mtp|draft|dspark|eagle)\b", re.I)
_EXPERT = re.compile(r"\.experts\.(\d+)\.")
# 递归式注意力的结构标记：因果深度卷积与衰减参数
_REC_MARK = re.compile(r"conv1d|^self_attn\.(A_log|dt_bias)$|^linear_attn\.")
# 递归层的 QKV 输入投影，用来和 config 的头数乘积对账
_REC_QKV = re.compile(r"^self_attn\.(q|k|v)_proj\.weight$|^linear_attn\.in_proj_qkv\.weight$")


def op_group(local: str, recurrent: bool = False) -> str:
    if recurrent:
        for k, rx in _REC_RX:
            if rx.search(local):
                return k
    for k, rx in _OP_RX:
        if rx.search(local):
            return k
    return "other"


def is_recurrent(has) -> bool:
    """这一层的注意力是不是递归式（线性注意力 / KDA / GDN）。

    判据是权重里有没有因果深度卷积与衰减参数（conv1d / A_log / dt_bias / linear_attn.*），
    不是名字里有没有 linear —— Kimi 的 KDA 层把递归输入叫 self_attn.q_proj，
    和 MiniMax 普通 MSA 的 q_proj 一模一样。同一个拼写两种语义，只能按结构分。
    """
    return any(_REC_MARK.search(x) for x in has)


def packed_format(cfg: dict):
    """容器 dtype（U8 / I8）里装的低位格式：几位、叫什么。

    两种声明法都认：DeepSeek 在 quantization_config.expert_dtype 里写 fp4；
    Kimi 走 compressed-tensors，位宽在 config_groups.<g>.weights.num_bits、
    格式名在顶层 format（mxfp4-pack-quantized）。
    认不出就返回 None —— 那时容器形状照原样记，**不替它猜逻辑宽度**。
    声明的位置两处都找（Kimi 的 VLM 把整块嵌在 `text_config` 下），见 lib/cfgio.py。
    """
    qc = cfgio.quant_cfg(cfg)   # 顶层与 text_config 下都找，见 lib/cfgio.py
    ed = str(qc.get("expert_dtype", "")).lower()
    if ed == "fp4":
        return {"bits": 4, "dtype": "FP4"}
    for g in (qc.get("config_groups") or {}).values():
        bits = ((g or {}).get("weights") or {}).get("num_bits")
        if bits and bits < 8:
            name = str(qc.get("format") or "").split("-")[0]
            return {"bits": bits, "dtype": dtypes.norm(name) if name else f"INT{bits}"}
    return None


def linear_heads(t: dict):
    """线性注意力的头数与 head_dim。两套 config 写法：

      * Kimi-K3：linear_attn_config.{num_heads, head_dim}（K 与 V 同 head_dim）
      * Qwen3.8：linear_{num_key_heads, key_head_dim, num_value_heads, value_head_dim}

    状态元素数 = value_heads × key_head_dim × value_head_dim。这个乘积不随 K / V 头的
    分组约定变化：Qwen 的 16 个 K 头各带 8 个 V 头，128×128×128 与 16×128×(8×128)
    是同一个数，所以不必先定下它用哪一种分组记状态。
    """
    la = t.get("linear_attn_config") or {}
    if la.get("num_heads") and la.get("head_dim"):
        return {"key_heads": la["num_heads"], "key_head_dim": la["head_dim"],
                "value_heads": la["num_heads"], "value_head_dim": la["head_dim"],
                "fields": "linear_attn_config.{num_heads,head_dim}"}
    keys = ("linear_num_key_heads", "linear_key_head_dim",
            "linear_num_value_heads", "linear_value_head_dim")
    if all(t.get(k) for k in keys):
        return {"key_heads": t[keys[0]], "key_head_dim": t[keys[1]],
                "value_heads": t[keys[2]], "value_head_dim": t[keys[3]],
                "fields": "{" + ",".join(keys) + "}"}
    return None


def sha(path: str) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()[:16]


def collect(shapes: dict, n_main: int):
    """按 (主干/MTP, 块号) 归拢张量；专家折叠成 experts.E.*，并核对副本是否同形同精度。

    块的定位要同时顶住两头：

      * **前缀不能锚死行首。** 六族的主干前缀各不相同 —— DeepSeek 裸 `layers.N.`、
        GLM / Qwen / StepFun `model.layers.N.`、MiniMax / Kimi
        `language_model.model.layers.N.`。锚行首就只认得出一族。
      * **也不能放成模糊匹配。** 视觉塔自己有一套层号（MiniMax 的
        vision_tower.vision_model.encoder.layers.N. 32 层），放松之后凭空多出 32 个假层,
        正是 STATUS.md 坑 #2 的第三次。所以前缀整条留着，塔先用 is_tower() 剔掉。

    MTP 两种存法都要认：另起一个栈（DeepSeek `mtp.N.`、Qwen `mtp.layers.N.`），
    或者借用主干层号接在后面（GLM 的第 78 层、StepFun 的 92–94 层）。两种都折成
    「主干 n_main 层 + MTP 第 ord 块」，ord 从 0 数，所以下游不必知道是哪一种。

    返回 (blocks, stacks, glue, issues)：stacks 是见到的栈前缀与张量数，提取不到层时
    拿它报诊断 —— 「0 层」最可能的原因就是某个前缀没被认出来。glue 记下哪些局部名是
    栈顶共用的拼接层，它们的算子组按结构定，不过名字正则。
    """
    blocks = defaultdict(lambda: defaultdict(list))
    glue = defaultdict(set)
    stacks, loose, issues = Counter(), defaultdict(list), []
    for name, meta in shapes.items():
        if is_tower(name):
            continue
        m = _BLOCK.match(name)
        if not m:
            d = _LOOSE.match(name)
            if d:
                loose[d.group(1) + "."].append((name, meta))
            continue
        stack = m.group(1) + m.group(2)[:-1]
        idx, rest = int(m.group(3)), m.group(4)
        stacks[stack] += 1
        if _DRAFT_STACK.search(stack):
            kind, ordinal = "mtp", idx
        elif idx >= n_main:
            # 借用主干层号的 MTP 块（GLM / StepFun）
            kind, ordinal = "mtp", idx - n_main
        else:
            kind, ordinal = "main", idx
        local = _EXPERT.sub(".experts.E.", "." + rest)[1:]
        blocks[(kind, ordinal)][local].append((name, meta))

    # 草稿栈共用的拼接层：栈只有一块时归进那一块。静默丢掉的话，图上那一列会少一行，
    # 而「少一行」和「这一族本来没有」长得一样（model-audit 规则 7）
    mtp_ords = sorted(o for k, o in blocks if k == "mtp")
    for pfx, items in loose.items():
        if len(mtp_ords) == 1:
            for name, meta in items:
                local = _EXPERT.sub(".experts.E.", "." + name[len(pfx):])[1:]
                blocks[("mtp", mtp_ords[0])][local].append((name, meta))
                glue[("mtp", mtp_ords[0])].add(local)
        else:
            issues.append(f"{pfx}* 下有 {len(items)} 个不在 layers.N. 里的张量，"
                          f"而 MTP 块有 {len(mtp_ords)} 个，无法归属")
    return blocks, stacks, glue, issues


def tensor_rows(block: dict, packed, mtp_from: int, recurrent: bool, glue=frozenset()):
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
        if meta["dtype"] in CONTAINERS and packed:
            logical[-1] *= 8 // packed["bits"]
            logical_dt = packed["dtype"]
        base = _WEIGHT_TAIL.sub("", local)
        sc = None
        for suffix in _SCALE_SUFFIXES:
            if base + suffix in block:
                sc = block[base + suffix][0][1]
                break
        row = {
            "tensor": local,
            # 栈顶共用的拼接层整组记 mtp_glue：它们「不属于任何一块」是结构事实，
            # 比任何名字正则都硬
            "group": "mtp_glue" if local in glue else op_group(local, recurrent),
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
            if not any(base + s in block for s in ("", ".weight", ".weight_packed")):
                issues.append(f"{local}: 找不到对应的权重张量")
    return rows, issues


# 逐层图里叫 index_k，记录里叫 indexer —— 两边沿用各自已有的名字，这里对一次
_COMPONENT_KEY = {"global_kv": "global_kv", "index_k": "indexer", "local_window": "local_window"}


def kv_storage(model_path: str, fallback: int) -> dict:
    """每个缓存分量的每元素字节（含 scale）与口径。"""
    comps = {}
    if model_path and os.path.exists(model_path):
        prec = (yaml.safe_load(open(model_path)) or {}).get("precision") or {}
        comps = prec.get("kv_cache_components") or {}
    out = {}
    for key, rec_key in _COMPONENT_KEY.items():
        c = comps.get(rec_key)
        d = dtypes.nbytes(c.get("dtype")) if c else 0
        if not d:
            out[key] = {"bytes_per_element": fallback, "basis": "dtype_fallback"}
            continue
        blk = c.get("block")
        sb = dtypes.nbytes(c.get("scale_dtype")) if blk and c.get("scale_dtype") else 0
        out[key] = {
            "dtype": dtypes.norm(c["dtype"]), "block": blk,
            "scale_dtype": dtypes.norm(c["scale_dtype"]) if sb else None,
            "data_bytes_per_element": d,
            "scale_bytes_per_element": sb / blk if sb else 0,
            "bytes_per_element": d + (sb / blk if sb else 0),
            "basis": "theoretical_floor",
        }
    return out


def recurrent_state(block: dict, heads, issue_tag: str, issues: list):
    """递归式注意力层的状态：每序列固定，**不随上下文增长**，所以不折每 token 值。

    两段分开记，口径不同：

      * 卷积状态整条由权重定：因果深度卷积的权重形状是 [通道, 1, 核宽]，状态要留住
        核宽 − 1 个历史位置。两个因子都在形状里，没有一个来自 config。
      * 递归矩阵状态的形状因子在 config 里（头数与两个 head_dim）。**先与权重的 QKV
        投影输出维对账**：输出维合计应等于 2 × key_heads × key_head_dim +
        value_heads × value_head_dim（q、k 同宽，v 另一宽）。对不上就不记、只报 ——
        照抄 config 的头数乘积正是 model-audit 缓存账第 2 问点名的那种高估。

    **不折字节。** 状态的落盘精度没有一手来源（Qwen 声明 mamba_ssm_dtype=float32，
    Kimi 没有对应字段），这里只记元素数。一边有字节一边没有，最容易被拿去算倍数。
    """
    out = {"covers": "per_sequence", "basis": "weights_measured"}
    convs = [(l, block[l][0][1]["shape"]) for l in sorted(block)
             if "conv1d" in l and len(block[l][0][1]["shape"]) == 3]
    if convs:
        kernels = {s[2] for _, s in convs}
        if len(kernels) == 1:
            k = convs[0][1][2]
            ch = sum(s[0] for _, s in convs)
            out.update({"conv_channels": ch, "conv_kernel": k,
                        "conv_state_elems": ch * (k - 1),
                        "conv_sources": [l for l, _ in convs]})
        else:
            issues.append(f"{issue_tag} conv1d 核宽不一致 {sorted(kernels)}，卷积状态不计")
    if heads:
        qkv = sorted(l for l in block if _REC_QKV.match(l))
        got = sum(block[l][0][1]["shape"][0] for l in qkv)
        want = 2 * heads["key_heads"] * heads["key_head_dim"] \
            + heads["value_heads"] * heads["value_head_dim"]
        if qkv and got == want:
            out.update({
                "matrix_state_elems": heads["value_heads"] * heads["key_head_dim"]
                * heads["value_head_dim"],
                "matrix_state_factors": {k: heads[k] for k in
                                         ("value_heads", "key_head_dim", "value_head_dim")},
                "matrix_state_basis": "config_derived",
                "factors_source": "config " + heads["fields"],
                "factors_check": f"QKV 投影输出维合计 {got} = 2×{heads['key_heads']}×"
                                 f"{heads['key_head_dim']} + {heads['value_heads']}×"
                                 f"{heads['value_head_dim']}，与权重一致",
            })
        elif qkv:
            issues.append(f"{issue_tag} 递归状态因子对账不过：QKV 投影输出维合计 {got}，"
                          f"config {heads['fields']} 推出 {want} —— 不记矩阵状态")
    out["elems_per_sequence"] = out.get("conv_state_elems", 0) + out.get("matrix_state_elems", 0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id", required=True)
    ap.add_argument("--facts-dir", default="facts")
    ap.add_argument("--out-dir", default="data/derived")
    ap.add_argument("--kv-bytes", type=int, default=2,
                    help="记录里没有分量精度时的兜底字节数，默认 BF16。只影响字节换算，不影响元素数")
    ap.add_argument("--model", help="模型 YAML，读 precision.kv_cache_components；缺省按 id 在 data/models 下找")
    args = ap.parse_args()
    if not args.model:
        hit = glob.glob(os.path.join("data", "models", "**", f"{args.id}.yaml"), recursive=True)
        args.model = hit[0] if hit else None
    storage = kv_storage(args.model, args.kv_bytes)

    fdir = os.path.join(args.facts_dir, args.id)
    sp, cp = os.path.join(fdir, "shapes.json"), os.path.join(fdir, "cfg.json")
    shapes = json.load(open(sp))
    if not shapes:
        sys.exit(f"{sp} 是空的。零差异孪生版故意不拉全量清单，逐层图要对它的孪生版生成")
    cfg = json.load(open(cp))
    t = cfg.get("text_config") or cfg
    n_main = t["num_hidden_layers"]
    packed = packed_format(cfg) or packed_format(t)
    ratios = t.get("compress_ratios") or []
    window = t.get("sliding_window")
    # 逐层的注意力类型声明。只用来分「全局 / 滑窗」，层的角色仍从权重判
    ltypes = t.get("layer_types") or []
    heads = linear_heads(t)
    pool_cfg = next((t[k] for k in _POOL_KEYS if t.get(k)), None)
    topk_cfg = next((t[k] for k in _TOPK_KEYS if t.get(k)), None)

    blocks, stacks, glue, issues = collect(shapes, n_main)
    main_stacks = [s for s in stacks if not _DRAFT_STACK.search(s)]
    if len(main_stacks) > 1:
        sys.exit(f"认出多个主干栈 {main_stacks} —— 判栈的前缀规则在这份清单上不成立，"
                 f"先看清楚是哪一个才是语言主干，不要让它们混成一叠层")

    layers, classes, sig_to_class = [], [], {}
    lw_source = None

    for (kind, idx) in sorted(blocks, key=lambda k: (k[0] != "main", k[1])):
        block = blocks[(kind, idx)]
        has = set(block)
        rec = is_recurrent(has)
        rows, iss = tensor_rows(block, packed, n_main, rec, glue[(kind, idx)])
        issues += [f"{kind}.{idx} {i}" for i in iss]
        flat_idx = idx if kind == "main" else n_main + idx
        tag = f"{kind}.{idx}"

        def out_dim(*locals_):
            """这些张量的输出维之和；缺一个就返回 None。

            缓存宽度是产出张量的输出维，不是 config 的头数乘积 ——
            index K 在 index 头之间 MQA 共享，按头数乘积算高估 32 倍。
            """
            if not all(l in block for l in locals_):
                return None
            return sum(block[l][0][1]["shape"][0] for l in locals_)

        def pick(component):
            for group in KV_SOURCES[component]:
                w = out_dim(*group)
                if w:
                    return w, " + ".join(group)
            return None, None

        # 层类型只回答「这层的 KV 是随上下文增长的还是窗口内固定的」。
        # config 没有这份声明时（DeepSeek / GLM / MiniMax / Kimi）两者都放行，
        # 由上面的候选表是否命中来决定 —— 不命中就本来没有那个分量
        ltype = ltypes[flat_idx] if flat_idx < len(ltypes) else None
        is_sliding = bool(window) and ltype in (None, "sliding_attention")
        is_global = not rec and ltype in (None, "full_attention")

        roles = []
        global_w, global_src = (pick("global_kv") if is_global else (None, None))
        index_w, index_src = pick("index_k")
        if global_w:
            roles.append("kv_source")
            if "attn.compressor.wkv.weight" in has and "attn.compressor.wgate.weight" not in has:
                roles.append("compressor_ungated")
        if index_w:
            roles.append("index_source")
        elif any(r["group"] == "indexer" for r in rows):
            roles.append("index_rescore")
        if any(k.startswith("engram.") for k in has):
            roles.append("engram")

        ratio = ratios[flat_idx] if flat_idx < len(ratios) else None
        # 没有 compress_ratios 的族系：一个 token 一个条目，折算比 1。
        # 不要写成 `ratio or 1` —— DeepSeek 的 L0 比值本来就是 0（那层不产全局 KV），
        # or 会把 0 变成 1，凭空造出一个条目
        fold = ratio if ratios else 1
        cache = {}
        lw, lw_src = (pick("local_window") if is_sliding else (None, None))
        if lw and window:
            lw_source = lw_source or lw_src
            cache["local_window"] = {
                "entry_elems": lw, "window": window,
                "elems_per_sequence": lw * window,
                "bytes_per_sequence": lw * window * storage["local_window"]["bytes_per_element"],
            }
        for key, w, src in (("global_kv", global_w, global_src), ("index_k", index_w, index_src)):
            if not w:
                continue
            c = {"entry_elems": w, "source": src, "compress_ratio": ratio}
            if fold:
                c["per_token_elems"] = w / fold
                c["per_token_bytes"] = w / fold * storage[key]["bytes_per_element"]
            cache[key] = c
        if rec:
            cache["recurrent_state"] = recurrent_state(block, heads, tag, issues)

        groups = sorted({r["group"] for r in rows}, key=lambda g: [k for k, *_ in OP_GROUPS + [("other",)]].index(g))
        bytes_by_group = defaultdict(int)
        for r in rows:
            bytes_by_group[r["group"]] += r["bytes_each"] * r["count"]
        # decode 每 token 实际搬运：路由专家只读 top_k 份，其余组全读。
        # MTP 块里的专家池与 top_k 是 DSpark 自己的（128 选 3），不是主干的 384 选 6
        if kind == "mtp" and t.get("dspark_n_routed_experts"):
            pool, k = t["dspark_n_routed_experts"], t.get("dspark_num_experts_per_tok")
        else:
            pool, k = pool_cfg, topk_cfg
        decode_by_group = dict(bytes_by_group)
        exp_rows = [r for r in rows if r["group"] == "expert"]
        n_seen = max((r["count"] for r in exp_rows), default=0)
        if n_seen == 1 and pool and all(len(r["shape"]) >= 3 and r["shape"][0] == pool for r in exp_rows):
            # 堆叠存储：专家维在张量内部（Qwen mlp.experts.gate_up_proj [512, …]、
            # StepFun moe.gate_proj [352, …]）。份数看不出来，专家维是 shape[0]，
            # 和 config 的池大小对上了才认 —— 对不上就走下面那条 issue，不替它猜
            n_seen = pool
        if n_seen and k:
            if pool and pool != n_seen:
                issues.append(f"{tag} 专家数：config {pool}，权重 {n_seen}")
            decode_by_group["expert"] = bytes_by_group["expert"] * k // n_seen
        elif exp_rows and not n_seen:
            issues.append(f"{tag} 有路由专家权重但数不出份数，decode 搬运按全读计")
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
    # 产物侧的闸：空产出与层数不符都是失败，且**不写文件**。
    # 这道检查锚在产物上（提出来几层），因此守不住「层提对了但内容错了」——
    # 那是 checks 与 issues 的事。但「0 层也算成功」连错都报不出来，
    # 本仓为同一个毛病付过一次全站三千多处的代价（STATUS.md 坑 #4）
    if not main_layers:
        sys.exit(f"从 {sp} 一个主干层都没提出来（清单 {len(shapes)} 个张量）。"
                 f"认出的栈前缀：{dict(stacks) or '无'}。"
                 f"最可能是这一族的前缀没被 collect() 的 _BLOCK 认出来，"
                 f"或者全被 is_tower() 当成视觉塔剔掉了")
    if len(main_layers) != n_main:
        sys.exit(f"主干层数对不上：config num_hidden_layers {n_main}，"
                 f"权重提出 {len(main_layers)} 层（MTP {len(layers) - len(main_layers)} 块）。"
                 f"认出的栈前缀：{dict(stacks)}。层数不对的产物不写 —— "
                 f"下游的每 token 缓存合计会按错的层集合求和")

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

    # 逐层类型的声明：各族字段名不同，问的都是同一件事 —— config 说哪些层是另一类，
    # 权重同意吗。只报不改，权重赢（CLAUDE.md）。**这几条只能查到内侧**：
    # config 声明了而权重没有、或反过来；config 压根没声明的差异它一条也查不出
    # （model-audit 规则 11）
    rec_w = sorted(l["index"] for l in main_layers if "recurrent_state" in l["cache"])
    if rec_w:
        declared, src = None, None
        if ltypes:
            declared = [i for i, v in enumerate(ltypes[:n_main]) if v == "linear_attention"]
            src = "layer_types 里的 linear_attention"
        elif (t.get("linear_attn_config") or {}).get("kda_layers"):
            declared = list(t["linear_attn_config"]["kda_layers"])
            src = "linear_attn_config.kda_layers"
        if declared is not None:
            checks.append({"item": "递归式注意力层", "config": declared, "weights": rec_w,
                           "status": "ok" if declared == rec_w else "MISMATCH",
                           "note": f"config 列取自 {src}；权重列是带因果深度卷积与衰减参数"
                                   f"（conv1d / A_log / dt_bias）的层。**这些层不产生每 token KV**，"
                                   f"所以这份层集合错了，每 token 缓存合计就按错的层数求和"})
    idx_w = sorted(l["index"] for l in main_layers if "index_k" in l["cache"])
    if idx_w and "index_source_layer_ids" not in t:
        declared, src = None, None
        if t.get("indexer_types"):
            declared = [i for i, v in enumerate(t["indexer_types"]) if v == "full"]
            src = "indexer_types 里的 full"
        elif ((t.get("sparse_attention_config") or {}).get("sparse_attention_freq")):
            freq = t["sparse_attention_config"]["sparse_attention_freq"]
            declared = [i for i, v in enumerate(freq[:n_main]) if v]
            src = "sparse_attention_config.sparse_attention_freq 的非 0 位"
        elif ltypes and "full_attention" in (t.get("sparse_config") or {}).get("apply_to_layer_types", []):
            declared = [i for i, v in enumerate(ltypes[:n_main]) if v == "full_attention"]
            src = "layer_types 的 full_attention（sparse_config.apply_to_layer_types 指定）"
        if declared is not None:
            mtp_idx = [l["index"] for l in layers if l["kind"] == "mtp" and "index_k" in l["cache"]]
            checks.append({"item": "产出 index K 的层", "config": declared, "weights": idx_w,
                           "status": "ok" if declared == idx_w else "MISMATCH",
                           "note": f"config 列取自 {src}，只覆盖主干；权重列是实际带 index K 投影的主干层"
                                   + (f"。MTP 块另有 {len(mtp_idx)} 块也带（第 "
                                      f"{','.join('MTP' + str(i) for i in mtp_idx)} 块），不在 config 这份列表的长度里"
                                      if mtp_idx else "")})

    totals = {}
    for key in ("global_kv", "index_k"):
        src = [l for l in main_layers if key in l["cache"]]
        if not src:
            continue
        totals[key] = {
            "layers": [l["index"] for l in src],
            "entry_elems_sum": sum(l["cache"][key]["entry_elems"] for l in src),
            "per_token_elems_sum": sum(l["cache"][key].get("per_token_elems", 0) for l in src),
            "per_token_bytes_sum": sum(l["cache"][key].get("per_token_bytes", 0) for l in src),
            "bytes_basis": storage[key]["basis"],
            "note": "per_token_elems_sum 按 compress_ratio 折算（一个条目池化 compress_ratio 个 token）；"
                    "entry_elems_sum 是未折算的条目宽度之和，只作对照"
                    if ratios else
                    "这一族的缓存没有时序压缩，一个 token 一个条目，所以折算前后相等；"
                    "entry_elems_sum 只作对照",
        }
    # 滑窗分主干与 MTP 两段记：权重里 MTP 块也有滑窗投影，投机解码时它们的窗口同样常驻；
    # 只跑主干时只有前一段
    win = {}
    for kind in ("main", "mtp"):
        ls = [l for l in layers if l["kind"] == kind and "local_window" in l["cache"]]
        if ls:
            win[kind] = {"layers": len(ls),
                         "elems_per_sequence": sum(l["cache"]["local_window"]["elems_per_sequence"] for l in ls),
                         "bytes_per_sequence": sum(l["cache"]["local_window"]["bytes_per_sequence"] for l in ls)}
    if win:
        totals["local_window"] = {
            **win,
            "elems_per_sequence": sum(v["elems_per_sequence"] for v in win.values()),
            "bytes_per_sequence": sum(v["bytes_per_sequence"] for v in win.values()),
            "bytes_basis": storage["local_window"]["basis"],
        }
    rec_layers = [l for l in layers if "recurrent_state" in l["cache"]]
    if rec_layers:
        totals["recurrent_state"] = {
            "layers": len(rec_layers),
            "layer_ids": [l["index"] for l in rec_layers if l["kind"] == "main"],
            "conv_state_elems": sum(l["cache"]["recurrent_state"].get("conv_state_elems", 0) for l in rec_layers),
            "matrix_state_elems": sum(l["cache"]["recurrent_state"].get("matrix_state_elems", 0) for l in rec_layers),
            "elems_per_sequence": sum(l["cache"]["recurrent_state"]["elems_per_sequence"] for l in rec_layers),
            "covers": "per_sequence",
            "note": f"{len(rec_layers)} 层是递归式注意力，**每 token 缓存贡献为 0** —— "
                    f"上面的每 token 合计只来自其余层，不是全部层的平均。状态每序列固定，"
                    f"和每 token 量不是同一个口径，两者不要相加也不要算倍数",
        }

    extra = {}
    if t.get("dspark_target_layer_ids"):
        extra["dspark_target_layers"] = t["dspark_target_layer_ids"]
    if t.get("candidate_source_layer_id") is not None:
        extra["candidate_source_layer"] = t["candidate_source_layer_id"]

    present_groups = {r["group"] for c in classes for r in c["tensors"]}
    present_cache = {k for l in layers for k in l["cache"]}
    cache_basis = {"entry_elems": "产出张量的输出维，权重实测"}
    cache_basis["per_token_elems"] = (
        "entry_elems ÷ compress_ratio。一个条目池化 compress_ratio 个 token —— "
        "参考实现的压缩器是 r:1 时序池化（比值 2 的组 compress_kv 为 L/2 条），"
        "与权重两处结构一致：比值为 1 的源层恰好没有 compressor.wgate（单 token "
        "无须加权汇聚），indexer.wk 的输入维等于压缩器输出维（index K 跟着同一个 r 折）"
        if ratios else
        "条目宽度即每 token 元素数：这一族的缓存没有时序压缩，一个 token 一个条目，折算比 1"
    )
    if "local_window" in present_cache:
        cache_basis["local_window"] = (
            f"{_WEIGHT_TAIL.sub('', lw_source or '')} 输出维（实测）× sliding_window（config），"
            f"每序列固定，不随上下文增长。主干与 MTP 块分开合计"
        )
    cache_basis["bytes"] = (
        "元素数 × kv_storage 里该分量的每元素字节（含 scale）。basis=theoretical_floor 是"
        "精确计数的理论下限，不含分页、环页、对齐粒度，实际部署在它之上；"
        "basis=dtype_fallback 是记录没给精度时的兜底，两种不能混比"
    )
    if "recurrent_state" in present_cache:
        cache_basis["recurrent_state"] = (
            "递归式注意力层的状态，每序列固定。卷积状态 = conv1d 通道 ×（核宽 − 1），"
            "两个因子都在权重形状里；矩阵状态 = value_heads × key_head_dim × value_head_dim，"
            "因子来自 config 但先与 QKV 投影输出维对账（见每层的 factors_check）。"
            "**不折字节也不折每 token 值** —— 状态精度没有一手来源，而「每序列固定」"
            "和「随上下文增长」不是一个量"
        )

    out = {
        "id": args.id,
        "generated_by": "scripts/extract_layers.py",
        "generated_at": datetime.date.today().isoformat(),
        "inputs": {"shapes": sp, "shapes_sha256_16": sha(sp), "cfg": cp, "cfg_sha256_16": sha(cp),
                   "tensors": len(shapes)},
        "op_groups": [{"key": k, "label": lb} for k, lb, _ in OP_GROUPS if k in present_groups],
        # 分量的格式各不相同，单值装不下；字节含 scale
        "kv_storage": {**{k: v for k, v in storage.items() if k in present_cache},
                       "source": args.model},
        "bytes_semantics": "bytes_by_group 是存储（含 scale，全部专家）；decode_bytes_by_group 是每 token "
                           "搬运，路由专家按 top_k / 专家数折算，其余组全读",
        "cache_basis": cache_basis,
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
