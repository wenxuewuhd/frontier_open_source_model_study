"""张量名 → 组件 → 子系统的归类。

只此一份。derive.py 按它分参数量与带宽，fetch_dtypes.py 按它挑代表张量测精度 ——
两边用不同的归类会让「参数量里的 expert」和「精度表里的 expert」指的不是同一批张量。

**两层**。生成类的一份权重里常常同时装着 DiT、文本编码器、VAE、声码器、上采样器，
它们是不同的模型：MiniMax-H3 的文本编码器是 33B 的 Qwen3-VL-32B，LTX-2.3 的单文件里
DiT、VAE、audio VAE、vocoder 四样打包在一起。先按顶层前缀分到组件（component()），
再在组件内跑子系统正则（classify()）—— 不分这一层，text_encoder 的 mlp 会和 DiT 自己的
FFN 合成一桶，一个外部文本编码器被算进 DiT 的参数量。

语言模型的张量没有组件前缀，全部落进缺省桶 backbone，现有行为不变。

顺序即优先级，先匹配先得。draft 排在 expert 之前是有意的：
mtp.N.ffn.experts.N.w1 属于草稿侧，它的权重要另算一份，不能并进主干 MoE。
"""
import json
import os
import re

SUBSYSTEM = [
    ("retrieval", re.compile(r"engram|memory_table|retriev", re.I)),
    # projector / patch_merge 是视觉到语言主干的投影（MiniMax-M3 的 multi_modal_projector 与
    # patch_merge_mlp、Kimi-K3 的 mm_projector、Step-5 的 vit_large_projector）。不加的话
    # patch_merge_mlp 因为带 mlp 落进 dense_ffn，其余落进 other
    ("vision", re.compile(r"vision|visual|vit\.|aligner|image_|projector|patch_merge", re.I)),
    ("audio", re.compile(r"audio|speech|whisper", re.I)),
    # eh_proj / enorm / hnorm / shared_head 是 DeepSeek V3 一代 MTP 模块的名字 ——
    # 那一代 MTP 借用层索引（第 61 层），名字里没有 mtp，只按 mtp 匹配会把它算进主干
    ("draft", re.compile(r"mtp|draft|dspark|eagle|eh_proj|enorm|hnorm|shared_head", re.I)),
    # \.moe\.(gate|up|down)_proj 是 Step-5 的 3 维堆叠专家 [352, …]，名字里没有 experts ——
    # 不认它就落进 dense_ffn，激活参数会把全部专家算成常驻
    ("expert", re.compile(r"\.experts?\.\d+\.|\bexperts\.|\.moe\.(gate|up|down)_proj\b", re.I)),
    # Step-5 写成 share_expert（单数、无 d）
    ("shared_expert", re.compile(r"shared?_expert", re.I)),
    ("router", re.compile(r"\bgate\b|router|e_score|expert_bias", re.I)),
    ("lm_head", re.compile(r"\blm_head\b|output_layer|^head\.weight$", re.I)),
    ("embed", re.compile(r"embed_tokens|\bwte\b|embed_in|^embed\.weight$", re.I)),
    # 必须排在 attention / dense_ffn 之前：mHC 的系数叫 hc_attn_* / hc_ffn_*，
    # 名字里带 attn 和 ffn，但它们是残差系数不是注意力或 FFN 权重，
    # 而且往往是 FP32 —— 混进去会把那两个子系统的精度带偏
    # _res_proj 是 Kimi-K3 的 Attention Residuals 系数（self_attention_res_proj / mlp_res_proj /
    # output_attn_res_proj，均 [1, hidden]）；对应的 *_res_norm 仍归 norm
    ("residual", re.compile(r"\bhc_[a-z]+_(base|fn|scale)\b|^hc_|residual|_res_proj\b", re.I)),
    # norm 必须排在 attention / dense_ffn 之前：attn_norm、ffn_norm、kv_norm、q_norm
    # 名字里带 attn / ffn / kv / q，按那两条先匹配会把归一化算成注意力或 FFN 权重。
    # 带 norm 的就是归一化，没有例外，所以提前判定是安全的
    ("norm", re.compile(r"norm|\bln\b|layernorm", re.I)),
    # indexer / compressor 必须排在 attention 之前：它们挂在 attn. 下，名字带 attn，
    # 但精度和注意力主投影不同 —— DeepSeek V4 一代主投影是 FP8，压缩器整条与
    # indexer.wk 却是 BF16，并成一桶后表上只剩 FP8，看不出 BF16 那一截。
    # indexer 又必须排在 compressor 之前：attn.indexer.compressor.* 是 indexer
    # 自己产出 index K 的投影，归 indexer，不归产出全局 KV 的压缩器
    # index_[qk]_proj 是 MiniMax-M3 的 MSA 索引投影（挂在 self_attn 下、名字带 k_proj）；
    # 它的 index_[qk]_norm 照上面的规则归 norm，与 GLM 的 indexer.k_norm 同口径
    ("indexer", re.compile(r"indexer|\bindex_[qk]_proj\b", re.I)),
    ("compressor", re.compile(r"compressor", re.I)),
    ("attention", re.compile(r"attn|\bwq|\bwk|\bwv|\bwo\b|q_proj|k_proj|v_proj|o_proj|kv_[ab]|q_[ab]", re.I)),
    ("dense_ffn", re.compile(r"mlp|ffn|gate_proj|up_proj|down_proj|w1|w2|w3", re.I)),
]

# 展示顺序与中文名存在 data/schema/subsystems.json，站点侧读同一个文件 ——
# 在两种语言里各写一份顺序，迟早会在加子系统时只改一边。
_SPEC = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "schema", "subsystems.json")))["subsystems"]

ORDER = [s["key"] for s in _SPEC]
LABEL = {s["key"]: s["label"] for s in _SPEC}


_COMPONENTS = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "schema", "components.json")))["components"]

COMPONENT_ORDER = [c["key"] for c in _COMPONENTS]
COMPONENT_LABEL = {c["key"]: c["label"] for c in _COMPONENTS}

# 前缀 -> 组件。最长前缀优先：MAGI-2 的 model.diffusion_model.video_embeddings_connector.
# 既匹配 dit 的 model.diffusion_model. 也匹配 connector 的全称，按长度排就不会被 dit 抢走。
_COMPONENT_PREFIXES = sorted(
    ((pfx, c["key"]) for c in _COMPONENTS for pfx in c.get("prefixes", [])),
    key=lambda kv: -len(kv[0]),
)
DEFAULT_COMPONENT = "backbone"


def component(name: str) -> str:
    """张量名 -> 顶层组件。

    没有任何组件前缀的落 backbone —— 语言模型的全部张量都走这一支，
    所以加这一层不改变语言模型侧的任何输出。
    """
    for pfx, key in _COMPONENT_PREFIXES:
        if name.startswith(pfx):
            return key
    return DEFAULT_COMPONENT


# 组件内还值得按子系统细分的只有主干两类。VAE / 声码器 / 上采样器的内部结构
# 对负载账没有意义，整块算；文本编码器是另一个模型，它的内部拆分属于它自己那份记录。
SPLIT_INSIDE = {"backbone", "dit", "refiner"}


def classify2(name: str, mtp_from: int = None):
    """张量名 -> (组件, 子系统)。

    SPLIT_INSIDE 之外的组件不再细分，子系统直接返回组件名本身 ——
    这样 derive.py 既可以按组件汇总，也可以在主干内按子系统拆，用的是同一次遍历。
    """
    comp = component(name)
    if comp not in SPLIT_INSIDE:
        return comp, comp
    return comp, classify(name, mtp_from)


_LAYER_IDX = re.compile(r"layers?\.(\d+)\.", re.I)


def classify(name: str, mtp_from: int = None) -> str:
    """张量名 -> 子系统。

    mtp_from：主干之后第一个 MTP 层的索引，传 config 的 num_hidden_layers 即可。

    DeepSeek V3 一代的 MTP 借用层索引（num_hidden_layers=61，主干是 0–60，
    MTP 是第 61 层），它的本体张量名里没有任何 mtp/draft 标记 —— 只有
    eh_proj / enorm / hnorm / shared_head 这几个胶水张量带标记。所以光靠名字匹配，
    第 61 层的 1536 个专家、注意力、norm、router、embed 副本会整块算进主干，
    参数量账上凭空多一层。层索引是唯一可靠的判据，但它只能由调用方给。
    """
    if mtp_from is not None:
        m = _LAYER_IDX.search(name)
        if m and int(m.group(1)) >= mtp_from:
            return "draft"
    for label, rx in SUBSYSTEM:
        if rx.search(name):
            return label
    return "other"
