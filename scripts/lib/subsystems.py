"""张量名 → 子系统的归类。

只此一份。derive.py 按它分参数量与带宽，fetch_dtypes.py 按它挑代表张量测精度 ——
两边用不同的归类会让「参数量里的 expert」和「精度表里的 expert」指的不是同一批张量。

顺序即优先级，先匹配先得。draft 排在 expert 之前是有意的：
mtp.N.ffn.experts.N.w1 属于草稿侧，它的权重要另算一份，不能并进主干 MoE。
"""
import json
import os
import re

SUBSYSTEM = [
    ("retrieval", re.compile(r"engram|memory_table|retriev", re.I)),
    ("vision", re.compile(r"vision|visual|vit\.|aligner|image_", re.I)),
    ("audio", re.compile(r"audio|speech|whisper", re.I)),
    # eh_proj / enorm / hnorm / shared_head 是 DeepSeek V3 一代 MTP 模块的名字 ——
    # 那一代 MTP 借用层索引（第 61 层），名字里没有 mtp，只按 mtp 匹配会把它算进主干
    ("draft", re.compile(r"mtp|draft|dspark|eagle|eh_proj|enorm|hnorm|shared_head", re.I)),
    ("expert", re.compile(r"\.experts?\.\d+\.|\bexperts\.", re.I)),
    ("shared_expert", re.compile(r"shared_expert", re.I)),
    ("router", re.compile(r"\bgate\b|router|e_score|expert_bias", re.I)),
    ("lm_head", re.compile(r"\blm_head\b|output_layer|^head\.weight$", re.I)),
    ("embed", re.compile(r"embed_tokens|\bwte\b|embed_in|^embed\.weight$", re.I)),
    # 必须排在 attention / dense_ffn 之前：mHC 的系数叫 hc_attn_* / hc_ffn_*，
    # 名字里带 attn 和 ffn，但它们是残差系数不是注意力或 FFN 权重，
    # 而且往往是 FP32 —— 混进去会把那两个子系统的精度带偏
    ("residual", re.compile(r"\bhc_[a-z]+_(base|fn|scale)\b|^hc_|residual", re.I)),
    # norm 必须排在 attention / dense_ffn 之前：attn_norm、ffn_norm、kv_norm、q_norm
    # 名字里带 attn / ffn / kv / q，按那两条先匹配会把归一化算成注意力或 FFN 权重。
    # 带 norm 的就是归一化，没有例外，所以提前判定是安全的
    ("norm", re.compile(r"norm|\bln\b|layernorm", re.I)),
    ("attention", re.compile(r"attn|\bwq|\bwk|\bwv|\bwo\b|q_proj|k_proj|v_proj|o_proj|kv_[ab]|q_[ab]|indexer", re.I)),
    ("dense_ffn", re.compile(r"mlp|ffn|gate_proj|up_proj|down_proj|w1|w2|w3", re.I)),
]

# 展示顺序与中文名存在 data/schema/subsystems.json，站点侧读同一个文件 ——
# 在两种语言里各写一份顺序，迟早会在加子系统时只改一边。
_SPEC = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "schema", "subsystems.json")))["subsystems"]

ORDER = [s["key"] for s in _SPEC]
LABEL = {s["key"]: s["label"] for s in _SPEC}


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
