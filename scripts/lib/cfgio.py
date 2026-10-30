"""读 HF config 的唯一入口。

**为什么要有这个模块**：同一份 `config.json`，仓里曾有四个消费点用两套读法 ——
`derive.py` 把 `text_config` 拍平后再取，而 `fetch_dtypes.py` / `extract_layers.py` /
`scaffold.py` 只读顶层。差别在多模态记录上才显形：

    Kimi 的四份 VLM（k2.5 / k2.6 / k2.7-code / k3）把整个 `quantization_config`
    嵌在 `text_config` 下，顶层只剩 `dtype: bfloat16`。

于是逐子系统精度表上那几格渲染成「未录」，而 config 明写着 `group_size: 32` /
`num_bits: 4` / `format: mxfp4-pack-quantized`。**不是没查，是读错了地方** ——
而「未录」这个说法让人以为只要去查就行，没人会想到判据本身有问题。

所以读 config 一律走这里，不要再写 `cfg.get("quantization_config")`。
"""

# 多模态 config 把语言主干放在 text_config 下。这三个是已知的分支键，
# 拍平时跳过它们本身，但 text_config 的内容要提到顶层。
_SUBCONFIGS = ("text_config", "vision_config", "audio_config")


def flatten(cfg: dict) -> dict:
    """多模态 config 拍平成一层：`text_config` 的内容提上来，顶层键不覆盖它。

    顺带认一个别名：**扩散主干的 config 把层数叫 `num_layers`**（Wan 的 WanModel
    就是），而仓里取层数的地方都写死了 `num_hidden_layers`。不认它的后果不是报错，
    是 Gate B 的「层数」那条静默变成 skip —— 双路对账少了一路而没人看得见。
    只在 `num_hidden_layers` 缺席时补，所以两个键都有的记录（glm-4-9b 两处值相同）
    行为不变。
    """
    t = dict(cfg.get("text_config") or {})
    for k, v in cfg.items():
        if k not in _SUBCONFIGS and k not in t:
            t[k] = v
    if t.get("num_hidden_layers") is None and t.get("num_layers") is not None:
        t["num_hidden_layers"] = t["num_layers"]
    return t


def quant_cfg(cfg: dict) -> dict:
    """`quantization_config`，顶层与 `text_config` 下都找。

    返回 {} 而不是 None —— 调用方一律 `.get()`，少一层判空。
    **「取到空字典」和「这个模型没量化」在这里是同一件事**，
    要区分得看别处（清单里有没有 scale 容器）。
    """
    q = cfg.get("quantization_config")
    if q:
        return dict(q)
    q = (cfg.get("text_config") or {}).get("quantization_config")
    return dict(q) if q else {}


def quant_site(cfg: dict) -> str:
    """量化声明在哪一层。记进产物用，好让「读错地方」这类问题自己显形。"""
    if cfg.get("quantization_config"):
        return "root"
    if (cfg.get("text_config") or {}).get("quantization_config"):
        return "text_config"
    return "none"
