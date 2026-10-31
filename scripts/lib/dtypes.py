"""数据类型的字节账。

两套记号都要认：safetensors 头里的（F32 / BF16 / F8_E4M3）和 schema 里的（FP32 / FP8_E4M3）。
低位格式的 scale 在 safetensors 里是独立张量，所以按张量求和天然包含了 scale 开销，
不需要再估算 —— 这是 shapes 路比 config 路可信的原因。
"""

# 规范名 -> 每元素字节数
BYTES = {
    "FP64": 8,
    "FP32": 4,
    "TF32": 4,
    "BF16": 2,
    "FP16": 2,
    "FP8_E4M3": 1,
    "FP8_E5M2": 1,
    "FP8_E8M0": 1,  # scale 容器，本身占 1 字节
    "MXFP8": 1,
    "HIF8": 1,
    "FP6": 0.75,
    "FP4": 0.5,
    "MXFP4": 0.5,
    "NVFP4": 0.5,
    "HIF4": 0.5,
    "INT64": 8,
    "INT32": 4,
    "INT16": 2,
    "INT8": 1,
    "UINT8": 1,
    "INT4": 0.5,
    "BOOL": 1,
}

# safetensors / HF 的别名 -> 规范名
_ALIAS = {
    "F64": "FP64", "F32": "FP32", "F16": "FP16", "BF16": "BF16",
    "F8_E4M3": "FP8_E4M3", "F8_E5M2": "FP8_E5M2", "F8_E8M0": "FP8_E8M0",
    "FLOAT8_E4M3FN": "FP8_E4M3", "FLOAT8_E5M2": "FP8_E5M2",
    "I64": "INT64", "I32": "INT32", "I16": "INT16", "I8": "INT8",
    "U8": "UINT8", "BOOL": "BOOL",
    "FLOAT32": "FP32", "FLOAT16": "FP16", "BFLOAT16": "BF16",
    "FP8": "FP8_E4M3",  # 量化配置里常写成裸 fp8
    "FP4": "FP4",
}


def norm(dt: str) -> str:
    """把任意写法归一到规范名。认不出来就原样抛回，由调用方决定怎么处理。"""
    if not dt:
        return "UNKNOWN"
    k = dt.strip().upper().replace("-", "_")
    k = _ALIAS.get(k, k)
    return k


def nbytes(dt: str) -> float:
    """每元素字节数。未知类型返回 0 并由调用方记成 unknown，不要瞎猜。"""
    return BYTES.get(norm(dt), 0.0)


# 亚字节类型的位宽。低位权重常被打包进整型容器存放：
# safetensors 里看到的是容器（如 I8），numel 是容器数而不是参数数。
BITS = {"FP6": 6, "FP4": 4, "MXFP4": 4, "NVFP4": 4, "HIF4": 4, "INT4": 4}
# 整型容器 -> 容器位宽。一字节容器装 2 个 FP4 是常见形态；compressed-tensors 的
# pack-quantized 用的是 **I32**，一个容器装 8 个 INT4（末维 = 逻辑维 / 8）。
# 少了 INT32 这一档，整条 MoE 的参数量会静默少算 8 倍 —— kimi-k2-thinking 与
# kimi-k2.5 都踩过，是 Gate B 的 HF safetensors tally 对账把它接住的。
_CONTAINERS = {"INT8": 8, "UINT8": 8, "INT32": 32}


def is_container(dtype: str) -> bool:
    """这个 dtype 有没有可能装着打包的低位权重。"""
    return norm(dtype) in _CONTAINERS


def packing_factor(logical_dtype: str, container_dtype: str) -> int:
    """每个容器元素里装了几个逻辑参数。

    声明的逻辑类型不足容器位宽、而实际张量是整型容器时，说明做了打包。
    返回 1 表示没有打包。这个因子只用于参数量，不用于字节数 —— 容器就是真实存储。
    """
    bits = BITS.get(norm(logical_dtype))
    width = _CONTAINERS.get(norm(container_dtype))
    if not bits or not width:
        return 1
    return max(width // bits, 1)


# scale 容器的叶名。各代不一样：ds-v3 一代是 weight_scale_inv，V4 一代改成了 scale
# （见 ds-v4-flash.yaml 的 lineage 条目）。另两个是旧代码按子串判断时写的变体。
#
# **scales / qzeros 是 GPTQ 那一套的量化参数**（autogptq 的叶名，复数、带 q 前缀，
# 和上面四个一个都配不上）。两者都是「每 group 一个值」的量化参数而不是模型参数：
# Qwen3.5-27B-GPTQ-Int4 里 scales 是 133,693,440 个 F16、qzeros 是 16,711,680 个
# 打包进 I32 的零点。不排掉它们，参数量会比主仓多出 1.5 亿，而两个仓本该逐位相同 ——
# 真实情况正是如此：排掉之后 27,781,427,952 与主仓分毫不差。
# HF 自己的 tally 也排掉它们（那份只报 I32 / BF16 / F32 三档，F16 的 scales 根本不出现）。
_SCALE_LEAVES = {"scale", "scale_inv", "weight_scale", "weight_scale_inv",
                 "scales", "qzeros"}


def is_scale_tensor(name: str, dtype: str) -> bool:
    """判断一个张量是 scale 容器还是模型参数本体。

    参数量统计要排除 scale，字节统计要包含 scale —— 两个口径不同，
    混了会让「参数量」和「权重存储」同时错。

    按**整段叶名**匹配，不用「含 scale」的子串判断。子串判断两头都错过：
    `_scale` 结尾这一条把 mHC 的 hc_attn_scale / hc_ffn_scale / hc_head_scale
    当成容器排掉（它们是真参数），而 `.scale_inv` 这条又认不出 ds-v3 一代的
    `weight_scale_inv`（叶名里 scale 前面是下划线不是点），于是 45808 个 scale
    被算成了参数。新后缀出现时这里宁可漏判也不误判：漏判让参数量偏高一点点，
    误判会把真参数整个抹掉，而后者在页面上看不出来。
    """
    if name.lower().rsplit(".", 1)[-1] in _SCALE_LEAVES:
        return True
    return norm(dtype) == "FP8_E8M0"


# 量化格式随权重一起发的元数据张量。compressed-tensors 的 pack-quantized 会给每个
# 打包权重配一个 weight_shape（I32 [2]，记打包前的逻辑形状）。它占字节，但既不是
# 参数也不是 scale —— HF 的 tally 把它排除在外，本仓原先按叶名白名单算进了参数。
# 单张量只有 2 个元素，但 69120 个张量就是 138,240，对账差不上。
#
# g_idx 是 GPTQ 的 group 索引（I32 [in_features]，desc_act 为 false 时是恒等排列）。
# 它和 scales / qzeros 不同档：那两个是量化参数，这个是重排下标 —— 所以归元数据。
# 两类的字节都算、参数都不算，分开只为「scale_elements 这个对账量别混进索引」。
_METADATA_LEAVES = {"weight_shape", "g_idx"}


def is_metadata_tensor(name: str, dtype: str = "") -> bool:
    """量化格式的元数据张量：占存储，但不计入参数量。"""
    return name.lower().rsplit(".", 1)[-1] in _METADATA_LEAVES


def counts_as_param(name: str, dtype: str) -> bool:
    """这个张量要不要计入参数量。scale 容器与格式元数据都不算，字节照样要算。"""
    return not is_scale_tensor(name, dtype) and not is_metadata_tensor(name, dtype)


def blocked_bytes(numel: float, data_dtype: str, block: int, scale_dtype: str = "FP8_E8M0") -> float:
    """按 block 量化的存储字节：数据 + 每 block 一个 scale。

    只在没有 shapes.json、必须从 config 估算时用。有 shapes 就直接按张量求和。
    """
    d = nbytes(data_dtype)
    s = nbytes(scale_dtype)
    if block <= 0:
        return numel * d
    return numel * d + (numel / block) * s


def fmt_bytes(b: float) -> str:
    """给人看的字节数。"""
    for unit, div in (("TiB", 2**40), ("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if b >= div:
            return f"{b / div:.2f} {unit}"
    return f"{b:.0f} B"


def fmt_count(n: float) -> str:
    """给人看的参数量。"""
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.3g}{unit}"
    return f"{n:.0f}"
