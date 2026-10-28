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
_CONTAINERS = {"INT8", "UINT8"}


def packing_factor(logical_dtype: str, container_dtype: str) -> int:
    """每个容器元素里装了几个逻辑参数。

    声明的逻辑类型不足 8 位、而实际张量是 8 位整型容器时，说明做了打包。
    返回 1 表示没有打包。这个因子只用于参数量，不用于字节数 —— 容器就是真实存储。
    """
    bits = BITS.get(norm(logical_dtype))
    if not bits or norm(container_dtype) not in _CONTAINERS:
        return 1
    return max(8 // bits, 1)


def is_scale_tensor(name: str, dtype: str) -> bool:
    """判断一个张量是 scale 容器还是模型参数本体。

    参数量统计要排除 scale，字节统计要包含 scale —— 两个口径不同，
    混了会让「参数量」和「权重存储」同时错。
    """
    n = name.lower()
    if n.endswith(".scale") or n.endswith("_scale") or ".scale_" in n:
        return True
    if n.endswith(".scale_inv") or n.endswith(".weight_scale"):
        return True
    return norm(dtype) == "FP8_E8M0"


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
