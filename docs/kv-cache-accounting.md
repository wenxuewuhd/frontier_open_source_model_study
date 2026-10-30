# 缓存账的口径

显存账是本仓错得最多的地方：同一个 KV 乘式被改过四轮，每轮都推翻上一轮刚提交的数。
四轮的案情在 [README.md](../README.md) 的「用例」一节，提问顺序在 `model-audit` skill 里。
这里只定义口径，并给一份逐格填满的账作样板。

**数字的真相在记录里**（`data/models/` 手写、`data/derived/` 由 `derive.py` 生成），不在这份文档里。
下面引的数都注明了出处与版本，是为了说明推导过程长什么样；要用数就去读记录。

## 两根轴

缓存容量有两根彼此独立的轴，混起来算就会把调研进度当成架构改进：

| 轴 | 字段 | 性质 |
|---|---|---|
| **元素数** | `per_token_elems`、各分量的 `elements_per_token` | 由结构决定：条目宽度 × 条目数。与精度无关 |
| **字节数** | `per_token_bytes`、`per_sequence_bytes` | 元素数 × 每元素字节。取决于落盘格式，而格式常常查不到 |

**跨代或跨族系比结构，用元素口径。** 它不会随「这一版查到了落盘格式、那一版还没查到」而变。

## basis：字节数是怎么来的

`bytes_basis` 标明这一列字节是哪种来源，三个值：

| basis | 含义 |
|---|---|
| `theoretical_floor` | 精确计数的**理论下限**：元素 × 落盘格式（含 scale），**不含分页、环页、对齐粒度**。实际部署在它之上 |
| `dtype_fallback` | 记录没给缓存精度，按模型权重 dtype 兜底。是个占位值，不是测量值 |
| `mixed` | 各分量 basis 不同，合计没有单一口径 |

### 不同 basis 之间不算倍数

这是范畴错误，不是精度误差。族谱里 V4 → V4.1 曾显示 **13600 B → 890 B，15.3 倍**：

<!-- kv-basis | 由 scripts/kv_table.py 生成，勿手改 -->
| 版本 | 元素/token | 字节/token | basis |
|---|---|---|---|
| DeepSeek-V4-Flash | **3,440** | 6,880 B | `dtype_fallback` |
| DeepSeek-V4-Pro | **4,924** | 9,848 B | `dtype_fallback` |
| DeepSeek-V4.1-Flash | **1,600** | 890 B | `theoretical_floor` |
<!-- kv-basis 生成区结束 -->

V4 一代的字节是按模型权重 dtype（BF16）兜底来的，V4.1 的是 FP4 + scale 精确计数。
属于结构的只有 **3440 / 1600 = 2.15 倍**（V4-Flash 的元素数在 ds-v4-flash 的 R1 由 6800 改为 3440：比值 4 层的条目宽是 compressor.norm 的 512，不是 wkv 输出的 1024）；另外 3.6 倍是「V4.1 查到了落盘格式、V4 还在兜底」——
调研进度的差。V4 的实际落盘精度**没有来源**，也不必去补：按元素口径写，这个未知就不影响结论。

页面侧由 `data.ts` 的 `diff: 'same-basis'` 守卫拦住（两侧 basis 不同就不出 from→to），
跨代的结构比较改由「KV 元素 / token」那一行承担。守卫只管页面，**正文里写倍数要自己过一遍这条**。

## 一份填满的账：DeepSeek-V4.1-Flash

取 `data/derived/deepseek-v41-flash.json`（上游 `DeepSeek-V4.1-Flash @ 314113f8`）。
三类缓存，每一类的四个因子都单独有来源：

| | 全局压缩 KV | index K | 滑窗局部 KV |
|---|---|---|---|
| 产出张量 | `attn.compressor.wkv` | `attn.indexer.wk` | `attn.wkv` |
| 条目宽度 | 512 | **128**（MQA 共享，不是 32×128） | 512 |
| 层覆盖 | 4 源层 L2/8/14/20 | 同 4 层 | 43 层（含 3 个 MTP 块） |
| 条目数 | 按 `compress_ratio` 折算（2/2/2/1） | 同样折算 | **每序列固定** 128 个位置 |
| 元素 / token | 512 × 2.5 = **1280** | 128 × 2.5 = **320** | 不随上下文增长 |
| 落盘格式 | FP4 块 16 + E4M3 scale | FP4 块 32 + E8M0（MXFP4） | FP8_E4M3 块 32 + E8M0 |
| 每元素字节 | 0.5 + 1/16 = **0.5625** | 0.5 + 1/32 = **0.53125** | 1 + 1/32 = **1.03125** |
| 合计 | 720 B/token | 170 B/token | 2,906,112 B/序列 |

**890 B/token + 2.77 MiB/序列**，1M 上下文 0.869 GiB。前两类的元素数是逐层量出来的（`fetch_kv_widths.py`），
落盘格式取自厂商仓的参考实现 `inference/model.py`（R35 起是直接读的源码，此前是二手转述，滑窗那一格因此错成过块 128），但它仍是参考实现的选择，不是 kernel 或实测显存，evidence 标 `judgement`，R21 继续开着。

四个因子都踩过坑，每个都单独错过一次：

- **宽度**：按 `index_n_heads × index_head_dim` = 4096 算，高估 32 倍（index K 在 index 头之间共享）
- **层覆盖**：indexer 按 8 层算（实际 4 层），以及滑窗按 40 层算（MTP 块也有窗口，是 43）
- **条目数**：没除压缩比 —— 比值 2 的层每 token 只摊半个条目；
  以及把每序列固定的滑窗 KV 逐 token 计费
- **每元素字节**：按 BF16 的 2 B 算，而落盘是 FP4 + scale

**元素数和精度分两处存，不乘进同一个字段** —— 两者可信度不同（一个实测、一个二手）。

## 跨族系对照：先对齐账的形状

比数字之前先比**账的形状**。形状不同，per_token 这一个数就不可比：

<!-- kv-roster | 由 scripts/kv_table.py 生成，勿手改 -->
| 族系旗舰 | 类型 | 缓存分量 | 元素/token | 字节/token | 每序列固定 | basis |
|---|---|---|---|---|---|---|
| DeepSeek-V4.1-Flash | MQA-LATENT | 全局压缩 KV + index K + 滑窗局部 KV | **1,600** | 890 B | 2.77 MiB | `theoretical_floor` |
| Qwen3.8-2.4T-A95B | HYBRID | full attention KV + 线性递归状态 | **47,104** | 94,208 B | 573.56 MiB | `dtype_fallback` |
| GLM-5.3 | MLA | full attention KV + index K | **47,616** | 95,232 B | 0.00 MiB | `dtype_fallback` |
| MiniMax-M3 | GQA | full attention KV + index K | **68,736** | 137,472 B | 0.00 MiB | `dtype_fallback` |
| Kimi-K3 | HYBRID | full attention KV + 线性递归状态 | **13,824** | 27,648 B | 226.41 MiB | `dtype_fallback` |
| Step-5-Preview | *待录* | | | | | |
<!-- kv-roster 生成区结束 -->

行取自 `data/tracking.yaml` 的追踪名册（每个族系的最新旗舰），数取自 `data/derived/`，
由 `scripts/kv_table.py` 生成；`--check` 在表与记录不一致时返回非零。
空格子就是还没做的工作，不要填推测值 —— 类型那一栏会说明是待录、还是记录已有但没 derive。

两件事从这张表读得出来，也是填新一行时要先答的：

- **混合结构必须两个分量。** 线性注意力的递归状态每序列固定、与上下文无关；
  full attention 的 KV 随上下文线性增长。合成一个数表达不了 —— `per_token_bytes` 曾经只有一个数，
  Qwen3.8-27B 逼出了重构
- **有账的行里只有 V4.1 是 `theoretical_floor`。** 其余是兜底值，所以**字节那一列互相之间也不能比**，
  只有元素那一列可比。新录一行时，查不到落盘精度就老实标 `dtype_fallback`，
  不要为了让表好看而把兜底值当测量值
