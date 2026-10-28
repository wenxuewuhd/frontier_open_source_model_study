# 前沿开源大模型演进跟踪

从 infra 视角跟踪主流开源大模型与新型范式的演进。每次模型变化都回答三个问题：
改了什么、对负载意味着什么、下一代 infra 要准备什么。

方案与分期见 [PLAN.md](PLAN.md)。当前进度：**P0-A 完成**（schema + 脚本 + 首个压力测试用例）。

## 这个仓库是什么

数据和叙述分开：数字只出现在 `data/` 的 YAML 里，可校验、可复算；分析引用数据，不复制数字。

```
data/
  schema/           JSON Schema：模型记录、算子表
  models/<家族>/    手写事实。每个字段带证据口径
  derived/          推导值，由脚本生成，不手写
facts/              取数缓存（不进仓库，可按 pinned sha 复现）
scripts/
  fetch_hf.py       HF 仓库 -> facts/（只读 safetensors 头，不下权重）
  derive.py         config + 权重清单 -> 负载特征 + 交叉核对
  validate.py       schema + 证据覆盖 + 防泄漏
  mk_ops_table.py   算子表 YAML -> xlsx
tests/fixtures/     渲染用的合成样例
data-private/       内部 overlay（不进仓库）：算子表、自测性能、硬件相关分析
```

## 证据口径

每个字段都要能在 `evidence` 块里找到归属，支持前缀继承（`arch: official` 覆盖整棵子树，更深的路径优先）：

| 口径 | 含义 |
|---|---|
| `official` | 官方披露 |
| `derived` | 由已知量算出 |
| `judgement` | 解释判断 |
| `unknown` | 未知 |

`validate.py --list-judgements` 列出所有判断类字段。**判断类必须人看过，`official`/`derived` 自动过** —— 判断类才是会错的。

## 流程与人工 review 点

```
S0  选题        本期追哪些模型/范式                      ◀ Gate A 人
S1  取数  自动  fetch_hf.py
S2  推导  自动  derive.py，双路对账                      ◀ Gate B 人
S3  深挖  agent 读推理代码+技术报告 -> arch/精度/瓶颈
S4  算子表 skill npu-op-table -> rows/params/groups+锚点
S5  核查  agent 对抗式逐条核源码 -> 修正留痕
S6  校验  自动  validate.py
S7  渲染  自动  公开站；叠 overlay 出内部站
S8  简报        双周报草稿                               ◀ Gate C 人
```

人只进三个闸口，多了就没人真看。**Gate B 是最关键的一个**：`derive.py` 走两条路——从 config 套公式、从权重清单逐张量求和——两条都能算的量一律交叉核对，不一致就报，不替人选一个。

## 用法

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/fetch_hf.py <org/repo> --id <模型id>
.venv/bin/python scripts/derive.py --id <模型id> --model data/models/<家族>/<模型id>.yaml --ep 4
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/mk_ops_table.py <算子表.yaml>
```

`derive.py` 在有不一致项时返回非零，`validate.py` 在有错误时返回非零，都可以直接给 CI 用。

## 三个口径不能混

这是本仓最容易出错的地方，工具里已经固化：

| 口径 | 含不含 scale | 低位打包 | 用途 |
|---|---|---|---|
| 参数量 | 排除 | **还原**（numel 是容器数，不是参数数） | 对外说「多大的模型」 |
| 权重存储 | 包含 | 不还原（容器就是真实存储） | 显存与带宽规划 |
| 激活参数 | 排除 | 还原 | decode 带宽需求 |

## 公开与内部

公开站和内部站是同一套代码构建两次：

- **公开仓**：架构、精度、公开引擎支持、公开 benchmark、新范式观察。
- **内部 overlay**（`data-private/`）：算子表、自测性能、硬件相关的瓶颈分析。

`validate.py` 默认拒绝 `visibility: internal`、`source.type: self` 和 `basis: measured`。脱敏做不干净，所以这里不做字段级脱敏，而是整条记录拒绝；内部构建加 `--allow-internal` 放行。

## 追踪对象

每个家族追**最新旗舰**。截至 2026-10-28 的选取结果：

| 家族 | 旗舰 | 发布 | 状态 |
|---|---|---|---|
| DeepSeek | `DeepSeek-V4.1-Flash` | 2026-10-16 | 已录 |
| Qwen | `Qwen3.8-2.4T-A95B` | 2026-08-12 | 已录 |
| GLM | `GLM-5.3` | 2026-10-14 | 待录 |
| MiniMax | `MiniMax-M3` | 2026-10-26 | 待录 |
| Kimi | `Kimi-K3` | 2026-09-02 | 待录 |
| 阶跃星辰 | `Step-5-Preview` | 2026-10-15 | 待录 |

较小档位只在有结构对照价值时才录，并标 `lineage.track: side`（如 `qwen3.8-27b`）。

## 用例

schema 由刻意选得差异最大的几个用例撑出来，而不是从一个模型推广。

### P0-A · DeepSeek-V4.1-Flash

字段覆盖最宽：MLA、细粒度 MoE、跨层 KV 共享、稀疏选块、残差流改造、外置查表、自带草稿模型、混合精度打包。它撑出四处修改，都是会静默出错的那类问题：

1. **低位打包**：FP4 权重两个一字节装进 INT8 容器，`numel` 是容器数。直接求和把参数量少算一半（485B vs 实际 763B）。
2. **跨层 KV 共享**：只有 4 层产出 KV、其余层跨层读取，indexer 另装在 8 层上。按总层数乘高估近十倍。
3. **抽样快照**：权重清单只覆盖部分层时，形状可用但参数量不可用。少算 63% 且不报错 —— 现在会拦住。
4. **口径差异**：声明的 748B 不含草稿模型。不把口径写成机器可读的，就会报出一个假的 2% 差异。

修完后对账闭合到 0.13%（声明 748B vs 实测 748.98B）。

### P0-D · Qwen3.8-27B

原计划拿一个「结构朴素」的旗舰作对照，但调研发现**2026 年的旗舰已无朴素结构**：Qwen3.8 全线是混合线性注意力（27B 是 48 层 linear + 16 层 full），GLM-5.3 是 MLA + DSA 稀疏。于是改选与 V4.1 差异最大的这一个 —— dense FFN、混合线性注意力、不同的视觉融合路径，三个轴都不同。

它又撑出三处修改，其中两处是概念性的：

1. **混合结构的显存模型要分两个分量**。full attention 的 KV 随上下文线性增长，线性注意力的递归状态每序列固定、与上下文无关。原来的 `kv_per_token` 只返回一个数，表达不了，已重构成 `per_token_bytes` + `per_sequence_bytes`。这对后面所有混合结构的模型都适用。
2. **`lm_head` 必须计入带宽，输入词表不能**。原来把两者归成一类排除掉，理由是「每 token 只触及极少行」—— 这对 embed 成立，对 lm_head 完全不成立（每步对全词表做 GEMM）。Qwen 的 248K 词表让这个 bug 显形。
3. **官方「激活参数」≠ 带宽相关的激活参数**。前者含输入词表，后者不含；差 4% 且概念不同。现在两个数分开报，不混成一个。

顺带暴露了一件原来看不见的事：重构后按分量拆解，发现 V4.1 的 **indexer 缓存 65536 B/token 是 full attention KV 的 14 倍** —— 它的 KV 压力主要来自稀疏选块的索引，不是注意力本身。

### Qwen3.8-2.4T-A95B（家族旗舰）

补录旗舰时又出两处，都来自「万亿参数这一档」的现实：

1. **取数要并发**。213 个分片、4.45 TiB，串行读头要十几分钟。`fetch_hf.py` 加了线程池（默认 8 并发），并把失败分片单独报出来 —— 清单缺分片会让参数量静默少算。
2. **专家是堆叠存储的**。每层一个 `[512, 4096, 8192]` 张量装下全部 512 个专家（gate 与 up 还融合在一起），所以按 `experts.<n>.` 命名数专家的办法失效。改成从 3 维张量的首维取，并把布局记进 `expert_storage`。这不只是统计问题：**grouped GEMM 吃 3 维权重，算子签名和 EP 切分方式都与逐专家存储不同**。

还有一个反向的收获：config 没给线性注意力的状态布局，而按 key 头数（16）还是 value 头数（128）展开差 8 倍。**权重清单把它定了** —— 每层有 `linear_attn.A_log [128]` 和 `dt_bias [128]`，衰减参数每状态头一个，故状态头数为 128；`in_proj_qkv [20480, 8192]` 也对得上（q 16×128 + k 16×128 + v 128×128）。配置里查不到的容量关键参数，权重清单往往能回答。

### 现在的状态

| | V4.1-Flash | Qwen3.8-2.4T-A95B | Qwen3.8-27B |
|---|---|---|---|
| 总参数（实测） | 763.2B | 2.45T | 27.8B |
| 权重存储 | 475 GiB | 4.45 TiB | 51.75 GiB |
| 激活（带宽口径） | 15.8B | 93.3B | 25.6B |
| 激活（可比官方） | 16.5B（卡称 16B） | 95.3B（名称称 95B） | 26.9B |
| 注意力显存 | 70144 B/token | 94208 B/token + 554 MiB/序列 | 65536 B/token + 145.5 MiB/序列 |
| FFN | MoE 384×top-6 | MoE 512×top-10（堆叠） | dense |

三份记录都通过 schema 校验、证据覆盖检查和交叉核对，`derive.py`、`validate.py`、`mk_ops_table.py` 均返回 0。P0 的 schema 部分到此收口。
