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
.venv/bin/python scripts/fetch_hf.py <org/repo> --id <模型id> --names-only   # 只要张量名，用于跨版本结构比对
.venv/bin/python scripts/scaffold.py --id <模型id> --family <家族> --repo <org/repo>  # 从 config 生成草稿
.venv/bin/python scripts/fetch_dtypes.py --id <模型id> --repo <org/repo>  # 逐子系统测精度，只读最小分片集
.venv/bin/python scripts/apply_dtypes.py                                  # 写进记录，已有值时只比对
.venv/bin/python scripts/derive.py --id <模型id> --model data/models/<家族>/<模型id>.yaml --ep 4
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/mk_ops_table.py <算子表.yaml>
```

`derive.py` 在有不一致项时返回非零，`validate.py` 在有错误时返回非零，都可以直接给 CI 用。

## 网站

```bash
cd site && npm install && npm run dev     # 本地预览
npm run build                             # 产物在 site/dist
```

Astro 静态站，直接读 `data/` 下的 YAML 与 derived JSON，**不另定一套 content collection schema** —— 校验是 `validate.py` 的职责，站点只负责渲染。

页面：总览（追踪名册 + Infra 信号 + 时间线）、家族演进、模型列表、模型详情、平台矩阵、口径说明。

家族页有四块，**都从记录算出来，页面不自带任何表** —— 手写的表会和记录各自漂移，而记录才是 `validate.py` 覆盖的那一份：

| 块 | 内容 | 来源 |
|---|---|---|
| 变化矩阵 | 维度 × 版本，格子里是「字段：旧值 → 新值」 | `changesFor` 逐字段比两份记录，不经过任何人写的文字 |
| 逐维度演进 | 改了什么 + **为什么改**（收益与代价） | `lineage.changed_from_predecessor` 的 `text` / `benefit` |
| 规格对照 | 各版本关键规格 | `arch` / `context` |
| 精度对照 | 量化这条独立的线 + **逐子系统的实测精度** | `precision`，子系统行来自 `fetch_dtypes.py` |

三条规则，都是被实际错误逼出来的：

- **和 `lineage.predecessor` 比，不和时间上的前一个比。** 支线（V4-Pro、日期版）的前代不是时间前一个，按时间比会把支线和主线串成一条假链。
- **getter 里不做回退推断。**「config 没单独声明专家精度」和「专家精度等于主干」是两件事：前者是事实，后者是推论。回退会把推论当成记录值放进矩阵，造出记录里根本没有的「变化」。
- **缺值不等于取消。** 实测字段只在相邻两版都测过时才比（`diff: 'both-measured'`），否则矩阵会把「这一版还没测」显示成「这一版取消了」。自由文本字段干脆不比（`diff: false`）—— 措辞差异不是架构变化。

精度表只留 **dtype 看不出来的** config 声明项。整体量化方法、权重元素格式、单独声明的专家精度三项都被逐子系统实测覆盖了，留着是同一件事说两遍；其中权重元素格式还会在 V4.1 显示成「e4m3 → 取消」（config 不再声明这个键），读起来像放弃了 E4M3。`scale 格式` 留着，因为它和实测**不一致**，而那个不一致正是要说的事。

实测行同样进变化矩阵，于是「量化精度」这一行读起来是这样的：

| 版本 | 矩阵里显示的 |
|---|---|
| V3 | 量化块 无 → [128,128]；路由专家 BF16 → FP8_E4M3 / FP32；注意力、共享专家、稠密 FFN 同 |
| V3.1 | **只有** scale 格式（config 声明）无 → ue8m0 —— 实测零变化，声明与存储的落差就摆在这 |
| V3.2-Exp | 归一化 BF16 → FP32 |
| V4-Flash | 路由专家 FP8_E4M3 / FP32 → **FP4 / FP8_E8M0** —— 权重降精度和 scale 换存储在同一格里 |
| V4.1-Flash | **只有** 量化块 [128,128] → [32,32] —— 专家精度没有再变 |

两个要点：

- **每个手写数字旁边都有证据徽标**（官 / 推 / 判 / ?），前缀继承规则与 `validate.py` 共用一套实现。两边不一致会让校验通过的记录在页面上显示错的口径。
- **内部 overlay**：设 `OVERLAY_DIR` 指向私有数据目录即可构建内部站，记录按 id 深合并、数组追加。不设就是公开构建，页头会标明当前是哪一种。

`.github/workflows/site.yml` 先跑校验（含防泄漏）再构建部署到 GitHub Pages —— 数据不合规时站点不会被构建出来。

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
| 注意力显存 | 890 B/token + 2.71 MiB/序列（理论下限） | 94208 B/token + 554 MiB/序列 | 65536 B/token + 145.5 MiB/序列 |
| FFN | MoE 384×top-6 | MoE 512×top-10（堆叠） | dense |

三份记录都通过 schema 校验、证据覆盖检查和交叉核对，`derive.py`、`validate.py`、`mk_ops_table.py` 均返回 0。P0 的 schema 部分到此收口。

> 这一栏被改过两次，两次都是 indexer 缓存算大了。
>
> 第一次：K 产出层按 8 层算（实际 4 层），70144 → 37376（R1 / R2）。
> 第二次：**每层的宽度**按 `index_n_heads × index_head_dim` = 32×128 = 4096 算，
> 而 index K 在各 index 头之间共享 —— 产出它的 `indexer.wk` 是 `[128, 512]`，输出维只有 128。
> 37376 → **5632**（R27），1M 上下文 36.5 GiB → 5.5 GiB。
>
> 第三次：`attn.wkv` 输出恒为 512，而全量清单里 K 侧**没有任何 rope 投影** ——
> `qk_rope_head_dim` 的 64 维在 latent 之内，每层是 512 不是 576。
> 5632 → **5120**（R22 由此关闭），1M 正好 **5.0 GiB**。
>
> 三次都是同一个乘式的不同因子：哪些层、每层多宽、每维算几次。
> 查了两轮才把三个因子都验过 —— **一个乘式只验一个因子，剩下的因子照样能错一个量级。**
>
> 第四次又是两个因子（R30 / R21）：**一个条目覆盖几个 token** —— 压缩器是 r:1 时序池化，
> 比值 2 的层每 token 只摊半个条目，元素数 2048+512 → **1280+320**；**每元素几字节** ——
> 参考实现的落盘格式是 FP4 + scale（全局 KV 块 16 配 E4M3、index K 块 32 配 E8M0），不是 BF16。
> 合起来 **890 B/token**，但它和 5120 不是同一种数：5120 是错假设下的上界，890 是精确计数的
> **理论下限**，不含分页与环页粒度，两者之间不算倍数。滑窗也从 40 层改成 43 层（MTP 块同样有窗口）。
> 精度取自读实现（二手），证据标 judgement；记录里元素数和精度分两处存，可信度不同的东西不乘进一个字段。

### DeepSeek 谱系（13 个历史版本）

把整条线补齐时多出一个取数模式：**只取张量名**。

`fetch_hf.py --names-only` 从 `model.safetensors.index.json` 的 `weight_map` 一次请求拿到全部张量名，不含形状与 dtype。代价是算不了参数量，收益是「某个张量存不存在」这类结构问题能批量问 —— 而跨版本比对问的几乎全是这类问题。十几个万亿模型逐个读分片头是上千次请求，读 index 是 13 次。

名字清单单独存 `facts/<id>/names.json`，**不混进 `shapes.json`**：形状为空的清单会让 `derive.py` 把参数量静默算成 0，而不是报「清单不完整」。

这个模式当场纠正了脚手架的一处误判，也验证了三件 config 里看不出来的事：

- **V4 起没有 KV 上投影**。脚手架按「有 `kv_lora_rank` 就是 MLA」判型，而 V4 的 config 根本不声明 `kv_lora_rank`，它改用 `head_dim: 512` + `num_key_value_heads: 1`。权重侧 `kv_b_proj` / `k_b_proj` / `v_b_proj` 在整个 V4 线零命中 —— 所以 **MLA → MQA-on-latent 的转折发生在 V4，不是 V4.1**。沿用 MLA 的注意力算子会直接对不上权重。
- **`num_hash_layers: 3` 的权重形态是 `ffn.gate.tid2eid`**。config 只给一个数字，权重里正好 3 个 token id → expert id 的查表张量。它和打分路由是两条路径，融合 router 要能走旁路。V4.1 把它去掉了。
- **哪些版本架构零差异**。R1 与 V3 的 config 只差一个 `transformers_version`，张量名逐个相同；V3.2 与 V3.2-Exp 的 config 逐字段相同；V4-Flash-0731 与 DSpark 版同理。**影响最大的那一代（R1），算子侧什么都没变。** 跟踪时先比张量名集合，相同就不必重走算子评估。

13 份历史记录都没取形状，所以**都不报参数量** —— `params.counting_note` 直接写明原因与补取方式。有 dtype 才有可信总量，FP4 打包的专家权重尤其如此。

#### 把圆点换成「旧值 → 新值」当场查出两处错

变化矩阵一开始只画 ● 表示「这个维度动过」。改成显示具体数值变化之后，立刻暴露两处脚手架的字段错误 —— 圆点把它们完全盖住了：

1. **`fmt` 被当成了 `scale_fmt`。** config 里 `fmt: e4m3` 是量化权重的**元素格式**，`scale_fmt: ue8m0` 是 **scale 张量的格式**，两个不同的东西。脚手架写的是 `scale_fmt or fmt` 兜底，于是 V3（没声明 `scale_fmt`）被记成 `scale_format: e4m3`，矩阵里就冒出一次根本不存在的「e4m3 → ue8m0」迁移。实际是 V3 **没有** scale 格式声明，V3.1 才加。schema 拆成 `weight_format` 与 `scale_format` 两个字段。
2. **config 值冒充实测值。** 脚手架把 config 的 `expert_dtype` 回填进了 `per_subsystem`，而 `per_subsystem` 是留给权重清单实测结果的。于是 6 份只读过 config 的记录，在标着「实测」的那一行显示出值来。两者经常对不上 —— V4.1 的 config 说量化块是 `[32,32]`，实测路由专家却是沿 K 的 `1×32`。

顺带还补上了 V4 线六份记录缺的 `attn_sink`：它只存在于权重里，config 没有这个开关，而矩阵在「attention sink 有 → 取消」这种不可能的变化上把缺口显了出来。

**圆点能通过校验，因为它不声称任何具体值。** 一旦要求写出从什么变成什么，错误就必须现形。

#### 按子系统测精度：不读全量清单也能回答

「这个模型的专家是什么精度、注意力是什么精度」不需要全部张量的形状 —— 每个子系统拿一个代表张量就够。`fetch_dtypes.py` 因此只读**覆盖代表张量的最小分片集**：

```bash
.venv/bin/python scripts/fetch_dtypes.py --id ds-v3 --repo deepseek-ai/DeepSeek-V3
#   ds-v3: 91991 个张量 / 9 个子系统，读 5 个分片（全量是 163 个）
.venv/bin/python scripts/apply_dtypes.py      # 写进记录；已有值时只比对不覆盖
```

归类用 `scripts/lib/subsystems.py`，**和 `derive.py` 共用同一份** —— 两边用不同的归类，会让「参数量里的 expert」和「精度表里的 expert」指的不是同一批张量。展示顺序与中文名在 `data/schema/subsystems.json`，Python 和站点读同一个文件。

挑代表张量的办法换过两次，两次都是被错误结果逼的：

1. **按名字长度挑 → 选中了 `.scale`。** `layers.0.ffn.experts.0.w1.scale` 比 `.weight` 短，于是整条 V4 线的专家精度被报成 `FP8_E8M0` —— 报的是 scale 的精度。改成权重与 scale 分开挑，各报各的（专家是 FP4 权重 + FP8_E8M0 scale，只报一个会让反量化实现猜错）。
2. **仍按名字长度挑 → 选中了 `attn_sink`。** 它是 `[64]` 的可学习向量，FP32，比 `attn.wkv.weight` 短。于是「注意力精度」报成 FP32。改成**读完分片之后按张量大小挑** —— 子系统里最大的那个才是带宽与存储的主体，也才是精度表该报的那个。

同时修了归类顺序的两处：`attn_norm` / `ffn_norm` 要先归 norm（带 norm 的就是归一化，没有例外），`hc_attn_*` / `hc_ffn_*` 要先归 residual（mHC 系数不是注意力也不是 FFN，而且是 FP32，混进去会把那两项的精度带偏）。修完 V4-Flash 的 `dense_ffn` 这一类就正确地消失了 —— 它 43 层全是 MoE，本来就没有稠密 FFN。

挑代表张量的办法最后定成**按元素数加权的多数**：带宽主体说了算，又不会被少数异类翻盘。占比低于 60% 记 `MIXED`，各 dtype 的占比留在 `facts/` 里。

同一次还量了**实测量化块**（权重形状 ÷ 同名 scale 形状，FP4 先还原打包）。这个必须实测，不能抄 config —— 见下面 V4 那条。

**「4 bit」是实测，「FP4 而不是 INT4」是声明。** 容器末维正好是逻辑维的一半（`w1 [2304,2560]` vs 同层未打包的 `shared_experts.w1 [2304,5120]`），这证明每值 4 bit。但形状和 dtype 都区分不了 4 bit 浮点与 4 bit 整型——`FP4` 这个具体格式只有 `config.expert_dtype` 一个来源。两件事要分开说。

结果写 `facts/<id>/subsystem_dtypes.json`，**不进 `shapes.json`**：抽样只能回答精度，拿去算参数量会静默少算。`apply_dtypes.py` 对已有 `per_subsystem` 的记录默认只比对不覆盖 —— 抽样和全量清单不一致时，该是人去看为什么，而不是让抽样把全量盖掉。

两道自检都过了：

- **抽样 vs 全量**：V4.1-Flash 的 `per_subsystem` 是早先从完整清单（48 分片 96085 张量）逐项核出来的。抽样只读 14 个分片，**11 个子系统的代表精度与 scale 精度全部吻合**。（只是代表值吻合，不是分布完全相同：`attention` 的 F8_E4M3 占比抽样 0.993 vs 全量 0.996。）
- **同架构版本之间**：V3 vs R1、V3.2 vs V3.2-Exp、V4-Flash-DSpark vs 0731、V4.1-Flash vs 1017 —— 四对「张量名完全一致」的版本，逐子系统的权重精度与 scale 精度全部一致。有一对一开始对不上，查出来是其中一个的抽样跑挂了（`head -1` 把 traceback 吃掉了），留着旧结果。

#### config 的 `weight_block_size` 对 FP4 专家是错的

量化块必须从「权重形状 ÷ 同名 scale 形状」实测，不能抄 config：

```
V4-Flash  experts.0.w1  [2048,2048] I8 ÷ scale [2048,128]
          还原 FP4 打包 → 逻辑 [2048,4096]
          4096/128 = 32，2048/2048 = 1       →  block [1,32]
          但 config 声明的是                  →  [128,128]

V4-Flash  shared_experts.w1 [2048,4096] F8 ÷ scale [16,32]  →  [128,128] ✓
```

按 `[out, in]` 的轴序，**FP4 路由专家是 `[1, 32]`**（逐输出行、沿 K 每 32 个一个 scale，MX 风格的微缩放），**FP8 子系统才是 config 声明的那个值**。照 `[128,128]` 去反量化专家，会期望 scale 行数是权重行数的 1/128，实际是 1:1 —— 差 128 倍，权重被静默解错。

这推翻了本仓自己写过的一句话。V4.1 的记录里原本写着「路由专家精度没变，这一代只收紧了块」—— 那句话本是为了纠正「以为专家精度又降了」，却在同一句里引入新错：**专家的块也没收紧**，V4 一代就已经是 `[1,32]`，收紧只发生在 FP8 子系统（128→32）。留痕见 V4.1 记录的 R23。

同时量出 V3 一代的一个边界：`kv_a_proj_with_mqa` 输出维 `kv_lora_rank + qk_rope_head_dim = 576`，而 `576 / 128 = 4.5` 不整除，scale 实际有 5 条 —— 末块不满。工具把这种情况记作 `128ceil`。按「block 一律整除」写的反量化实现会在这个张量上错位。

根治办法不是改注释，是把实测块按子系统存进 `per_subsystem_block`，和 dtype 一样处理。改完之后变化矩阵自己把结论证了出来 —— V4.1 那一列里**路由专家根本不出现**，因为它没变：

```
V4.1-Flash  量化块（config 声明）   [128,128] → [32,32]
            共享专家（实测）  … / [128,128] → … / [32,32]
            注意力（实测）    … / [128,128] → … / [32,32]
            路由专家                        不在列（[1,32] → [1,32]）
```

#### 两个只有归类对了才看得见的 bug

这一轮还查出两处工具缺陷，都不是数据错而是**归类**错：

**`.bias` 和量化 scale 被归进了同一个桶。** `_AUX` 正则写成 `\.(scale|…|bias|_amax)$`，于是 `ffn.gate.bias`（F32 路由偏置）被当成 router 的 scale 精度报了出来 —— 而 router 权重是 BF16、未量化，根本没有 scale 张量。记录本身侥幸没被污染（`apply_dtypes.py` 只给量化过的子系统写 scale），但那是运气不是设计。已拆成只匹配 scale 的 `_SCALE`。顺带删掉 `_amax` 这条死代码：真实名字是 `*.weight_amax`，`\._amax$` 匹配不上，它看起来在防守，其实什么都没做。

**V3 一代的 MTP 整块算进了主干。** `subsystems.py` 的注释声称 `eh_proj|enorm|hnorm|shared_head` 已经解决了「V3 的 MTP 借用层索引」这件事 —— 实际只捕到这 5 个胶水张量。实测第 61 层共 1564 个张量，只有 5 个归到 draft，**其中 1536 个专家算在了主干账上**（`num_hidden_layers=61`，主干是 0–60）。`derive.py` 按同一套归类分参数量，主干会凭空多出一层 256 个专家。层索引是唯一可靠的判据，但它只能由调用方给，所以 `classify()` 加了 `mtp_from` 参数。修完 V3 一代的 `draft` 精度从 BF16（5 个胶水张量）变成 FP8_E4M3（整层的真实构成）。

**一条注释声称某个坑已经填了，是这轮代价最高的一处。** 它让人不会再去看。

#### 第三轮：注意力与 MoE 专项，又查出 5.2 倍

拉到全量清单后，两路代理各盯一个维度。**最重的一条推翻的是我刚提交的数。**

V4 一代的 KV 推导高估 **5.2 倍**，三个独立错因叠加：

```
V4-Flash   compressor 存在 ⟺ compress_ratios[l] != 0     （41/43 层，L0/L1 没有）
           indexer    存在 ⟺ compress_ratios[l] == 4     （21/43 层，不是 43）
           compressor.ape 第一维 == 该层压缩比
             L2 ratio=4   → ape [4,1024]  wkv [1024,4096] → 每 token 256 元素
             L3 ratio=128 → ape [128,512] wkv [512,4096]  → 每 token   4 元素
```

三条规律在权重里零例外。而推导：(1) indexer 按总层数算；(2) indexer 宽度用了条目宽 256，没除比值（`indexer.compressor.ape [4,256]`，4 个 token 共用一个条目，每 token 实为 64）；(3) 把 `attn.wkv` 的 512 维当成全局 KV 逐 token 计费，而它是 `sliding_window=128` 的**局部** KV，每序列固定。

| | 原值 B/token | 实测 | |
|---|---|---|---|
| V4-Flash / DSpark / Vision-Exp | 71552 | **13600** | +5.4 MiB/序列 |
| V4-Pro | 101504 | **19448** | +7.6 MiB/序列 |
| V4.1-Flash | 5632 | **5120**（后经 R30 / R21 改为 890，理论下限） | +5.0 MiB/序列（后改 2.71 MiB） |

「存在逐层压缩比、本公式未建模」那句免责声明**盖不住 (1) 和 (3)** —— 层数用错和把窗口 KV 当全局 KV，跟压缩比无关。**免责声明只能免掉它点名的那件事。**

V4.1 的 5632→5120 顺带关掉了 R22：`attn.wkv` 输出恒 512，而全量清单里 K 侧**没有任何 rope 投影**，所以 `qk_rope_head_dim` 的 64 维在 latent 之内。1M 正好 5.0 GiB。

新增 `kv_sharing.measured_per_token`：逐层量出来的元素数，`derive.py` 一旦看到它就接管公式，status 标 `measured`。公式表达不了的结构，就不要让公式去猜。

#### MoE 侧：四处真错，都是「config 与权重不符时选错了边」

- **`mtp_layers: 1`**（DSpark / 0731）照抄 config 的 `num_nextn_predict_layers`，而权重里有 **3 个完整 MTP 块**，`compress_ratios` 也是 46 = 43+3。同一份记录的 `kv_sharing.note` 自己写着「项数 = 主干层数 + MTP 层数」—— **文件内部自相矛盾**。
- **「MTP 层 1→3，草稿侧权重乘 3」**（Vision-Exp）：3 块在 **DSpark** 那一代就有了，本版草稿侧只多了 3 个 `bias_vl` 张量（4705→4708）。真实的 1→3 发生在 DSpark，而 DSpark 的记录一个字没提。
- **「草稿头不再是『再跑一层 transformer』」**：`mtp.0` 带完整 `attn.*` + 256 专家 MoE，就是个 transformer 块；`markov_head`/`confidence_head` 只在 `mtp.2` 上，是附加头不是替换。草稿成本**涨了**三倍。
- **「hash 层完全不算 router，省掉打分与 topk」**：L0/L1/L2 各有一张 `gate.tid2eid [129280, 6]`（词表 × top_k），**但也有和其余层一样的完整 `gate.weight [256, 4096]`**，缺的只是 `gate.bias`。查表与打分各自承担什么，形状定不了。

另外三处说过头的：V2 的「分组按组数收缩 all-to-all」——分组约束的是**目的地设备数上界**（6→3），载荷 `2·top_k·hidden·bytes` 与分组无关；V3/V4.1 的「专家更多更细」——**单专家反而变大了**（23.6M→44.0M、25.2M→35.4M），只有「中间维/hidden」这个比值略细；V3 的「激活增幅可控」——每层激活 FFN 参数涨 2.10 倍，稀疏度只从 3.75% 改善到 3.12%。

补上了 14/14 全缺的 `storage_note`（全系都是逐专家独立张量，grouped GEMM 因此吃 2 维列表而非 3 维张量）和 13/14 缺的 `balance`。

#### 补完 missing 项时又多出两条

**`head_dim: 192` 只管 QK。** V2–V3.2 的 config 另有 `v_head_dim: 128`，14 份记录一份都没记。实害可验证：`o_proj` 实测 `[hidden, 16384]`，16384 = 128 头 × **128**，而按单一 `head_dim 192` 去推会得到 24576 —— **高 50%**。新增 `v_head_dim` 与 `head_dim_note` 两个字段。

**三类覆盖面不同，不能混着说「逐层」。** V4 一代的权重里：

```
compressor 在 41/43 层（Flash，L0/L1 没有）   存在 ⟺ compress_ratios[l] != 0
indexer    在 21/43 层                        存在 ⟺ compress_ratios[l] == 4
attn.wkv   在 43/43 层（滑窗局部 KV）
```

V4.1 更细：`compressor.wkv`/`norm` 在 4 个源层都有，但 `compressor.wgate` **只在 3 层** —— 比值为 1 的那个源层不带门控，源层之间不同构。新增 `kv_sharing.layer_coverage` 专记这件事，因为「逐层」这个词正是上一轮 5.2 倍高估的源头。

顺带把 `derive.py` 里一段死代码修了：`src_kind` 赋值两次却从不读取，那句「未区分产出/重打分，可能高估」的警告**永远不会出现在任何输出里**，formula 字符串也恒印 `index_source_layers`。现在没有产出层声明时会明确标 `partial` 并说明按全部层计是上界。

还有一处 evidence 口径：`attn_sink` / `hash_layers` / `routing_bias` 这些**内容来自权重清单**的字段，在五份 V4 记录里继承了 `arch: official`，于是永远不会出现在 `--list-judgements` 里，绕过了 Gate C 的人工复核。`attn_sink` 尤其刺眼 —— 字段正文自己写着「config 里没有这个开关」，口径却是 official。修完判断类字段从 112 条涨到 126 条。

#### 两件只有 dtype 看得见的事

**一、`scale_fmt: ue8m0` 不等于 scale 存成 8 bit。** 声明从 V3.1 开始，但 scale 张量实测**仍是 FP32**，连着三个发布都是；到 V4-Flash 才真的换成 8 bit 的 `F8_E8M0`：

| | V3 | V3.1 | V3.2-Exp | V3.2 | V4-Flash | V4.1-Flash |
|---|---|---|---|---|---|---|
| config `scale_fmt` | — | ue8m0 | ue8m0 | ue8m0 | ue8m0 | ue8m0 |
| scale 张量实测 dtype | FP32 | **FP32** | **FP32** | **FP32** | FP8_E8M0 | FP8_E8M0 |

按「声明 ue8m0 就是 8 bit 存储」去算 scale 张量大小，这三个发布会算小 4 倍。

> 「ue8m0 在这几代约束的是取值（2 的幂次）而非容器」是**推断**，不是实测：`facts/` 里只有形状和 dtype，没有张量数值。要证实得去读权重数值。

**二、V3.2-Exp 把 layernorm 权重从 BF16 换成了 FP32。** config 没有这个信息，张量名也和 V3.1 逐个相同 —— 只有 dtype 看得出来。影响是 RMSNorm 变成混合精度算子：FP32 权重配 BF16 激活，按「norm 权重与激活同 dtype」写的核会直接读错。

> 抽样只读了 163 个分片里的 10 个，覆盖第 0–8 层。那几层里 `input_layernorm` / `post_attention_layernorm` **占比 100%** 是 F32，但**占比不是覆盖率** —— 第 9–60 层没抽到，所以不能说「全部层」。
