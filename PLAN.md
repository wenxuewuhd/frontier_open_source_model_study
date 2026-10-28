# 前沿开源大模型演进跟踪 · 项目方案

> 状态：方案已定稿，尚未开工。语言约定：中文为主，术语（MLA、MoE、TPOT、EP 等）保留英文。

## 1. 定位

参考站 [model-research-atlas](https://limjiunnbin.github.io/model-research-atlas/) 讲的是**模型本身**：家族 → 版本 → 逐张量审计。本项目取 **Infra 视角**，每一次模型变化都回答三个问题：

1. **改了什么**：架构、精度、模态、推理范式。
2. **对负载意味着什么**：计算、显存、KV、通信、算子、推理引擎。
3. **下一代 infra 要准备什么**：落成一条可执行的「Infra 信号」。

硬件重心是**昇腾为主、NVIDIA 对照**：平台矩阵、瓶颈分析、算子需求都先站在昇腾侧写，NVIDIA 作为参照系。

从参考站借鉴两点：
- **证据口径**：每个字段标注 `official`（官方披露）/ `derived`（计算推导）/ `judgement`（解释判断）/ `unknown`（未知）。
- **参数口径分开**：逻辑参数、激活参数、权重存储、KV 缓存分开统计，各自写明口径。

## 2. 覆盖范围

历史回溯从 **2024 年**起（DeepSeek-V2、Qwen2、GLM-4 这一代：MLA、细粒度 MoE 的起点）。

| 层 | 内容 | 跟踪深度 |
|---|---|---|
| **主流家族** | DeepSeek、GLM（智谱）、MiniMax、Kimi（月之暗面）、Qwen、阶跃星辰 | 全量：演进、架构、精度、平台、性能、瓶颈、算子需求 |
| **多模态 · 理解** | 各家 VL / Omni 线，以及重要的独立多模态模型 | 视觉/音频编码器、模态融合方式、prefill 负载、encoder 并行与分离部署 |
| **多模态 · 生成** | 图像/视频生成（DiT 类，含各家族及主流开源生成模型） | 独立 schema（见 §3.2）：步数、分辨率/帧数、VAE、文本编码器、单步时延 |
| **新型范式** | Loop/递归深度 Transformer、dLLM（LLaDA、Dream、SDAR 等）、线性/混合注意力、稀疏注意力（NSA/DSA/MoBA 等）、MTP/投机解码 | 观察名单 + 成熟度分级 |

新型范式的**成熟度分级**：`L0 论文` → `L1 开源权重` → `L2 主流引擎支持` → `L3 昇腾可用` → `L4 生产部署`。分级直接回答「这件事什么时候需要 infra 真正投入」。

## 3. 数据 schema

### 3.1 语言模型 / 理解类多模态（`kind: llm | vlm | omni`）

1. **基本信息**：发布日期、license、总参数/激活参数、上下文长度、权重链接、所属主线/支线。
2. **架构**：
   - 注意力：MHA / GQA / MLA / 线性 / 稀疏 / 混合，以及混合比例与层分布
   - MoE：专家数、top-k、共享专家、路由与负载均衡方式
   - 层数、hidden size、norm、位置编码、MTP
   - 多模态：编码器结构与参数量、projector、token 压缩方式
3. **精度**：训练精度、发布权重精度、官方量化（FP8 block-wise、INT4 QAT、MXFP4/NVFP4、W8A8 等）、KV cache 精度；并按硬件代际记录**原生支持**的数据类型。
4. **推导负载特征**（`scripts/derive.py` 从 `config.json` 自动计算，口径标 `derived`）：
   - 每 token FLOPs（线性部分 / 注意力部分分开）
   - 权重存储（按精度）、每 token 的 KV 缓存
   - prefill / decode 的算术强度
   - EP 部署下的 all-to-all 通信量
5. **平台支持矩阵**：
   - 引擎：vllm-ascend、MindIE、SGLang（Ascend 后端）为主；vLLM、SGLang、TensorRT-LLM 作为 NVIDIA 侧对照；另记录 LMDeploy、llama.cpp、KTransformers 的支持情况
   - 硬件：昇腾 910 系列 / 950 系列为主；H100/H200、B200/GB200 作对照
   - 记录每个组合的状态：day-0 / 已支持 / 进行中 / 不支持，以及支持到什么精度
6. **性能**：吞吐、TTFT、TPOT，每条都带来源类型（`official` / `community` / `self`）、硬件、引擎版本、并行配置、输入输出长度。
7. **瓶颈分析**（昇腾视角）：HBM 容量与带宽、Cube/Vector 负载配比、HCCS / 跨机通信、缺失或需融合的算子、长上下文下的 KV 压力。
8. **算子需求**：链接到 NPU 算子表产物（decode 算子表 xlsx、架构图、融合建议），与已有模型做规格差分，标出新增算子。
9. **Infra 信号**：1–3 条结论，每条带口径标签。

示例：

```yaml
id: deepseek-v3
family: deepseek
kind: llm
released: 2024-12-26
params: { total: 671e9, active: 37e9, evidence: official }
arch:
  layers: 61
  hidden: 7168
  attention: { type: MLA, kv_lora_rank: 512, qk_rope_head_dim: 64 }
  moe: { routed_experts: 256, shared_experts: 1, top_k: 8 }
  mtp: 1
precision:
  training: FP8
  weights_release: [FP8-blockwise]
platforms: []        # 引擎 × 硬件 × 精度 × 状态
performance: []      # 每条带 source.type
infra_signals:
  - text: "MLA 把每 token 的 KV 压到 latent 维度，decode 瓶颈从 KV 容量转向权重带宽与 EP 通信"
    evidence: judgement
```

### 3.2 生成类多模态（`kind: image-gen | video-gen`）

和 LLM 的负载差别很大，单独建 schema：
- 主干结构（DiT / MMDiT 等）、参数量、文本编码器、VAE
- 默认采样步数、支持的分辨率/帧数/时长、蒸馏或步数压缩方案
- 负载：单步 FLOPs、单步时延、生成一张图或一段视频的端到端耗时、显存峰值
- 平台：Diffusers、xDiT 等框架的支持情况，以及在昇腾上的适配状态

dLLM 同样是扩散范式，它的步数、并行解码、KV 复用这些问题和生成类可以放在一起对照。

## 4. 网站结构

- **总览**：最近两周的时间线 + 「Infra 信号」汇总。
- **技术专题**（跨家族，这是和参考站最大的区别）：
  - 注意力演进：MHA → GQA → MLA → 稀疏 / 线性 / 混合
  - MoE 细粒度化与 EP 通信
  - 精度演进：BF16 → FP8 → FP4，以及昇腾侧的对应路线
  - 多模态融合与 encoder 分离部署
  - 生成模型负载
  - 新范式观察（含成熟度看板）
- **家族页**：演进时间线 + 版本差异对比。
- **模型详情页**：§3 的全部维度。
- **平台矩阵页**：模型 × 引擎 × 硬件 × 精度，可筛选。
- **双周简报**：每期一篇，归档在 `reports/biweekly/`。

## 5. 仓库与数据隔离

采用**公开站 + 私有数据仓**：

- **本仓库（公开）**：只放公开来源的数据与分析，构建出公开网站，部署到 GitHub Pages。
- **私有仓库**（例如 `frontier_open_source_model_study_private`）：目录结构与 schema 和本仓库一致，放自测性能、内部瓶颈分析、内部算子适配状态。
- **内部构建**：`scripts/build --overlay <私有仓 data 路径>`，把私有数据叠加到公开数据上，构建内部站点。
- **防泄漏**：公开仓库的 CI 拒绝任何 `source.type: self` 或 `visibility: internal` 的记录。

```
data/                       # 唯一事实源，YAML，JSON Schema 校验
  models/<family>/<model>.yaml
  platforms/                # 引擎、硬件定义
  paradigms/                # 新型范式条目与成熟度
  schema/                   # JSON Schema
analysis/topics/*.md        # 专题分析，引用 data/ 里的数据
reports/biweekly/YYYY-MM-DD.md
scripts/
  fetch_hf.py               # 拉 HF config / 模型卡
  derive.py                 # 计算推导指标
  watch.py                  # 监控新发布，生成候选变更清单
  validate.py               # schema 校验 + 防泄漏检查
site/                       # Astro（content collections 自带 schema 校验）
.github/workflows/          # validate、build、deploy、定时采集
```

原则：**数据和叙述分开**。数字只出现在 YAML 里，可以校验、可以复算；Markdown 只引用数据，不复制数字。

## 6. 双周刷新流程

1. **自动采集**（GitHub Actions 定时运行）：
   - HF org：`deepseek-ai`、`zai-org`、`MiniMaxAI`、`moonshotai`、`Qwen`、`stepfun-ai`，以及多模态和新范式的关注列表
   - 引擎仓库：vllm-ascend、MindIE 公开发布、SGLang、vLLM、TensorRT-LLM 的 release 与关键 PR
   - arXiv 关键词：loop transformer、diffusion language model、linear attention、sparse attention 等
   - 产出一份「候选变更清单」
2. **Claude 起草**（定时任务）：读候选清单 → 更新 YAML → 跑 `derive.py` → 起草双周简报 → 提 PR。
3. **人工审核**：核对口径标签和来源后合并，公开站自动发布。私有仓走同样的流程，单独审核。

## 7. 分阶段

| 阶段 | 内容 |
|---|---|
| **P0** | schema（LLM + 生成两套）、`fetch_hf` / `derive` / `validate` 脚本、6 个家族的最新旗舰、专题「注意力演进」、网站骨架、公开站部署 |
| **P1** | 2024 年起的历史版本回溯、精度演进专题、平台矩阵（昇腾为主） |
| **P2** | 公开性能数据收集与瓶颈分析、多模态（理解 + 生成）、新范式观察名单、双周自动化上线、私有仓与 overlay 构建 |
| **P3** | 昇腾自测基准（统一配置，结果进私有仓）、算子需求与 NPU 算子表打通 |

## 8. 待定事项

- 内部站点部署在哪里（内网静态托管 / 私有 Pages）。
- 双周固定在哪一天采集和发布。
- 昇腾具体关注哪些代际与机型组合。
- PR 审核人。
