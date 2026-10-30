# 视频生成线 · 第 1 步补管线实施清单

> 这是 [video-gen-gate-a.md](video-gen-gate-a.md) §5.5 的展开，供第 1 步执行。
> Gate A 已定：12 份记录、8 个 family，录入先只做视频。本清单覆盖 **schema + derive.py +
> subsystems.py** 三件管线改造，以及第一批三份记录的排期（§E、§F）。
> 编制日期 2026-10-30，复核更新一次（见 §0）。基线 `main` = 37e9404。
>
> **进度（2026-10-30）**：§E 第一段已全部落地 —— B1、B2、B3、B4、A3a、C1、C2、C3 都做完了，
> LLM 侧 15 份记录的 `derive.py` 摘要与 JSON 逐字不变、`validate.py` 0 错 0 警。
> 剩下卡在 P1 整合之后的是 **A3b**（`dit` 桶内正则扩展）与依赖它的 `fetch_dtypes.py`。
> 两条实现中发现的事写在 §H。

## 0. 先决条件：P1 的共享脚本改动

Gate A 交接时记的是「三处待整合」。这一节两次复核过，结论改过一轮，**现状如下表**。

| 待整合项 | 来自哪条线 | 现在在哪 | 状态（2026-10-30 复核二） |
|---|---|---|---|
| `derive.py` 的 `bytes_basis` / 厂商公布 KV 字节（含 schema 的 `kv_sharing.declared_per_token_bytes` 与 `precision.kv_cache_components.source` 描述改写） | MiniMax 线提出，在 DeepSeek 记录上落地 | **`main`，commit `7e65c70`** | **已在 main**。原先在 `p1-glm` 上叫 `bad7cf6`，已 cherry-pick 到 main；`p1-glm` 退回它自己的 9 个提交（`0793fd1`），搬进独立 worktree `wt-p1-glm` |
| `scripts/lib/subsystems.py` 要补语言模型侧缺的子系统划分 | GLM 线（报「必改」） | **还只是待办，不是代码** | 未整合 |
| `scripts/lib/dtypes.py` 不认 INT4-in-INT32 打包 | Kimi 线 | **还只是待办，不是代码** | 未整合。`kimi-k2-thinking` / `kimi-k2.5` 各有 2 处 Gate B MISMATCH 卡在这上面没收口 |

本轮核实（可重跑）：

```bash
git merge-base --is-ancestor 7e65c70 main && echo "7e65c70 已在 main"
git show main:scripts/derive.py | grep -c vendor_declared              # 3
git diff --stat 352c114..main -- scripts/lib/                          # 空
```

### ⚠️ 「没有改动」不等于「依赖已清」

上一版本清单在这里下过一个错误结论。当时我查到
「`subsystems.py` / `dtypes.py` 未见任何分支或工作区有改动」，于是写成「依赖比预想的轻」。

**查到的事实是对的，推出的结论是错的。** 主会话明令禁止五条 P1 会话改共享文件，
要求它们把需求**写进交接笔记报回主会话**统一整合。所以那两处改动是以**待办**的形式存在的 ——
git 里查不到，正是因为流程要求它们不该出现在 git 里。拿 `git diff` 为空去证明「没有待整合项」，
是 model-audit 规则 7 的同一个坑：**空结果和「真的没有」在代码里长得一样。**

### 撞车点

| 本清单的节 | 和谁撞 | 怎么处理 |
|---|---|---|
| **A（`subsystems.py` 分两层 + 扩正则）** | **GLM 线待整合的 `subsystems.py` 改动。同一个文件，而且顺序即优先级** | **真撞车。必须等 P1 五条线全部跑完、共享改动整合进 main 之后再做。** 用户已定 P1 整合一次性做完 |
| C1（`derive.py` 按 `kind` 分流） | `7e65c70` 改过的 `kv_memory()` | **已解除** —— 那个改动已在 main。从 main 开分支就带着它，改法是在函数入口加 kind 判断提前返回，不动函数体 |
| B3（`precision` 加 `inference_default`） | 无 | **已解除** —— `precision.kv_cache_components.source` 的描述改写已在 main，是开分支时的现状，不是待合并的冲突 |
| 其余（B1 family 枚举、B2 `generation` 块、C2、C3、D） | 无 | 可以先做。`generation` 块是新建的子树，和任何 P1 改动都不重叠 |

**执行顺序**：从 `main` 直接开分支即可，不需要先合 `p1-glm`。
B1 / B2 / C 可以立刻动；**A 节排在 P1 整合之后**。

---

## A. `scripts/lib/subsystems.py` —— 归类分两层

### A1. 问题

现在是「张量名 → 子系统」一层映射。对生成类会**安静地错**，不是报错。

### A2. 实跑验证

用仓里现有的 `scripts/lib/subsystems.py` 跑真实张量名（取自 MAGI-2-preview、MiniMax-H3、
LTX-2.3 的权重清单），结果如下。**这不是推测，是实跑输出**：

| 张量名（真实） | 现在归到 | 应该归到 |
|---|---|---|
| `block.layers.5.mlp.moe_mlp.W_up` / `W_down` / `W_gate` | **dense_ffn** ❌ | expert |
| `block.layers.5.mlp.moe_mlp.gate` | router ✅ | router |
| `block.layers.5.mlp.moe_mlp.router.expert_bias` | router ✅ | router |
| `block.layers.5.mlp.shared_expert_fc1.weight` | shared_expert ✅ | shared_expert |
| `block.layers.5.mlp.modality_specific_shared_expert_fc1.weight` | shared_expert ✅ | shared_expert（可考虑细分） |
| `block.layers.5.mlp.split_linear.weight` / `merge_linear.weight` | **dense_ffn** ❌ | expert 的投影，单列 |
| `block.layers.5.attention.linear_qkv.weight` | **other** ❌ | attention |
| `block.layers.5.attention.linear_proj.weight` | **other** ❌ | attention |
| `block.layers.5.attention.linear_g.weight` | **other** ❌ | attention（输出门） |
| `block.layers.5.attention.sinks` | **other** ❌ | attention（sink） |
| `block.layers.5.mhc_alpha_post_attn` | **attention** ❌ | residual |
| `block.layers.5.mhc_phi_fused_attn` | **attention** ❌ | residual |
| `block.layers.5.mhc_bias_pre_mlp` | **dense_ffn** ❌ | residual |
| `block.layers.5.mhc_norm.weight` | norm ✅ | norm |
| `pre_adapter.video_embedder.weight` | **other** ❌ | patchify / 模态入口 |
| `pre_adapter.audio_embedder.weight` | audio（巧合命中） | patchify / 模态入口 |
| `post_adapter.final_linear_video.weight` | **other** ❌ | 模态出口 |
| `text_encoder.model.language_model.layers.3.mlp.down_proj.weight` | **dense_ffn** ❌ | text_encoder |
| `text_encoder.model.language_model.layers.3.self_attn.q_proj.weight` | **attention** ❌ | text_encoder |
| `vae.decoder.proj_in.weight` / `vae.encoder.conv_in.weight` | **other** ❌ | vae_decoder / vae_encoder |
| `audio_proj_in.weight`（H3 的 DiT 音频入口） | audio（巧合命中） | patchify |
| `model.diffusion_model.video_embeddings_connector.…ff.net.0.proj.weight` | **other** ❌ | connector |
| `model.diffusion_model.adaln_single.linear.weight` | **other** ❌ | adaln |
| `vocoder.conv_pre.weight` | **other** ❌ | vocoder |
| `audio_vae.encoder.…conv1.weight` | audio（巧合命中） | audio_vae |
| `text_embedding_projection.0.weight` | **other** ❌ | text_encoder 投影 |

复现脚本（放 `/tmp` 跑即可，不要进仓）：

```python
import sys; sys.path.insert(0, 'scripts')
from lib import subsystems as S
for n in [...]:        # 上表左列
    print(n, '->', S.classify(n))
```

三个最严重的：

1. **MAGI-2 的 36 层专家堆栈（约 108.7B 参数，占全模型 95%）整体落进 `dense_ffn`。**
   `expert` 的正则是 `\.experts?\.\d+\.|\bexperts\.|\.moe\.(gate|up|down)_proj\b`，
   `moe_mlp.W_up` 一条都不匹配；而 `dense_ffn` 的 `mlp` 命中了 `moe_mlp` 里的 `mlp`。
   后果：参数量拆分、带宽账、EP 切分全部错位，而且**不会报错**。
2. **mHC 系数落进 attention / dense_ffn。** 这正是 `subsystems.py` 注释里已经警告过的那个坑，
   只是 MAGI-2 写作 `mhc_*` 而现有正则写的是 `\bhc_[a-z]+_(base|fn|scale)\b|^hc_` —— 接不住。
   这些系数是 F32，混进 BF16 的注意力桶会把精度表带偏（规则 6：可信度不同的东西不要乘进同一个字段）。
3. **H3 的 33B 文本编码器与 DiT 自己的权重混成一桶。** `text_encoder.*` 的 attention / mlp
   会和 DiT 的同名子系统合并，一个外部文本编码器被算进 DiT 的参数量。

另有一类**不是错但是运气**：`pre_adapter.audio_embedder` / `audio_proj_in` / `audio_vae.*`
恰好被 `audio` 正则接住了。`audio` 在 LLM 侧的语义是「音频理解编码器」，在生成侧是
「音频 VAE / 音频分支」—— 同一个桶装两种东西，跨族系对照时会错。

### A3. 改法 —— 拆成 A3a / A3b 两半

A3 不是一整块。它是两件性质不同的事，**只有后一半和 P1 撞车**：

| | 改什么 | 碰不碰那张正则表 | 能不能现在做 |
|---|---|---|---|
| **A3a 组件层** | 新加 `component()` + 新文件 `data/schema/components.json` | **不碰** | **能** |
| **A3b 正则表扩展** | 往 `SUBSYSTEM` 里插新条目 | 碰，而且顺序敏感 | 等 P1 整合 |

这个拆分有实际后果：**组件级的账 A3a 一落地就能算**。
`minimax-h3`（要分开 DiT 33B / 文本编码器 33B / VAE 5.2B）和
`ltx-2.3`（要分开主干 18.56B / connector 2.0B / 文本投影 1.16B）靠的都是组件层。
只有 `magi-2-preview` 要的是 DiT **桶内**的专家 / mHC / 注意力拆分 —— 那才是 A3b。

---

#### A3a 组件层（不撞车，可先做）

**按顶层组件前缀分桶。** 在 `classify()` 之前加一个 `component()`：

```
dit          ← 无前缀 / block. / model.diffusion_model. / transformer.
text_encoder ← text_encoder. / prompt_enhancer. / text_embedding_projection.
vae          ← vae. / turbo_vae.
audio_vae    ← audio_vae. / stable-audio-open-1.0.
vocoder      ← vocoder.
upsampler    ← latent_upsampler. / temporal_latent_upsampler. / upsampler.
refiner      ← refiner.            （MAGI-2 的第二阶段）
connector    ← connectors. / *_embeddings_connector.
```

前缀表必须**数据驱动**（放 `data/schema/components.json`），不要硬编码 —— 候选之间命名差异极大，
硬编码等于每录一个新模型改一次代码。

桶内是否再细分由调用方决定：`dit` 桶内继续跑 `classify()`，其余桶一律整块算
（VAE 的内部结构对负载账没意义）。**A3a 单独落地时，`dit` 桶内先跑现有的 `classify()`**
—— 对 H3 和 LTX-2.3 够用，对 MAGI-2 会错（见 A2），所以 MAGI-2 要等 A3b。

**A3a 的验收**：A2 表里带组件前缀的那 9 行
（`text_encoder.*` 两行、`vae.*` 两行、`vocoder.*`、`audio_vae.*`、`connector`、
`adaln_single`、`text_embedding_projection`）归到正确的组件桶；
LLM 侧现有记录全部落进 `dit` 桶，`derive.py` 输出一字不变。

---

#### A3b 正则表扩展（撞车，等 P1 整合）

**在 `dit` 桶内把现有正则扩掉 A2 剩下的那些错。** 要加的条目（按优先级顺序插入，注意顺序即优先级）：

| 子系统 | 新增正则 | 必须排在谁之前 |
|---|---|---|
| `residual` | `^mhc_\|\.mhc_` | attention、dense_ffn、norm 之前（但 `mhc_norm` 要留给 norm，所以 norm 仍排在前面，或在 residual 正则里排除 `_norm$`） |
| `expert` | `moe_mlp\.W_(up\|down\|gate)\b\|moe_mlp\.experts?\b` | dense_ffn 之前（关键：要先于 `mlp`） |
| `attention` | `\battention\.\|linear_qkv\|linear_proj\|\bsinks?\b\|linear_g\b` | dense_ffn 之前 |
| `patchify`（新） | `patchify\|_embedder\b\|proj_in\b\|proj_out\b\|final_linear_` | 全部之前（否则 `audio_embedder` 被 audio 抢走） |
| `adaln`（新） | `adaln\|scale_shift_table\|time_embedder\|timestep` | attention、dense_ffn 之前 |
| `connector`（新） | `connector` | attention、dense_ffn 之前 |

**`data/schema/subsystems.json` 要加的 key**（展示顺序按带宽分量从大到小）：

```
expert（已有）, shared_expert（已有）, attention（已有）, dense_ffn（已有）, router（已有）,
adaln, connector, patchify, residual（已有）, norm（已有）,
text_encoder, vae_encoder, vae_decoder, audio_vae, vocoder, upsampler, other（已有）
```

`adaln` 必须单列：MiniMax-H3 的 33B 里约 13B 在这里，厂商说推理可预计算缓存。
不单列就永远验不了那句话。

**A3b 的验收**：A2 那张表的 26 行全部归到「应该归到」列，且 LLM 侧现有记录的
`derive.py` 输出一字不变（跑 `data/models/` 下全部记录对比）。

> **P1 依赖：A3b 是全清单唯一真撞车的一节，必须排在 P1 整合之后。A3a 不受影响。**
> GLM 线已经报了「`subsystems.py` 必改」—— 要补语言模型侧缺的子系统划分。那条改动目前
> 还只在交接笔记里、不在 git 里（主会话禁止 P1 会话直接改共享文件），所以 `git diff` 查不到它，
> **但它确实存在**。两边改的是同一张正则表，而这张表**顺序即优先级**，分别改完再合并几乎必然出错。
>
> 用户已定：P1 五条线全部跑完后一次性整合。本节等那次整合进 main 再动。
> 整合完要重跑一次 LLM 侧全量回归，确认 P1 的新正则没有被本节的分层包住；
> A2 那 26 个张量名可以直接当本节的验收用例（主会话已表示整合时会这么用）。

---

## B. `data/schema/model.schema.json`

### B1. `family` 枚举（阻塞一切）

现在是封闭枚举 `["deepseek","glm","minimax","kimi","qwen","stepfun","other"]`。
加：`wan, hunyuan-video, minimax-video, ltx, cogvideo, stepvideo, sana-video, magi`。

不加就一份记录都写不进去。**这是整条线的第一个阻塞点。**

> **P1 依赖**：无。已在 main 的 `7e65c70` 只碰 `kv_sharing` 子树，不碰 `family`。可以立刻做。

### B2. 重建 `generation` 块

现有的 9 个标量字段（`backbone` / `text_encoder` / `vae` / `vae_downsample` / `default_steps` /
`distilled_steps` / `resolutions` / `max_frames` / `max_duration_s`）全部替换。
`additionalProperties: false` 意味着不替换就一个新字段都加不进去。

新结构（字段依据见 Gate A §5.1 / §5.2，括号里是候选间的取值范围）：

```yaml
generation:
  backbone:
    stream_design:      # single | dual_stream | dual_then_single | in_sequence
    double_blocks:      # HunyuanVideo 20 / HV1.5 54 / 其余 0
    single_blocks:      # HunyuanVideo 40 / 其余 0
    layers: hidden: heads: head_dim: ffn:
    text_conditioning:  # cross_attn | in_sequence | dual_stream | adaln
    patch_size: [t, h, w]        # (1,2,2) / (1,1,1) / (2,2,2) / 1
  vae:
    spatial_compression:   # 8 / 16 / 32
    temporal_compression:  # 4 / 8
    latent_channels:       # 16 / 24 / 32 / 48 / 64 / 128
    causal: params: decoder_arch:    # conv3d | vit
    hf_repo: ref_model_id: ref_external:
    turbo_decoder:        # MAGI-2 独有：蒸馏出来的浅层解码器
  audio:                  # 仅音视频联合模型
    in_channels: vae: sample_rate_in: sample_rate_out: latent_hz: channels:
  text_encoders:          # 数组，多编码器并用是常态
    - role: name: hf_repo: params: hidden:
      shipped_with_weights: gated: release_dtype:
      hidden_layer_taken:   # H3 取第 50 层
      substitutable: note:
  sampling:
    steps:
    cfg: { scale:, distilled:, forwards_per_step: }   # 1 或 2，直接乘进总量
    step_distilled: { available:, steps: }            # 与 cfg.distilled 是两根独立的轴
    scheduler: shift:
  pipeline_stages:        # 端到端不是一段
    - { name:, model:, steps:, cfg:, resolution:, open_source: }
  buckets:                # 档位数组 + 默认标记
    - { width:, height:, frames:, fps:, duration_s:, default: }
  constraints: { width_divisor:, height_divisor:, frame_formula: }
  checkpoint_variants:    # 一次部署实际要驻留几套
    - { name:, role:, bytes:, resident: }
  expert_switching:       # Wan2.2 专用，和 arch.moe 互斥
    { axis: timestep, boundary:, experts: }
```

**三个最关键、缺了就算不出任何负载的必填**：
`vae.{spatial_compression, temporal_compression, latent_channels}`、
`backbone.patch_size`、`sampling.cfg.forwards_per_step`。

### B3. `arch` 侧的小改

| 改动 | 原因 |
|---|---|
| `arch.moe` 加 `routing_axis: {token, timestep}` | MAGI-2 是真 token 级路由（3072 专家 + aux-loss-free bias + EMA），Wan2.2 是时间步切换。两者共存一个字段会造假事实，分开成两个字段又会让 MAGI-2 填不进 `arch.moe` |
| `arch.moe.storage_note` | MAGI-2 的专家是堆叠 3 维张量 `[3072, 256, 1280]`，和 Step-5 同形态。字段已有，确认描述覆盖生成类 |
| `arch.residual` / `arch.attention.attn_sink` / `arch.attention.output_gate` | **保留，不要因为「生成类用不上」砍掉** —— MAGI-2 三样全有（权重实测） |
| 整个 KV cache 子树 | **保留** —— Gate A §2.4 的世界模型线是自回归的，KV 会回来 |
| `precision` 加 `inference_default` | Wan 全系与 HunyuanVideo-1.5 落盘 FP32、脚本跑 BF16 |
| `license` 考虑加 `restriction` | H3 要求美/欧/英/韩另行申请；HV1.5 的卡有 `extra_gated_eu_disallowed` |

> **P1 依赖：已解除。** `precision.kv_cache_components.source` 的描述改写已经在 main
> （`7e65c70`），所以从 main 开分支时它就在那儿了 —— 是现状，不是待合并的冲突。
> B3 往同一个 `precision` 对象里加 `inference_default`，在那份现状之上直接加即可。

### B4. `platforms[].engine` 取值

LLM 侧的引擎名（vllm-ascend / MindIE / SGLang / vLLM / TensorRT-LLM / LMDeploy / llama.cpp /
KTransformers）在生成类上基本用不上。要加：
`diffusers`、`ComfyUI`、`xDiT`、`vLLM-Omni`、`SGLang-diffusion`、`TensorRT-LLM-visualgen`、
`LightX2V`、`Wan2GP`、`MagiAttention`（MAGI-2 自研）。

昇腾侧尤其要注意：查到的适配走的是 **Ascend-SACT（AtomGit）与华为云 ModelArts 的独立配方，
不是 vllm-ascend**。`engine` 枚举要能表达这个入口差别，否则平台矩阵会把「昇腾能跑」
错挂到 vllm-ascend 名下。

---

## C. `scripts/derive.py` —— 按 `kind` 分流

### C1. 跳过三件 LLM 专属推导

对 `kind ∈ {image-gen, video-gen}`：`kv_memory()`、`active_params()`、`comm_per_token()`
**必须跳过并留显式标记**，不能返回 0。

依据是 model-audit 规则 7：空结果和「真的没有」在代码里长得一样。
现在 `derive.py` 完全不分 kind，直接跑会把「不适用」渲染成「0 字节 KV」，
站点上看起来就是「这个模型不占缓存」。

建议在输出里写 `{"status": "not_applicable", "why": "非自回归，推理过程中无跨步复用的 KV"}`，
并让 `site/src/lib/data.ts` 认这个状态、在页面上显示「不适用」而不是空格。

> **P1 依赖：已解除。** `kv_memory()` 的 `vendor_declared` 分支已经在 main（`7e65c70`，
> `grep -c vendor_declared` 得 3），从 main 开分支就带着它。
> 改法上本来也不冲突：本条是在函数入口加一道 kind 判断提前返回，不动函数体。

### C2. 新增生成类推导

```
latent_tokens   = (F / t_vae / t_patch) × (H / s_vae / h_patch) × (W / s_vae / w_patch)
tokens_total    = latent_tokens + audio_tokens + text_tokens   # 看 text_conditioning
flops_step_lin  ∝ tokens_total × active_params
flops_step_attn ∝ tokens_total²                                # 稀疏时按 SSTA 的 tile/topk 折算
flops_e2e       = Σ_stage(steps × forwards_per_step × flops_step) + 文本编码一次 + VAE decode 一次
weights_resident_bytes = Σ(checkpoint_variants[].resident)
```

**逐因子验**（CLAUDE.md 的硬要求）：报 `latent_tokens` 之前，六个因子
（F、H、W、t_vae、s_vae、patch）每个都要能指到来源。
MiniMax-H3 的卡自己把「VAE 压缩」和「patch 后的有效压缩」分开写，正是因为这两个数常被合并后报错。

**两路对账（Gate B）**：`latent_tokens` 可以从 config 推，也可以从厂商给的
「某分辨率某帧数」反推。两路不一致就报，不替人选边 —— 和现有 `reconcile()` 同样的做法。

### C3. 参数量按组件先分一层

`from_shapes()` 的统计逻辑可以复用，但要先按 A3 的 `component()` 分一层，
再在 `dit` 组件内分子系统。否则 H3 的文本编码器会进 DiT 的参数量
（A2 已验证这个归类错误真实存在）。

`params.breakdown` 的 key 用 A3 的新子系统集；`params.excludes` 用来声明
「声明值不含文本编码器 / VAE」这类口径差。

---

## D. 其它（不阻塞，但同一批做更省事）

| 项 | 改什么 |
|---|---|
| ~~`scripts/validate.py`~~ | **不用改**（已验证）。它只做 schema 校验 + 证据覆盖 + 防泄漏，三件都与 kind 无关；三个新必填由 schema 的 `allOf` 管，`leaf_paths` 对 `generation.*` 的前缀继承天然生效（列表不展开下标，`buckets` 整体算一个归属点）。拿一份生成类记录跑 `--data-dir` 实测：0 错，只剩「未核查」这类与本改动无关的告警 |
| `scripts/fetch_hf.py` / `scaffold.py` | 生成类的 config 散在 `transformer/` `vae/` `text_encoder/` 等子目录，还有 LTX-2.3 那种单文件打包、MAGI-2 那种 `.pth`/`.ckpt`。取数要能按子目录分别拉 |
| `scripts/fetch_dtypes.py` | 复用 A3 的分层归类（和 `derive.py` 共用同一份，不能两套） |
| `site/` | 生成类详情页的「性能」区默认留空并显式说明（Gate A §5.3）；`data.ts` 的 `diff: 'same-basis'` 守卫要认识「VAE 压缩率 × patch」这个新口径对 |
| `data/tracking.yaml` | 追加 Gate A §6.4 的 `video_policy` / `video_families` / `video_watchlist` |
| 双周采集 | 加 ModelScope 源；按 `createdAt` 扫全量 video pipeline_tag，不要只按下载量排序（Gate A §7 第 16 条：MAGI-2 的 HF 下载量显示为 0，按榜单扫直接漏掉） |

---

## E. 执行顺序与第一批录入

从 `main`（现 37e9404）直接开分支，不需要先合 `p1-glm`。
**第一批录哪三份已经定了**（2026-10-30，用户拍板）：
`wan2.2-ti2v-5b` → `minimax-h3` → `magi-2-preview`。
选这三份的理由与分工见 §F。

### 第一段 —— 现在就能做

1. **B1** `family` 枚举 —— 一行，解除整条线的阻塞
2. **B2 + B3 + B4** schema 重建（`generation` 块是新建子树，和任何 P1 改动不重叠）
3. **C1** `derive.py` 按 `kind` 分流（`vendor_declared` 分支已在 main，在它之上加入口判断）
4. **C2** 生成类推导（latent token 数、单步/端到端 FLOPs、常驻权重字节）
5. **A3a** 组件层 `component()` + `data/schema/components.json`（不碰正则表，不撞车）
6. **C3** 参数量按组件先分一层（依赖 A3a，不依赖 A3b）
7. **D** 里不依赖 `dit` 桶内细分的那几项：`validate.py`、`fetch_hf.py` / `scaffold.py`、
   `tracking.yaml`、双周采集源

→ 这一段跑完，**第 1、2 份记录就能录**（`wan2.2-ti2v-5b`、`minimax-h3`）。

### ⏸ 卡点：等 P1 五条线跑完、共享改动整合进 main

待整合的两项见 §0：GLM 线的 `subsystems.py`、Kimi 线的 `dtypes.py`（INT4-in-INT32）。
前者和 **A3b** 改同一张顺序敏感的正则表。

### 第二段 —— P1 整合之后

8. **A3b** `dit` 桶内正则扩展，跑 LLM 侧全量回归确认输出不变
9. **D** 里的 `fetch_dtypes.py`（和 `derive.py` 共用同一份归类，不能两套）、`site/`
10. 录第 3 份：`magi-2-preview`

### 验收标准

三份都能过 `validate.py` 与 `derive.py`（返回 0，用 `.venv/bin/python`），
`--list-judgements` 里的新增项逐条看过，且 LLM 侧现有记录（`main` 上 15 份，P1 还在加）的
`derive.py` 输出一字不变。A3a / A3b 另有各自的归类验收，见 §A3。

---

## F. 第一批三份的分工

| 顺序 | model_id | 它压的是什么 | 依赖 |
|---|---|---|---|
| 1 | `wan2.2-ti2v-5b` | **对照组。** 单 checkpoint、单文本编码器、无音频、config 齐全 —— 整条线里最简单的一份。先用它确认改完的管线在「正常模型」上不出错，比先录最难的那个稳 | 第一段 |
| 2 | `minimax-h3` | **压 schema 字段。** 四个大模型（DiT 33B ×2 + Qwen3-VL-32B 33B + VAE 5.2B）、未开源的两个模块、可预计算的 13B AdaLN、音视频双 scheduler、CFG 已蒸馏 | 第一段（组件级账靠 A3a） |
| 3 | `magi-2-preview` | **压脚本归类。** 36 层 ×3072 专家的 MoE 账、mHC 残差、attention sink、跨厂商复用的 Wan2.2 VAE、两阶段流水线 | **A3b**（DiT 桶内细分） |

第 2 份要留的 open question（Gate A §7 已列）：H3 的默认步数卡上没写；
「33B 里 13B 推理可免」没验过，**不要把 33 − 13 = 20 写进任何字段**，那是两个口径相减。

第 3 份要先解决的（Gate A §7 第 17、18 条）：MAGI-2 的采样参数卡上一条都没有，
要去读 sand.ai 技术博客或 GitHub 推理代码；MoE 的 top-k 与「每 token 激活 6B」的口径不定，
激活参数和 EP 通信量就算不出来。

---

## G. 本线之外的三件待办

这三件不属于第 1 步，列在这里是为了不丢。

1. **LTX-2.5 申请访问 —— 需要用户的 HF 账号，本会话做不了。**
   `Lightricks/LTX-2.5` 与 `Lightricks/LTX-2.5-Diffusers` 两个仓都 gated，
   连 `README.md` 都取不到。批下来之前 `ltx` family 的最新记录只能停在 `ltx-2.3`，
   名册里要一直挂 `status: blocked-gated`。

2. **世界模型 / 交互式视频线的 Gate A —— 建议排在第一批录完之后。**
   候选见 Gate A §2.4（Hy-Nomas、Matrix-Game-4.0、AlayaWorld/Evoke、Cosmos3、
   LongLive-2.0、lingbot-world）。它们自回归逐块生成，KV cache 与流式时延都回来了。
   现在开会凭空猜字段 —— 它能复用多少 `generation` 块，要先看视频线录完三份之后那个块长什么样。

3. **图像生成的小勘探 —— 2–3 个模型即可，目的只是验证扩展位。**
   PLAN §3.2 要求 schema 按 image-gen + video-gen 一次建齐，但本轮**一个图像模型都没调研**，
   凭空建字段必然漏。图像侧没有时间维（无帧数 / 时长 / fps），但有视频侧没有的多尺度与 refiner。
   扫视频时撞见过几个线索，**未经查证，只是名字**：`Qwen-Image-2.1` / `Qwen-Image-2512`、
   `Z-Image-Turbo`（以上阿里）、`HunyuanImage-2.1`（腾讯）、`FLUX.1-Redux-dev`（BFL，
   HunyuanVideo-1.5 的 SigLIP 就从这个 gated 仓取）。要做得先按视频线那套方法正经扫一遍 HF。

---

## H. 实现中发现的两件事

写在这里是因为它们都不在原清单里，而且第二件会影响到每一份记录的数。

### H1. `validate.py` 不用改

原清单 §D 写着要给它加生成类分支。实测不需要：它只做 schema 校验、证据覆盖、防泄漏，
三件都与 `kind` 无关。三个新必填由 schema 的 `allOf` 条件分支管住；
证据覆盖的 `leaf_paths` 对 `generation.*` 的前缀继承天然生效，
而且它不把列表展开到下标，所以 `buckets` / `text_encoders` / `pipeline_stages`
各算一个归属点，不会要求逐档位写证据。

拿一份生成类记录跑 `validate.py --data-dir <临时目录>` 实测：0 错误，
只剩「还没有 reviewed 块」这类与本改动无关的告警。

### H2. 时间因果 VAE 的 latent 帧数不是 F / t_vae

**这条会影响每一份视频记录的 token 数，原清单漏了。**

时间因果 VAE 的第一帧单独编码，之后每 `t_vae` 帧压成一帧：

```
latent 帧数 = (F - 1) / t_vae + 1        而不是  F / t_vae
```

这正是各家帧数都写成 `4n+1` / `8n+1` 的原因 —— Wan 的 121、HunyuanVideo 的 129、
LTX 的 `8n+1`、HunyuanVideo-1.5 训练数据要求的 `4n+1`。

按 `F / t_vae` 算，Wan2.2-TI2V-5B 的 121 帧会得到 **30.25** 这种非整数，真值是 **31**。
偏差只有 2%，但它是**口径错误而不是精度误差**：一路乘下去进单步 FLOPs、再乘步数进端到端。
而且非整数本身就是「这个公式用错了」的信号，不该被静默吞掉。

`derive.py` 现在按 `generation.vae.causal` 分支，并在任一轴算出非整数时报出来；
`causal` 未填而时间轴不整除时，提示去填 `causal: true`。

> 顺带一个可以写进专题的数：Wan2.2-TI2V-5B 在 1280×704×121f 下 latent token 27,280，
> **单步注意力 FLOPs 已经和线性项持平（各占 50%）**。这就是「生成类是计算受限、
> 不是 decode 那种带宽受限」最直白的证据 —— 而且 n² 那一项还会随分辨率继续涨。
> 数字本身不要写进分析文字，它在 `data/derived/` 里。
