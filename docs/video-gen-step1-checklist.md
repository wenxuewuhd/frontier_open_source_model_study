# 视频生成线 · 第 1 步补管线实施清单

> 这是 [video-gen-gate-a.md](video-gen-gate-a.md) §5.5 的展开，供第 1 步执行。
> Gate A 已定：12 份记录、8 个 family，录入先只做视频。本清单只覆盖 **schema + derive.py +
> subsystems.py** 三件，不涉及录记录本身。
> 编制日期 2026-10-30，基线 `main` = 352c114。

## 0. 先决条件：等 P1 的共享脚本改动落地

Gate A 交接时记的是「三处待整合」。本轮复核了全部六个 worktree 与所有分支，实际状态是：

| 待整合项 | 实际位置 | 状态（2026-10-30 复核） |
|---|---|---|
| `derive.py` 的 `bytes_basis` / 厂商公布 KV 字节 | `p1-glm` 分支，commit `bad7cf6` | **已提交**，尚未并回 main。同一 commit 还改了 `model.schema.json`（新增 `kv_sharing.declared_per_token_bytes`、改写 `precision.kv_cache_components.source` 的描述） |
| `scripts/lib/subsystems.py` | — | **未见任何分支或工作区有改动**。`git diff main..<branch> -- scripts/lib/` 在五个 p1 分支上全为空 |
| `scripts/lib/dtypes.py` 的 INT4-in-INT32 打包 | — | **同上，未见改动** |

复核命令（可重跑）：

```bash
for b in p1-glm p1-kimi p1-minimax p1-qwen p1-stepfun; do
  echo "=== $b"; git diff --stat main..$b -- scripts/ data/schema/
done
```

**所以依赖关系比预想的轻**：真正会和本清单撞车的只有 `p1-glm` 那一个 commit，
而且它改的是 `derive.py` 的 KV 分支与 schema 的 `kv_sharing` 子树 —— 和本清单的改动点
（按 `kind` 分流、新建 `generation` 块）**不重叠**。另外两处若后续出现，撞车点见各条目的「P1 依赖」标注。

**执行顺序建议**：先把 `p1-glm` 并回 main，再在 main 上开本清单的分支。
如果 `subsystems.py` / `dtypes.py` 的改动在此期间出现，**A2 必须等它们落地**（见 A2 的说明）。

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

### A3. 改法

**第一层：按顶层组件前缀分桶。** 在 `classify()` 之前加一个 `component()`：

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

**第二层：桶内跑子系统正则。** `dit` 桶内用扩展后的正则；其余桶内一律不再细分
（VAE 的内部结构对负载账没意义，整块算就行）。

**正则要加的条目**（按优先级顺序插入，注意顺序即优先级）：

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

**验收**：A2 那张表的 26 行全部归到「应该归到」列，且 LLM 侧现有记录的
`derive.py` 输出一字不变（跑 `data/models/` 下全部记录对比）。

> **P1 依赖**：如果后续出现 `subsystems.py` 的 P1 改动，**本条必须等它合入再做** ——
> 两边都在改同一张正则表且顺序敏感，分别改完再合并几乎必然出错。
> 合入后要重跑一次 LLM 侧全量回归，确认 P1 的新正则没有被本条的分层包住。

---

## B. `data/schema/model.schema.json`

### B1. `family` 枚举（阻塞一切）

现在是封闭枚举 `["deepseek","glm","minimax","kimi","qwen","stepfun","other"]`。
加：`wan, hunyuan-video, minimax-video, ltx, cogvideo, stepvideo, sana-video, magi`。

不加就一份记录都写不进去。**这是整条线的第一个阻塞点。**

> **P1 依赖**：无（p1-glm 的 schema 改动在 `kv_sharing` 子树，不碰 `family`）。

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

> **P1 依赖**：B3 的 `precision` 改动与 p1-glm 的 `precision.kv_cache_components.source`
> 描述改写在同一个对象里，**同一文件相邻行，会产生 merge 冲突但不是语义冲突**。
> 先合 p1-glm 再做即可，冲突手工取并集。

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

> **P1 依赖**：`kv_memory()` 正是 p1-glm 改过的函数（加了 `vendor_declared` 分支）。
> **本条必须在 p1-glm 合入之后做**，否则会把那个分支覆盖掉。
> 改法上没有冲突：本条是在函数入口加一道 kind 判断提前返回，不动函数体。

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
| `scripts/validate.py` | 对生成类跳过 LLM 专属必填；加 B2 的三个新必填；确认 `--list-judgements` 对 `generation.*` 生效 |
| `scripts/fetch_hf.py` / `scaffold.py` | 生成类的 config 散在 `transformer/` `vae/` `text_encoder/` 等子目录，还有 LTX-2.3 那种单文件打包、MAGI-2 那种 `.pth`/`.ckpt`。取数要能按子目录分别拉 |
| `scripts/fetch_dtypes.py` | 复用 A3 的分层归类（和 `derive.py` 共用同一份，不能两套） |
| `site/` | 生成类详情页的「性能」区默认留空并显式说明（Gate A §5.3）；`data.ts` 的 `diff: 'same-basis'` 守卫要认识「VAE 压缩率 × patch」这个新口径对 |
| `data/tracking.yaml` | 追加 Gate A §6.4 的 `video_policy` / `video_families` / `video_watchlist` |
| 双周采集 | 加 ModelScope 源；按 `createdAt` 扫全量 video pipeline_tag，不要只按下载量排序（Gate A §7 第 16 条：MAGI-2 的 HF 下载量显示为 0，按榜单扫直接漏掉） |

---

## E. 建议执行顺序

1. 把 `p1-glm` 并回 `main`（它带着 `derive.py` 的 KV 分支与 schema 的 `kv_sharing` 改动）
2. **B1** family 枚举 —— 一行，解除阻塞
3. **A3** subsystems 分层 + 正则扩展，跑 LLM 侧全量回归确认输出不变
4. **B2 + B3 + B4** schema 重建
5. **C1** derive 按 kind 分流（必须在第 1 步之后）
6. **C2 + C3** 生成类推导与组件分层
7. **D** 其余
8. 用 `magi-2-preview` 做第一份记录验收 —— 它一次性压到 MoE 账、mHC、sink、
   跨厂商 VAE 复用、两阶段流水线；`minimax-h3` 压的是另一头（组件与部署口径），紧随其后

验收标准：两份记录都能过 `validate.py` 与 `derive.py`（返回 0，用 `.venv/bin/python`），
`--list-judgements` 里的新增项逐条看过，且 LLM 侧现有记录（`main` 上 15 份，P1 还在加）的 `derive.py` 输出一字不变。
