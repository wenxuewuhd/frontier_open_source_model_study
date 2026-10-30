# 视频生成跟踪线 · Gate A 选题

> 状态：第 0 步产物，只做选题与字段需求勘探，**不改 schema、不改脚本、不录记录**。
> 取数日期：2026-10-30。取数范围：HuggingFace 模型卡 / config / safetensors 头（range 读）/ 厂商参考实现 / 厂商 GitHub。
> 对应 [PLAN.md](../PLAN.md) §2 覆盖范围「多模态 · 生成」、§3.2 生成类 schema、§7 的 P2。

本文件回答四件事：录哪几家几代（§1–§3）、每个入选候选到底什么在变（§4）、
§3.2 的字段轮廓要改成什么（§5）、family 与命名怎么切（§6）。§7 是没查到或查不准的部分。

**口径提示**：下文每条事实后面括号里标来源与证据口径。`official` = 模型卡 / 技术报告原话；
`derived` = 由 config 或权重头直接算出；`judgement` = 从参考实现或引擎文档读出的解释；
`unknown` = 没找到。按 [CLAUDE.md](../CLAUDE.md) 的来源优先级，引擎文档（sglang/vLLM/TRT-LLM cookbook）
只能标 `judgement`，不能当 `official`。

---

## 1. 入选清单

一共 **12 份记录、8 个 family**，分两波。第一波 7 份是当代前沿，第二波 5 份是历史基线与旁支路线。
波次只是排期建议，Gate A 的结论是这 12 份都该录。

### 第一波（当代前沿，7 份）

| family | model_id | hf_repo | 发布日期 | 凭什么入选 |
|---|---|---|---|---|
| `minimax-video` | `minimax-h3` | `MiniMaxAI/MiniMax-H3` | 2026-07-28 | 音视频在**同一条注意力序列**里联合去噪；把一个 33B 的 Qwen3-VL-32B 全权重当文本编码器随仓发布；VAE 解码器本身是 36 层 Transformer（5.2B）。一次部署要驻留四个大模型 |
| `ltx` | `ltx-2` | `Lightricks/LTX-2` | 2026-01-03 | VAE 压到 8×32×32、128 latent 通道后 DiT 侧 patch=1；独立音频分支 + vocoder；官方同时发 FP8 与 NVFP4；两阶段 latent 上采样流水线 |
| `ltx` | `ltx-2.3` | `Lightricks/LTX-2.3` | 2026-03-04 | 19B→22B 的增量**不在主干**：主干仍是 48 层 4096，涨的是 connector 与文本投影（权重实测，见 §4.3）。单文件把 DiT/VAE/audio VAE/vocoder 打包成一个 46 GB 文件，是「权重存储口径」的反例 |
| `wan` | `wan2.2-t2v-a14b` | `Wan-AI/Wan2.2-T2V-A14B` | 2025-07-24 | 第一个把 MoE 引进视频 DiT 的开源模型，但**路由轴是时间步不是 token**：两套各 14B 的专家按 boundary 切换。这和 LLM 侧的 MoE 不是一回事，硬塞进 `arch.moe` 会造出假事实 |
| `wan` | `wan2.2-ti2v-5b` | `Wan-AI/Wan2.2-TI2V-5B` | 2025-07-18 | 同代同厂但**换了 VAE**（4×8×8/16ch → 4×16×16/48ch），latent token 数掉到 1/4。按「换 VAE 就是另一回事」的判据单独录 |
| `hunyuan-video` | `hunyuanvideo-1.5` | `tencent/HunyuanVideo-1.5` | 2025-11-20 | SSTA 时空块稀疏注意力（厂商称 10s/720p 端到端 1.87× vs FA3）；VAE 换到 16×/4×/32ch 后 patch=(1,1,1)；CFG 蒸馏与步数蒸馏**两根轴同时存在**；11 个各 33 GB 的全量 checkpoint 并存 |
| `magi` | `magi-2-preview` | `sand-ai/MAGI-2-preview` | 2026-08-03 | **唯一一个 token 级 MoE 的视频 DiT**：114B 总参 / 每 token 激活 6B，36 层各 3072 个专家。同时带 attention sink、逐头输出门控、**mHC 残差（×4 扩张）**，并复用 Wan2.2 的 VAE。几乎把本仓 LLM 侧跟踪的全部结构一次性搬进了视频生成 |

### 第二波（历史基线与旁支，5 份）

| family | model_id | hf_repo | 发布日期 | 凭什么入选 |
|---|---|---|---|---|
| `wan` | `wan2.1-t2v-14b` | `Wan-AI/Wan2.1-T2V-14B` | 2025-02-25 | Wan 线的差分基线。单流 DiT + cross-attn 文本注入 + 4×8×8/16ch VAE，是 2025 年的「标准配方」，Wan2.2 的两处改动都要对着它说 |
| `hunyuan-video` | `hunyuanvideo-13b` | `tencent/HunyuanVideo` | 2024-12-01 | 双流→单流 MMDiT（20 double + 40 single）的代表；**嵌入式引导**（`guidance_embeds: true`，每步单次前向），与 1.5 代退回真 CFG 形成对照 |
| `cogvideo` | `cogvideox-1.5-5b` | `zai-org/CogVideoX1.5-5B` | 2024-11-02 | PLAN §2 要求的 2024 年锚点。文本与视频 token **拼进同一条全注意力序列**（不是 cross-attn），expert-adaLN；且带时间维 patchify（`patch_size_t: 2`） |
| `stepvideo` | `stepvideo-t2v-30b` | `stepfun-ai/stepvideo-t2v` | 2025-02-14 | 压缩率最激进的一代（16×16 空间 / 8 时间 / 64 latent 通道）；双文本编码器（自家 19.6B Step-LLM + HunyuanCLIP）；**文本编码器、VAE、DiT 拆成三个独立服务**部署。另外它是少数在卡上给了峰值显存与端到端耗时的 |
| `sana-video` | `sana-video-2.0-5b` | `Efficient-Large-Model/SANA-Video_2.0_5B_720p` | 2026-08-20 | 唯一一个在视频 DiT 里用**线性注意力**的开源模型（75% gated linear + 25% dense softmax anchor）。直接对上 LLM 侧「线性/混合注意力」专题；而且它复用 LTX-2.3 的 VAE，是跨 family 共用组件的现成用例 |

> `sana-video-2.0-5b` 是唯一一个可以争论的：它来自 NVlabs/MIT 而不是本仓主要跟踪的厂商，
> 下载量也不大。保留它的理由是**结构唯一性**，不是影响力。如果要砍，砍它，不要砍别的。

### 为什么是 12 份而不是 7 份

本仓 `data/tracking.yaml` 的 policy 是「每个家族追最新旗舰」。那条 policy 是给语言模型线定的，
前提是**家族内代际差异可以靠 lineage 文字讲清楚**。视频线不成立：
Wan2.2 的两个档位用的是两套 VAE，LTX-2→2.3 的参数涨幅不在主干，HunyuanVideo 1.5 相对 13B 把
VAE、注意力形态、文本编码器组合全换了。这些差异都落在**会进推导公式的数**上，不是叙述。
所以视频线的 policy 应当写成：**每个 family 追最新旗舰，外加一个差分基线；VAE 或注意力形态变了的同代档位单独录。**

---

## 2. 不录的逐条依据

### 2.1 纯档位 / 纯任务变体（主干与 VAE 不变）

| 不录 | 依据 |
|---|---|
| `Wan2.1-T2V-1.3B` | 纯参数档位，dim/层数等比缩小，VAE 与文本编码器相同 |
| `Wan2.1-I2V-14B-480P` / `-720P` | 纯分辨率档 + i2v 条件变体 |
| `Wan2.1-FLF2V-14B-720P` | 首尾帧条件变体 |
| `Wan2.2-I2V-A14B` | 与 `Wan2.2-T2V-A14B` 同结构、同 VAE、同 MoE 切分，只是条件输入不同 |
| `Wan2.2-S2V-14B` / `Wan2.2-Animate-14B` / `Wan2.2-Animate-2-14B` / `Wan-Dancer-14B` | 音驱、角色动画等任务化模型，主干与 VAE 不变 |
| `Wan2.1-VACE-14B` / `-1.3B` | 统一编辑的条件分支。**保留疑问**：若后续确认 VACE 改了 token 拼接方式（条件 token 进不进主序列），要重新评估 —— 那会改序列长度 |
| `CogVideoX-2b` / `-5b` / `-5b-I2V` / `CogVideoX1.5-5B-I2V` / `-SAT` | 档位与条件变体；`CogVideoX-5b → 1.5-5B` 的差分（加时间 patchify、分辨率变大）写进 `cogvideox-1.5-5b` 的 `lineage.changed_from_predecessor` |
| `stepvideo-ti2v` | i2v 变体 |
| `stepvideo-t2v-turbo` | 步数蒸馏档，进 `stepvideo-t2v-30b` 的蒸馏字段 |
| `HunyuanVideo-I2V` / `-Avatar` / `-Custom` / `-Portrait` / `Hunyuan-GameCraft-1.0` | HunyuanVideo 13B 的任务变体 |
| `HunyuanVideo-1.5` 的 11 个 transformer checkpoint | 480p/720p × t2v/i2v × {基座, CFG 蒸馏, 步数蒸馏, 稀疏} + 两个 SR，**全部进同一份记录的变体表**。唯独 `720p_i2v_distilled_sparse` 要在记录里点名 —— 它是目前唯一一个真开了 SSTA 的权重 |
| `MiniMax-H3` 的 FL2VA / Ref2VA | 同一份记录的两个任务 checkpoint；但 Ref2VA 多带一套 33B `transformer_ref`，这要进权重存储口径 |
| `SANA-Video_2.0_5B_720p_4step` / `SANA-Video_2B_480p` / `_2B_720p` | 步数档与参数档 |
| `Lightricks/LTX-2.3-fp8` / `-nvfp4` | 厂商自己发的量化档，进 `ltx-2.3` 的 `precision.quantization`，不单开 |

### 2.2 被取代或已停更

| 不录 | 依据 |
|---|---|
| `Lightricks/LTX-Video`（0.9.x，13B） | 被 LTX-2 整体取代：VAE、文本编码器、音频分支全换。0.9.x 唯一值得留的是「极高压缩 VAE」这个历史事实，写进 `ltx-2` 的 lineage 即可 |
| `genmo/mochi-1-preview`（10B AsymmDiT） | 2024-10 发布后无后续；AsymmDiT（非对称双流）这条路线已被 MMDiT 取代，没有继承者可以做差分 |
| `hpcai-tech/Open-Sora-v2`（11B） | 价值在训练成本与全流程开源，不在推理结构。其 DC-AE 高压缩 VAE 值得在专题里提一句，模型本身影响力不足以占一份 |
| `rhymes-ai/Allegro` 系列 | 2024 小规模，已停更 |
| `ali-vilab/*`（ModelScope T2V、i2vgen-xl 等） | 2023 一代，早于 PLAN §2 的 2024 起点，且结构路线（UNet）已断 |
| `Skywork/SkyReels-V1`（HunyuanVideo 微调）/ `V2`（Wan 微调）/ `V3` | 都是在别家主干上微调。**但 V2 的 DF（diffusion forcing）长视频路线值得作为观察条目单列**，它改的是采样调度不是主干 |

### 2.3 第三方派生，进不了证据体系

按 CLAUDE.md 的来源优先级，非厂商发布的权重一律不录。这一类数量极大，列代表：

- 量化/打包转换：`QuantStack/*-GGUF`、`unsloth/*`、`city96/*`、`bullerwins/*`、`Abiray/*`、
  `ChrisColeTech/*`、`leejet/*`、`MATLOWAI/*`、`rockerBOO/minimax-h3-nvfp4-convrot`
- 步数/CFG 蒸馏 LoRA 与合并权重：`larryvrh/MiniMax-H3-Turbo-Lora`、`drbaph/*`、
  `FastVideo/FastVideo-FastH3-4-step-*`、`lightx2v/Wan2.2-Distill-Loras`、
  `ZhengmingYu/DMAD`（MiniMax-H3 上的蒸馏 LoRA，arXiv 2610.02188）、
  `alibaba-pai/MiniMax-H3-Acc-LoRAs`、`Efficient-Large-Model/LongLive-Plug-*`
- 微调 / 重打包：`SulphurAI/Sulphur-2-base`（LTX-2.3 派生）、`krea/krea-realtime-video`（Wan2.1 派生）、
  `Comfy-Org/Bernini-R`（ByteDance 原仓的 Comfy 重打包）、`zai-org/RealVideo`（Wan2.2-S2V 微调）、
  `WarmBloodAban/Minimax-h3_Singularity`、`alibaba-pai/Wan2.*-Fun-*`
- 控制与风格 LoRA：`alibaba-pai/*-Controlnet-Union-*`、`fal/*`、`Lightricks/LTX-2.*-IC-LoRA-*`

这些**不进记录，但进专题**：它们是「步数压缩实际能做到几步」的最好旁证（社区在 H3 上做到 4–8 步，
在 LTX 上做到 8 步），写进「生成模型负载」专题时引用，证据口径标 `judgement`。

### 2.4 世界模型 / 自回归交互式视频 —— 建议另开一条线

`tencent/Hy-Nomas`（2026-10-21，自回归 + 相机控制）、`Skywork/Matrix-Game-4.0`、
`AlayaLab/Evoke` + `AlayaWorld` 系列、`robbyant/lingbot-world-*`、`nvidia/Cosmos3-Super` / `-Nano`、
`Efficient-Large-Model/LongLive-2.0-5B`、`tencent/HY-WorldPlay`。

这一类**不是不重要，是负载模型完全不同**：自回归逐块生成 → KV cache 回来了、要算每帧时延、
要做流式与首帧延迟、动作/相机是逐帧条件。把它们塞进「一次性生成一整段」的 DiT 这条线，
会让「单步 × 步数」这个基本量在半数记录上失效。建议 P2 内单独立「世界模型 / 交互式视频」一条线，
schema 可以共用生成类的 VAE / 文本编码器块，但采样与负载块要另建。本轮一律不录。

### 2.5 其它判断

| 不录 | 依据 |
|---|---|
| `OpenMOSS-Team/MOVA-360p` / `-720p` | 音视频联合，与 H3 / LTX 同类。结构上没有看到这两家没有的东西，采用度也低一档。进观察名单，若 720p 版确认有独立结构再议 |
| `ByteDance/Bernini-R` | MLLM 语义规划器 + DiT 渲染器的两段式，latent 层做语义规划确有新意（arXiv 2605.22344）。但只开源了 Renderer 一侧，planner 是否开源未确认，且下载量极低。进观察名单 |
| `zai-org/SCAIL-2` | 姿态驱动的角色动画，任务模型 |
| `netflix/void-model` | 下载量为 0，未确认是否可用 |
| `Lightricks/LTX-2.5` | **见 §3.3** —— 想录但取不到证据 |

---

## 3. 观察名单

观察名单与正式记录分开放：它们**进不了本仓的证据体系**，因为没有可校验的权重。
放在这里是为了让「我们知道它存在、也知道为什么不录」留痕。

### 3.1 完全无公开权重（闭源 API）

| 系统 | 厂商 | 为什么只能观察 |
|---|---|---|
| Sora 2 | OpenAI | 无权重发布，架构信息只有技术博客，无法复算任何一个数 |
| Veo 3.x | Google | 同上 |
| 可灵 Kling | 快手 | 同上 |
| Seedance / 即梦 | 字节跳动 | 同上（字节开源的是 Bernini-R、SeedVR2 等旁支，不是主力生成模型） |
| Runway Gen-4 / Pika | Runway / Pika | 同上 |
| Hailuo 视频（API 版） | MiniMax | H3 开源了 Base，但线上产品还包含未开源的两个模块，见下 |

这些模型可以在专题里作为质量参照提及，**不得出现在平台矩阵、负载推导或任何带数字的表里**。

### 3.2 部分开源：系统里有一块取不到

这一类是本轮最值得记一笔的发现 —— 它们**有正式记录**，但记录里必然留着洞：

| 模型 | 取不到的那块 | 后果 |
|---|---|---|
| `minimax-h3` | H3-Context-IR（多模型编排的提示预处理）、H3-Regenerate-2K（2K 重生成） | 开源的只是 768p 的 Base。端到端耗时、2K 路径的负载**在公开仓永远算不出来**，只能算 Base 一段 |
| `hunyuanvideo-1.5` | 官方 HunyuanMLLM 文本编码器未发布，文档推荐用 Qwen2.5-VL-7B-Instruct 替代；SigLIP 视觉编码器要从 gated 的 `black-forest-labs/FLUX.1-Redux-dev` 取 | 记录里「文本编码器」填什么要先定口径；而且一个模型要凑四份外部权重、其中一份要申请 |
| `hunyuanvideo-1.5` | 卡上的 Open-source Plan 仍写着 "Release all model weights (Sparse attention, distill model, and SR models)" 未勾选，但仓里已经有 sparse / distilled / sr 目录 | 卡与仓不一致，录的时候以仓为准并留 open question |

### 3.3 LTX-2.5：想录但读不到

`Lightricks/LTX-2.5`（2026-07-23，2.27M 下载、9879 likes，是当前下载量最高的开源视频模型之一）
与 `Lightricks/LTX-2.5-Diffusers` **两个仓都是 gated**，未授权时连 `README.md` 和 `config.json` 都取不到。

能确定的只有从公开元数据和周边仓推出来的：
- 参数规模仍是 22b 档（由 `Lightricks/LTX-2.5-22b-IC-LoRA-*` 的仓名推，`judgement`）
- Diffusers 版的**文件清单**可见（HF API 不受 gate 限制），相对 LTX-2 新增了
  `transformer_full/`、`diffusion_decoder/`、`duration_head/`、`prompt_enhancer/`、
  `temporal_latent_upsampler/`、`processor/` 六个组件目录（`derived`，来自文件清单）

社区有 `unsloth/LTX-2.5-FP8`、`comfyicu/LTX-2.5`、`Abiray/LTX-2.5-Distilled-GGUF` 等镜像，
照它们读等于用第三方转换推翻厂商口径，正是 model-audit 规则 10 禁止的那条链。

**结论**：LTX-2.5 列为「待申请访问」，不进第一波。`ltx` family 的最新旗舰暂记为 `ltx-2.3`，
并在 tracking 名册里写明 LTX-2.5 存在但 gated。`hf.gated` 这个 schema 字段已经有了，正好用它。

### 3.4 观察条目（路线，不是模型）

- **Diffusion Forcing / 自回归长视频**：SkyReels-V2-DF、LongLive-2.0。改的是采样调度与因果性，不是主干
- **训练无关的缓存加速**：deepcache / teacache / taylorcache / MagCache（HunyuanVideo-1.5 卡上列了全部三种）。
  它们直接改变「步数 × 单步」这个乘式的有效步数，是生成类特有的一类加速
- **世界模型线**：见 §2.4

---

## 4. 入选候选的事实表

每个表只写**候选之间会变的量**。空格表示没查到，不是没有。

### 4.0 一张横向对照（最关键的三列）

| model_id | VAE 压缩 (t×h×w) / latent ch | DiT patch (t,h,w) | 有效 token 压缩 (t×h×w) | 每步前向次数 | 默认步数 |
|---|---|---|---|---|---|
| `wan2.1-t2v-14b` | 4×8×8 / 16 | (1,2,2) | 4×16×16 | 2（CFG） | — |
| `wan2.2-t2v-a14b` | 4×8×8 / 16 | (1,2,2) | 4×16×16 | 2（CFG 3.0/4.0） | 40 |
| `wan2.2-ti2v-5b` | 4×16×16 / 48 | (1,2,2) | 4×32×32 | 2（CFG 5.0） | 50 |
| `hunyuanvideo-13b` | 4×8×8 / 16 | (1,2,2) | 4×16×16 | **1**（嵌入式引导 6.0） | 50 |
| `hunyuanvideo-1.5` | 4×16×16 / 32 | (1,1,1) | 4×16×16 | 2（CFG 6）/ 1（CFG 蒸馏版） | 50 / 8–12（步数蒸馏） |
| `minimax-h3` | 4×16×16 / 24 | (1,2,2) | 4×32×32 | **1**（发布权重已 CFG 蒸馏） | 未在卡上声明 |
| `ltx-2` | 8×32×32 / 128 | (1,1,1) | 8×32×32 | 2（stage1 CFG 4.0）/ 1（stage2 CFG 1.0） | 40 + 3（两阶段）/ 8（蒸馏档） |
| `ltx-2.3` | 8×32×32 / 128 | (1,1,1) | 8×32×32 | 1（蒸馏档 CFG=1） | 8（蒸馏档） |
| `cogvideox-1.5-5b` | 4×8×8 / 16 | (2,2,2) | 8×16×16 | 2（CFG） | — |
| `stepvideo-t2v-30b` | 8×16×16 / 64 | 1 | 8×16×16 | 2（CFG 9.0） | 50 |
| `sana-video-2.0-5b` | 8×32×32 / 128（复用 LTX-2.3 VAE） | — | — | 2（CFG 8） | 50 |
| `magi-2-preview` | 4×16×16 / 48（复用 Wan2.2 VAE） | — | — | — | — |

> 这张表就是 §5 的论据：**VAE 压缩率和 DiT patch 必须分两个字段存**，两者相乘才是 token 口径。
> MiniMax-H3 的卡自己就这么写：「visual tokens 的有效空间下采样是 32×，时间下采样仍是 4×」（`official`）。
> 把它们合成一个数，和 CLAUDE.md 说的「不同口径之间不算倍数」是同一类错误。

---

### 4.1 `minimax-h3` — MiniMaxAI/MiniMax-H3

发布 2026-07-28（HF `createdAt`，`derived`）。License: MiniMax H3 Community License（`official`）。
arXiv 未在卡上给出。

**DiT 主干**（`transformer/config.json`，`derived`；参数量来自卡，`official`）

| 项 | 值 | 来源 |
|---|---|---|
| 架构族 | 单流（single-stream）稠密 Transformer，多模态 token packed 进一条序列 | 卡 `official` |
| 参数量 | 33B，其中约 13B 在 AdaLN 相关分支 | 卡 `official` |
| 层数 / hidden | 50 / 5376 | config `derived` |
| 注意力 | 56 heads × 128 head_dim = **7168 内维 ≠ 5376 hidden** | config `derived` |
| FFN | 14336 | config `derived` |
| refiner 层 | 2 | config `derived` |
| 位置编码 | 3D MM-RoPE (t,h,w)，`rope_freq_dim` 16，`rope_theta` 10000 | config + 卡 `official` |
| patch | (1,2,2) | config `derived` |
| 视频 latent 入口 | `in_channels` 24 | config `derived` |
| 音频 latent 入口 | `audio_in_channels` 32（**同一个 DiT**） | config `derived` |
| 文本条件入口 | `text_dim` 5120 | config `derived` |
| 注意力形态 | 本次发布只给全注意力；卡称原生支持稀疏注意力训练与推理，实现「将在后续更新中发布」 | 卡 `official` |
| 权重落盘 | 66,280,430,080 B，638 张量；首分片 BF16 52 + F32 12（in/out 投影与 time embedder 是 F32） | 权重头 `derived` |

**文本编码器**

| 项 | 值 | 来源 |
|---|---|---|
| 是什么 | **Qwen3-VL-32B 全预训练权重**，取其**第 50 层**的 hidden state 喂给 DiT | 卡 `official` |
| 是否随权重发布 | **是**，`text_encoder/` 14 个分片 | 仓清单 `derived` |
| 规模 | 66,714,780,128 B；text: 64 层 / hidden 5120 / 64 heads / 8 kv heads / inter 25600；vision: 27 层 / hidden 1152 / patch 16 | config + index `derived` |
| 注意 | 卡要求必须用 H3 仓里的 tokenizer（加了 `<d>` 等特殊 token） | 卡 `official` |

**VAE**

| 项 | 值 | 来源 |
|---|---|---|
| 视觉 VAE | 时间因果；空间 16×、时间 4×、24 latent 通道，记作 **f16t4d24** | 卡 `official` |
| config 对账 | `spatial_downsample_factors [2,2,2,2,1,1]` 积 = 16；`temporal [1,2,2,1,1,1]` 积 = 4；`latent_channels` 24 —— 与卡一致 | config `derived` |
| 解码器形态 | **ViT 解码器**：36 层、32 heads × 64、RoPE（`decoder_rope_theta` 100）。卡说是在编码器训完后另训的，为了降低解码成本 | config + 卡 `official` |
| VAE 权重量 | 10,415,475,936 B → 约 5.2B（BF16 口径） | index `derived` |
| 切片 | `clip_length` 17、`token_drop` 3 | config `derived` |
| 音频 VAE | DAC 风格；32 kHz 输入，`encoder_rates [2,4,4,5,5]` 积 800 → **40 Hz latent**；32 latent 通道；左右声道共用同一套编解码器、各跑一遍后合成立体声 | config + 卡 `official` |

**采样**

| 项 | 值 | 来源 |
|---|---|---|
| CFG | **发布权重是 CFG 蒸馏的**（卡原话："The released checkpoints are CFG-distilled Omni Transformer model weights"）→ 每步单次前向 | 卡 `official` |
| 默认步数 | **卡上没有写**。sglang cookbook 给 50；社区蒸馏 LoRA 做到 8 步（Turbo）、4 步（LightX2V） | 引擎文档 `judgement` |
| scheduler | 视频与音频各一套 `MiniMaxH3Scheduler`，shift 分别 12.0 / 3.0 | config `derived` |

**输出规格**（全部 `official`，来自卡的规格表）

时长 4–15 s；24 fps；短边默认 768 px；长宽比 21:9 / 16:9 / 4:3 / 1:1 / 3:4 / 9:16 等；
音频 32 kHz 立体声；2K 需走未开源的 H3-Regenerate-2K。

**发布精度与变体**

- 卡的 checkpoint 表写 Precision = **BF16**（`official`）；官方无量化版本
- 两个自包含 checkpoint：**FL2VA**（文生 / 首尾帧生）与 **Ref2VA**（多模态参考）
- Ref2VA 侧仓里有 `transformer` 与 `transformer_ref` **两套各 66.28 GB** 的 DiT（`derived`）

**一次部署的权重账**（`derived`，这是本条记录最值得报出去的数）

Ref2VA 路径：DiT 33B + DiT-ref 33B + Qwen3-VL-32B 文本编码器 33B + 视觉 VAE 5.2B + 音频 VAE
≈ **104B 参数、约 137 GB 落盘**，而卡同时说 33B 里约 13B 的 AdaLN 分支「推理时不需要加载」。
**这两个口径不能相减得到一个「实际显存」数** —— 13B 是否真能省掉要按张量名分桶实测，本轮没做。

**推理框架**（`official`，卡上列出）：SGLang、vLLM、diffusers、ComfyUI。
sglang 示例 `--num-gpus 4 --ulysses-degree 4`，即 **Ulysses 序列并行**，不是 TP/EP。

---

### 4.2 `ltx-2` — Lightricks/LTX-2

发布 2026-01-03（`derived`）。License: LTX-2 Community License。论文 arXiv 2601.03233
《LTX-2: Efficient Joint Audio-Visual Foundation Model》（`official`）。

**DiT 主干**（`transformer/config.json`，`derived`）

- 48 层、32 heads × 128 → hidden 4096；`in/out_channels` 128；**`patch_size` 1 / `patch_size_t` 1**
- `vae_scale_factors` **[8, 32, 32]**（t, h, w）直接写在 DiT config 里
- 音频分支同在一个 DiT：`audio_in_channels` 128、`audio_num_attention_heads` 32 × 64、
  `audio_cross_attention_dim` 2048、`audio_sampling_rate` 16000
- `caption_channels` 3840、`cross_attention_dim` 4096、`rope_type` "split"、`rope_double_precision` true
- 权重 37,758,861,824 B，首分片 BF16 448 + F32 26 → 约 **18.9B BF16**（卡称 19b）

**文本编码器**

- **Gemma-3-12B**（`Gemma3ForConditionalGeneration`），hidden 3840 / head_dim 256 / inter 15360，
  滑窗:全注意力 = 5:1（`layer_types`）（`derived`）
- **落盘 dtype 是 float32**：`total_parameters` 12,187,325,040、`total_size` 48,749,300,160 B（`derived`）。
  一个 12B 的文本编码器占 48.7 GB，这是部署时很容易漏算的一块

**VAE**（`vae/config.json`，`derived`）

- `latent_channels` **128**、`spatial_compression_ratio` **32**、`temporal_compression_ratio` **8**、`patch_size` 4
- `encoder_causal` true / `decoder_causal` false（编码器因果、解码器非因果）
- 音频 VAE：mel 64 bins、16 kHz、latent 8 ch、输入输出 2 声道；vocoder 输出 **24 kHz** 立体声
  （注意：音频 VAE 采样率 16 kHz 与 vocoder 输出 24 kHz 不是同一个数）

**其它组件**（`derived`）

- `connectors`：视频与音频各 2 层连接器，各 30 heads × 128，各 128 个可学习 register；
  `text_proj_in_factor` 49
- `latent_upsampler`：空间 ×2；另有独立的 temporal upscaler checkpoint

**采样**（卡的 diffusers 示例，`official`）

- 阶段一：`num_inference_steps=40`、`guidance_scale=4.0` → **双次前向**；121 帧、24 fps、768×512
- 阶段二：加载 `ltx-2-19b-distilled-lora-384`，`num_inference_steps=3`、`guidance_scale=1.0`，
  用 `STAGE_2_DISTILLED_SIGMA_VALUES`
- 独立蒸馏 checkpoint：`ltx-2-19b-distilled`，**8 步 CFG=1**
- 约束：宽高被 32 整除；帧数 8n+1

**发布精度**（卡的 checkpoint 表，`official`）

bf16 全量 / **fp8** / **nvfp4** 三档官方权重 + 蒸馏档 + 蒸馏 LoRA + 空间/时间上采样器。
这是候选里官方量化覆盖最全的一家。

---

### 4.3 `ltx-2.3` — Lightricks/LTX-2.3

发布 2026-03-04（`derived`）。卡上**没有任何结构数字**，以下全部来自权重头实测（`derived`）。

单文件 `ltx-2.3-22b-dev.safetensors`，46.15 GB，`__metadata__.model_version` = "2.3.0"，
BF16 5657 张量 + F32 290 张量，**总计 23.070B 参数**。

单文件里打包了四个组件：

| 组件 | 参数量 |
|---|---|
| `model.diffusion_model` | 21.005B |
| `text_embedding_projection` | 1.156B |
| `vae` | 0.726B |
| `vocoder` | 0.129B |
| `audio_vae` | 0.053B |

`diffusion_model` 内部：

| 子块 | 参数量 |
|---|---|
| `transformer_blocks`（48 块，hidden 4096，音频支 2048） | 18.561B |
| `video_embeddings_connector` | 1.613B |
| `audio_embeddings_connector` | 0.404B |
| `adaln_single` 及 6 个同族 adaln 模块 | 合计 ≈ 0.31B |
| `patchify_proj` | 0.001B |

**这是本轮最有说服力的一条字段论据**：LTX-2（19b）→ LTX-2.3（22b）的 3B 增量，
主干 48 层 4096 一点没变，涨的全在 connector 与文本投影。如果 schema 只有一个 `params.total`，
这个代际差分在页面上会显示成「主干变大了」，而事实相反。

其它实测细节：
- 注意力里有 `to_gate_logits [32, 4096]` → **逐头门控**（32 个头各一个 gate logit）
- 音视频跨模态有独立的 `scale_shift_table_a2v_ca_audio [5, 2048]` / `_video [5, 4096]`
  与 `av_ca_a2v_gate_adaln_single` / `av_ca_v2a_gate_adaln_single`，即音→视、视→音两个方向各一套门控

蒸馏与配套（卡的 checkpoint 表，`official`）：
`ltx-2.3-22b-distilled` 与 `-distilled-1.1`，**8 步 CFG=1**；`distilled-lora-384`（7.61 GB）；
空间上采样器 ×2 / ×1.5，时间上采样器 ×2。
官方量化档 `Lightricks/LTX-2.3-fp8`、`Lightricks/LTX-2.3-nvfp4`（独立仓，`derived`）。

未确认：2.3 的 VAE / 文本编码器是否与 2.0 完全一致（单文件里 vae 是 0.726B，与 LTX-2 的 VAE 未逐张量比对）。

---

### 4.4 `wan2.2-t2v-a14b` 与 `wan2.2-ti2v-5b` — Wan-AI

发布 2025-07-18 / 2025-07-24（`derived`）。License **Apache-2.0**（候选里唯一真正宽松的）。
技术报告 arXiv 2503.20314（`official`）。

**共同点**

- 文本编码器：**umT5-XXL**，`text_len` 512，bf16（厂商 `wan/configs/shared_config.py`，`judgement`
  —— 这是参考实现而不是卡，但与卡的下载清单一致）
- 并行：FSDP + **DeepSpeed Ulysses**（`--ulysses_size 8`），卡上写单卡 720P 需 ≥80 GB 显存（`official`）
- 权重落盘 **全部 F32**（权重头实测，`derived`），而厂商 `shared_config.param_dtype = torch.bfloat16`
  —— 即**发布精度与推理精度不是一个值**

**`wan2.2-t2v-a14b`**

| 项 | 值 | 来源 |
|---|---|---|
| 架构 | 单流 DiT + cross-attn 文本注入；**两套专家按时间步切换** | config + 卡 `official` |
| 每套专家 | dim 5120、40 层、40 heads（×128）、ffn 13824、in/out 16ch、patch (1,2,2) | `high_noise_model/config.json` `derived` |
| 每套权重 | 57,153,966,336 B，F32 → 14.29B。两套合计 ≈ 28.6B | 权重头 `derived` |
| 切换点 | `boundary` 0.875 | 厂商 config `judgement` |
| VAE | Wan2.1-VAE，`vae_stride` (4, 8, 8)，`z_dim` 16，`temperal_downsample [false,true,true]` | 厂商 config + diffusers config `derived` |
| 采样 | 40 步、`sample_shift` 12.0、**`sample_guide_scale` (3.0 低噪, 4.0 高噪)** → 双次前向 | 厂商 config `judgement` |
| 分辨率 | 480P 与 720P，5 s | 卡 `official` |
| 卡的说法 | MoE「在保持相同计算成本的前提下扩大总容量」 | 卡 `official` |

> 注意这条：**计算成本不变但权重存储翻倍**。对 LLM 的 MoE，「总参数 vs 激活参数」描述的是同一层里选 k 个专家；
> 这里两套专家是**按去噪时间步整体换入换出**的，没有 router、没有 top-k、没有 all-to-all。
> 套进 `arch.moe.routed_experts/top_k/routing` 会造出一个不存在的事实（见 §5.4）。

**`wan2.2-ti2v-5b`**

| 项 | 值 | 来源 |
|---|---|---|
| 架构 | dim 3072、30 层、24 heads、ffn 14336、**in/out 48ch**、patch (1,2,2) | config `derived` |
| 权重 | 19,999,150,848 B，F32 → 5.0B | 权重头 `derived` |
| VAE | **Wan2.2-VAE**，`vae_stride` (4, 16, 16)；卡写「压缩率 16×16×4」 | 厂商 config `judgement` + 卡 `official` |
| 采样 | 50 步、`sample_shift` 5.0、**CFG 5.0**、121 帧、24 fps | 厂商 config `judgement` |
| 规格 | 720P@24fps，T2V 与 I2V 统一；卡称可在 4090 这类消费卡上跑 | 卡 `official` |

**`wan2.1-t2v-14b`（第二波基线）**：dim 5120 / 40 层 / 40 heads / ffn 13824 / in 16ch / patch (1,2,2)，
VAE 4×8×8 / 16ch，umT5-XXL（text_dim 4096），权重 57.15 GB F32 → 14.29B（全部 `derived`）。
共享默认：`frame_num` 81、`sample_fps` 16。

---

### 4.5 `hunyuanvideo-1.5` 与 `hunyuanvideo-13b` — tencent

**`hunyuanvideo-1.5`**，发布 2025-11-20（卡的 News，`official`），arXiv 2511.18870。

| 项 | 值 | 来源 |
|---|---|---|
| 参数量 | **8.3B**；权重实测 8.327B，1793 张量 **全 F32** | 卡 `official` + 权重头 `derived` |
| 架构 | `mm_double_blocks_depth` **54**、`mm_single_blocks_depth` **0** → 纯双流 MMDiT，无单流段 | config `derived` |
| hidden / heads | 2048 / 16（head_dim 128）、`mlp_width_ratio` 4 | config `derived` |
| patch | **[1, 1, 1]** —— VAE 已经压到 16×/4×，DiT 不再 patchify | config `derived` |
| latent 通道 | 32 | config `derived` |
| VAE | `ffactor_spatial` 16、`ffactor_temporal` 4、`latent_channels` 32、`AutoencoderKLConv3D`，5.04 GB | config `derived` + 卡 `official` |
| 文本编码器 | MLLM + byT5 字形编码器 + SigLIP 视觉编码器（三件套） | 卡 + 厂商文档 `official` |
| ── MLLM | 官方 HunyuanMLLM **未发布**；文档推荐 `Qwen/Qwen2.5-VL-7B-Instruct` 替代；config 的 `text_states_dim` 3584 与之一致 | 厂商文档 `official` |
| ── byT5 | `google/byt5-small` + Glyph-SDXL-v2（`glyph_byT5_v2: true`，`byt5_max_length` 256） | 厂商文档 + config `official` |
| ── SigLIP | 从 **gated** 的 `black-forest-labs/FLUX.1-Redux-dev` 取；`vision_states_dim` 1152、`vision_num_semantic_tokens` 729 | 厂商文档 + config `official` |
| 稀疏注意力 | **SSTA**（Selective and Sliding Tile Attention）。`720p_i2v_distilled_sparse` 的 config：`attn_mode` flex-block-attn、`attn_sparse_type` ssta、`tile_size` [6,8,8]、`win_size` [[3,3,3]]、`ssta_topk` 64、`ssta_lambda` 0.7、`win_ratio` 10 | config `derived` |
| 稀疏收益 | 卡称 10 s 720p 合成端到端相对 FlashAttention-3 **1.87×** | 卡 `official`（基线配置未公开，不能换算成绝对时延） |
| 采样 | 见下表，卡上有完整的「Optimal Inference Configurations」 | 卡 `official` |
| checkpoint 数量 | transformer 目录下 **11 个**全量 checkpoint，各约 33.3 GB | 仓 blob 清单 `derived` |
| 显存 | 最低 14 GB（开 offload） | 卡 `official` |
| 加速 | FP8 gemm（sgl-kernel 0.3.18）、SageAttention、deepcache / teacache / taylorcache 三种缓存 | 卡 `official` |

采样表（卡原表，`official`）：

| 变体 | CFG | flow shift | 步数 |
|---|---|---|---|
| 480p / 720p T2V、I2V 基座 | 6 | 5 / 9 / 7 | 50 |
| CFG 蒸馏（480p、720p，T2V/I2V） | 1 | 同上 | **仍 50** |
| 480p I2V 步数蒸馏 | 1 | 7 | **8 或 12** |
| 720p T2V/I2V CFG 蒸馏 + 稀疏 | 1 | 9 / 7 | 50 |
| 480→720 SR 步数蒸馏 | 1 | 2 | 6 |
| 720→1080 SR 步数蒸馏 | 1 | 2 | 8 |

卡上加粗强调：**CFG 蒸馏版必须跑 50 步才有正确结果**。
所以「CFG 蒸馏」与「步数蒸馏」是两根**互相独立**的轴 —— 这是 §3.2 一句「蒸馏或步数压缩方案」装不下的。

**`hunyuanvideo-13b`（第二波基线）**，发布 2024-12-01（`derived`）：

- DiT：`mm_double_blocks_depth` 20 + `mm_single_blocks_depth` 40（双流→单流的代表形态），
  24 heads × 128 = 3072 hidden，patch (1,2,2)，in/out 16ch，`text_embed_dim` 4096，`pooled_projection_dim` 768（config `derived`）
- 权重（社区 diffusers 镜像）25,642,025,088 B → 约 12.8B（卡称 13B）（`derived` / `official`）
- VAE：CausalConv3D，卡原话「时间 4×、空间 8×、通道 16」（`official`）
- 文本编码器：**decoder-only MLLM**（卡专门论证为什么不用 T5/CLIP）+ CLIP-L 提供 pooled 向量（`official`）
- 采样：`--infer-steps` 50、**`--embedded-cfg-scale` 6.0**（`guidance_embeds: true`）→ **每步单次前向**、
  `--flow-shift` 7.0、129 帧、推荐 720×1280（`official`）
- 并行：xDiT（`official`）

> 1.5 代相对 13B 代**退回了真 CFG**（双次前向），再用独立的 CFG 蒸馏 checkpoint 把它压回单次。
> 这个来回正好证明 `cfg` 必须是结构化字段而不是一句话。

---

### 4.6 `cogvideox-1.5-5b` — zai-org

发布 2024-11-02（`derived`）。

- DiT：42 层、48 heads × 64 = 3072 hidden、`patch_size` 2、**`patch_size_t` 2**、in/out 16ch、
  `use_rotary_positional_embeddings` true（config `derived`）
- 文本：`text_embed_dim` 4096（T5-XXL）、`max_text_seq_length` 226。
  **文本 token 与视频 token 拼进同一条全注意力序列**（CogVideoX 的 expert-adaLN 设计），
  不是 cross-attn —— 所以注意力序列长度是 n+226，不是 n
- VAE：`AutoencoderKLCogVideoX`，`temporal_compression_ratio` 4，空间 8×，`latent_channels` 16，
  `use_quant_conv` / `use_post_quant_conv` 均 false（config `derived`）
- 默认帧数 81，latent 96×170 → 像素 768×1360（config `derived`）
- 相对 `CogVideoX-5b` 的差分：新增 `patch_size_t` 2（时间维 patchify，token 数减半）、
  `sample_frames` 49→81、分辨率档变大（config 对比 `derived`）

### 4.7 `stepvideo-t2v-30b` — stepfun-ai

发布 2025-02-14（`derived`），License **MIT**，arXiv 2502.10248。

| 项 | 值 | 来源 |
|---|---|---|
| 参数量 | 卡称 30B；权重 58,684,956,800 B → 约 29.3B（BF16 口径） | 卡 `official` + `derived` |
| DiT | 48 层、48 heads × 128 = 6144 hidden、`patch_size` 1、`norm_type` ada_norm_single、**`attention_type` "parallel"**、3D RoPE、QK-Norm | config + 卡 `official` |
| latent 通道 | `in/out_channels` **64** | config `derived` |
| VAE | 卡原话「16×16 空间、8× 时间压缩」。**VAE 不在 HF 仓的 `model_index.json` 里** | 卡 `official` + 仓清单 `derived` |
| 文本编码器 | **两个并用**：自家 Step-LLM（`Step1Model`，48 层 / hidden 6144 / GQA 8 组 / 39,159,201,792 B ≈ 19.6B）+ HunyuanCLIP（1024）。`caption_channels` [6144, 1024] 印证 | config + index `derived` |
| 采样 | `infer_steps` 50、`cfg_scale` **9.0**（双次前向）、`time_shift` 7.0 | 厂商 config `judgement` |
| 规格 | 默认 544×992×204 帧 | 厂商 config `judgement` |
| 部署形态 | **文本编码器、VAE、DiT 拆成三个独立服务**（`--vae_url` / `--caption_url`） | 厂商实现 `judgement` |
| 实测数（卡上给了！） | 544×992×204f：峰值 **77.64 GB**，50 步 flash-attn **743 s**、无 flash-attn 1232 s；544×992×136f：72.48 GB / 408 s / 605 s。测试用 4 卡 | 卡 `official` |

> 最后一行是本轮唯一一个可以直接进公开仓 `performance[]` 的实测数据：它是厂商在卡上公布的，
> `source.type: official` + ref 指向模型卡。其余候选的卡上只有「最低 14 GB（开 offload）」这类定性值。

### 4.8 `sana-video-2.0-5b` — Efficient-Large-Model

发布 2026-08-20（`derived`），License Apache-2.0。卡上给了一张规格表（以下全 `official`）：

| 项 | 值 |
|---|---|
| 参数量 | 4,466,980,960（4.47B） |
| Transformer | 32 层、hidden 2560 |
| 注意力 | **75% gated linear attention + 25% dense softmax anchors**；Attention Residual 每 8 层做一次共享、与时间步无关的聚合 |
| 文本编码器 | `google/gemma-2-2b-it` |
| VAE | **复用 LTX-2.3 的 VAE**，128 latent 通道，stride (8, 32, 32) |
| 采样 | BF16、**CFG 8**（双次前向）、flow shift 12、**50 步**、motion score 20 |
| 输出 | 736×1280、193 帧、24 fps（约 8 s） |
| 其它 | EMA 与 ReFL 适配器已在发布前合入；checkpoint 只含合并后的 state_dict |

另有 `SANA-Video_2.0_5B_720p_4step`（4 步档）。

> 「复用别家的 VAE」是 §6 里「共用组件怎么指过去」最干净的用例：它该指向本仓的 `ltx-2.3` 记录，
> 而不是把 LTX 的 VAE 数字抄一遍。

---

### 4.9 `magi-2-preview` — sand-ai/MAGI-2-preview

发布 2026-08-03（`derived`），License **Apache-2.0**，技术博客 sand.ai/blog/magi-2-preview。
HF 下载量显示为 0、likes 122 —— **按采用度它不算主流**，按结构它是整个候选集里最值钱的一个。
为什么漏掉过、以及这暴露了什么扫描方法问题，见 §7 第 16 条。

**整体**（卡 `official`）

> "Magi-2 Preview is a 114B-parameter unified audio-video generation model that activates just
> 6B parameters per token. Built on MagiMoE…"

两阶段：`magi2_preview` 低分辨率去噪 → `magi2_refiner` 升到 1080p。只支持 10 秒片段。
音频与视频一起生成并 mux 进输出。硬件要求写死 **8 卡 Hopper**，依赖自研的
MagiAttention 与 MagiCompiler。整仓约 307 GB。

**DiT 主干**（全部权重头实测，`derived`）

| 项 | 值 |
|---|---|
| 层数 / hidden | 40 / **3072** |
| 注意力 | `linear_qkv [9216, 3072]` = 3×3072 → MHA；`linear_proj [3072, 3072]` |
| 注意力头数 | **24**（由 `sinks [1, 24]` 与 `linear_g [24, 3072]` 定），head_dim 128 |
| attention sink | **有**，逐头一个（`attention.sinks`，每层都有） |
| 输出门控 | **有**，逐头一路（`attention.linear_g`） |
| 残差 | **mHC**：残差流扩到 **12288 = 4 × 3072**（`mhc_phi_fused_attn [12288, 24]`、各 embedder 出口都是 12288）；每层 12 个 alpha/bias 标量 + `mhc_norm` |
| FFN 层型 | 40 层里 **4 层稠密**（0、1、38、39，`up_gate_proj [49152, 3072]` → 中间维 24576）+ **36 层 MoE** |
| MoE | `moe_mlp.W_up [3072, 256, 1280]`、`W_down [3072, 1280, 256]`、`W_gate` 同形 → **3072 个专家**，每个 256→1280；`gate [3072, 256]` |
| MoE 的粒度 | 专家在 **256 维子空间**里工作，前后各有一个 `split_linear [3072,3072]` / `merge_linear [3072,3072]` —— 即把 3072 维切成 12 片 256 维再路由 |
| 路由 | `moe_mlp.router.expert_bias` + `expert_bias_ema` → **aux-loss-free bias 路由，带 EMA** |
| 共享专家 | `shared_expert_fc1 [2560, 3072]`；另有 `modality_specific_shared_expert_fc1 [7680, 3072]` = 3 × 2560 → **按模态分的共享专家**（视频 / 音频 / 文本各一个） |
| 模态特定部分 | 只在 `pre_adapter`（video 48ch / audio 64ch / text 5120 三个 embedder，出口都是 12288）与 `post_adapter`（video / audio 两路 final linear+norm） |
| 专家参数核对 | 36 层 × 3072 专家 × 3 × 256 × 1280 ≈ **108.7B**，加上其余约 114B —— 与卡的 114B 对得上 |
| 权重落盘 | preview 228,107,858,176 B（1371 张量，BF16 为主、router 与 mHC 系数 F32）；refiner 13,484,611,648 B（281 张量） |

**文本编码器**（`text_encoder/config.json`，`derived`）

**Qwen3.5**（`Qwen3_5ForConditionalGeneration`），hidden 5120、head_dim 256、inter 17408，
`attn_output_gate: true`，**linear:full = 3:1 的混合注意力**（`full_attention_interval: 4`）。
落盘 55.59 GB。随仓发布。

> 这是候选里唯一一个文本编码器**自己就是混合线性注意力**的 —— 本仓 LLM 侧的「线性/混合注意力」
> 专题可以直接接上。

**VAE 与音频**（`derived`）

| 组件 | 内容 |
|---|---|
| 视频 VAE | **`vae/Wan2.2_VAE.pth`** —— 直接用阿里 Wan2.2 的 VAE（4×16×16 / 48 通道）。跨厂商复用 |
| 快速解码 | `turbo_vae/TurboV3-Wan22-TinyShallow_7_7.json`：`AutoencoderKLTurboVAED`，只有解码器、latent 48ch、`patch_size` 2、`decoder_causal: false`。**一个蒸馏出来的浅层解码器**，专为降低 VAE decode 成本 |
| 音频 VAE | **`stable-audio-open-1.0`** —— 第三方开源音频 VAE，4.85 GB |

**对本仓的意义**

1. **它是「token 级 MoE 视频 DiT」的唯一样本**，正好和 Wan2.2 的「时间步级专家切换」构成对照。
   没有它，§5.4 里「Wan2.2 的 MoE 不是 LLM 那个 MoE」这句话只有否定没有肯定。
2. **attention sink + 逐头输出门 + mHC 残差 + aux-loss-free bias 路由 + 堆叠专家** ——
   这五件本仓都已经在 LLM 侧建了字段。MAGI-2 证明这些字段在生成类上要**保留而不是砍掉**。
3. **turbo_vae 是一根新轴**：VAE decode 可以单独蒸馏加速。其它候选都没有，§5.2 第 12 条要加这一项。
4. 它同时是**第 1 步脚本改造最狠的测试用例**：现有 `subsystems.py` 会把它 36 层的专家堆栈
   整个算成稠密 FFN、把 mHC 系数算成注意力、把 Megatron 命名的注意力投影算成 other
   （实跑验证见 [video-gen-step1-checklist.md](video-gen-step1-checklist.md) §A2）。

**压力测试该选谁**：`minimax-h3` 和 `magi-2-preview`各压一头 ——
H3 压的是**组件与部署口径**（四个大模型、两套 DiT、未开源模块、可预计算的 AdaLN）；
MAGI-2 压的是**推导与归类**（MoE 账、mHC、sink、跨厂商复用的 VAE）。
建议先录 MAGI-2：它暴露的是脚本缺陷，发现得越早越好；H3 暴露的是 schema 字段缺口，
在第 1 步定 schema 时已经按 §5 覆盖了。

---

## 5. 字段需求清单

### 5.1 §3.2 已列、且候选之间确有差异，值得进 schema

§3.2 的轮廓是：*主干结构（DiT/MMDiT 等）、参数量、文本编码器、VAE、默认采样步数、
支持的分辨率/帧数/时长、蒸馏或步数压缩方案、单步 FLOPs、单步时延、端到端耗时、显存峰值、
平台（Diffusers、xDiT 等 + 昇腾适配状态）。*

现有 schema 里已有一个 `generation` 块（`backbone` / `text_encoder` / `vae` / `vae_downsample` /
`default_steps` / `distilled_steps` / `resolutions` / `max_frames` / `max_duration_s`，9 个字段全是标量）。
它是个占位，不是空白 —— 但九个标量装不下下面任何一条。

| §3.2 字段 | 候选间差异 | 要改成什么 |
|---|---|---|
| **主干结构** | 单流 + cross-attn（Wan）/ 20 双流 + 40 单流（HunyuanVideo）/ 54 双流 + 0 单流（HV1.5）/ 单流 packed 多模态序列（H3）/ 单流 + 独立音频分支 + 双向跨模态门控（LTX）/ 文本视频拼同一序列 + expert-adaLN（CogVideoX）/ `attention_type: parallel`（StepVideo）/ 线性-稠密混合（SANA） | 一个字符串不够。至少要 `{stream_design, double_blocks, single_blocks, layers, hidden, heads, head_dim, ffn, text_conditioning}`，其中 `text_conditioning` ∈ {cross_attn, in_sequence, dual_stream, adaln} |
| **参数量** | LTX-2.3 的 22B 里主干只有 18.56B；H3 的 33B 里 13B 是推理可免的 AdaLN 分支；H3 Ref2VA 要驻留两套 DiT；Wan2.2 要驻留两套专家 | 复用 `params.total` + `breakdown` + `excludes` + `counting_note`，但 breakdown 的 key 要换成生成类子系统（§5.5），并新增**「一次部署实际驻留几套主干」**这一维 |
| **文本编码器** | umT5-XXL / LLaVA-Llama3+CLIP / Qwen2.5-VL-7B（替代品）+byT5+SigLIP / Qwen3-VL-32B 取第 50 层 / Gemma-3-12B FP32 / Step-LLM 19.6B + HunyuanCLIP / Gemma-2-2B | 必须是**数组**（多编码器并用是常态），每项 `{role, name, hf_repo, params, hidden, shipped_with_weights, gated, release_dtype, hidden_layer_taken, substitutable, note}` |
| **VAE** | 8×8×4/16ch → 16×16×4/{48,32,24}ch → 32×32×8/128ch → 16×16×8/64ch；H3 的解码器是 36 层 ViT 且 VAE 本身 5.2B | 结构化：`{spatial_compression, temporal_compression, latent_channels, causal, params, decoder_arch, hf_repo, ref_model_id}` |
| **默认采样步数** | 40 / 50 / 8 / 12 / 3 / 6；两阶段流水线的步数是两段相加 | 要和 CFG 分开，且要能表达多阶段，见 §5.2 第 1、6 条 |
| **分辨率 / 帧数 / 时长** | Wan 480P/720P@5s；H3 短边 768、4–15s、24fps；HV1.5 480p/720p + SR 到 1080p；LTX 宽高需被 32 整除、帧数 8n+1；StepVideo 544×992×204f；SANA 736×1280×193f | 要是**档位数组 + 默认标记 + 约束规则**：`buckets[]: {width, height, frames, fps, duration_s, default}` 外加 `constraints: {width_divisor, height_divisor, frame_formula}` |
| **蒸馏 / 步数压缩** | HV1.5 同时有 CFG 蒸馏和步数蒸馏，且 CFG 蒸馏版**仍要 50 步**；LTX 的 stage-2 蒸馏是 LoRA | 拆成两个独立字段，见 §5.2 第 1 条 |
| **平台** | Diffusers / ComfyUI / SGLang(diffusion) / vLLM-Omni / TensorRT-LLM visual-gen / xDiT / LightX2V / Wan2GP；并行维是 Ulysses-SP、USP、CFG-parallel、VAE tiling、group offloading | `platforms[]` 结构原样复用，只扩 `engine` 的取值；`parallel` 已是自由字符串，够用 |
| **单步 FLOPs** | 可推 | 不手写，由 `derive.py` 生成到 `derived`，见 §5.2 第 13 条 |
| **单步时延 / 端到端耗时 / 显存峰值** | — | 见 §5.3：公开仓基本填不了 |

### 5.2 §3.2 没写、但从候选里摸出来必须有的

1. **`cfg: {scale, distilled, forwards_per_step}`** —— 最重要的一条。
   `forwards_per_step` 在候选里取 1 或 2，直接给整个端到端计算量乘 2。
   漏掉它，HunyuanVideo-13B（嵌入式引导，1 次）和 HunyuanVideo-1.5（真 CFG 6，2 次）
   会被算成同一个口径。而且 **CFG 蒸馏与步数蒸馏必须是两个字段** ——
   HunyuanVideo-1.5 的 CFG 蒸馏版把前向次数从 2 砍到 1 但步数仍是 50，卡上为此专门加了一句警告。

2. **`patch_size: [t, h, w]`**（DiT 侧）—— §3.2 只提了 VAE。不写 patch 就算不出 token 数。
   取值在候选里是 (1,2,2) / (1,1,1) / (2,2,2) / 1，不是常数。
   **且必须和 VAE 压缩率分开存**，由 derive 相乘得到有效压缩率（§4.0 的表就是论据）。

3. **`latent_channels`** —— 16 / 24 / 32 / 48 / 64 / 128，跨越 8 倍。它决定 in/out 投影规模
   与每个 latent token 的字节数。

4. **`attention: {form, ...}`** —— 取值：全 3D 稠密 / 块稀疏（SSTA：tile 6×8×8、win 3×3×3、topk 64）/
   线性混合（SANA 75:25）/ parallel（StepVideo）/ 逐头门控（LTX 的 `to_gate_logits`）。
   LLM 侧的 `arch.attention.sparse` 与 `arch.attention.hybrid` 结构可以复用，
   但语义要扩到**时空块**（tile 是三维的，win 是三维的）。

5. **`text_conditioning`**（条件注入方式）—— cross-attn（Wan）vs 拼进主序列（CogVideoX 的 226 个文本 token、
   H3 的 packed 多模态序列）vs 双流 MMDiT（HunyuanVideo）。它决定注意力的序列长度到底是 n 还是 n+m，
   而注意力是 O(n²)，所以这个布尔性质的字段会进平方项。

6. **`pipeline_stages[]`** —— 端到端不是一段：
   H3 = Context-IR（未开源）+ Base + Regenerate-2K（未开源）；
   HunyuanVideo-1.5 = DiT + 720p SR（6 步）+ 1080p SR（8 步）；
   LTX-2 = stage1（40 步）+ latent upsampler + stage2（3 步）。
   §3.2 的「端到端耗时」隐含了单段假设。每段要能各自带 `{model, steps, cfg, resolution, open_source}`。

7. **`checkpoint_variants[]`** —— HunyuanVideo-1.5 的 11 个各 33 GB 全量 checkpoint、
   Wan2.2 的高/低噪两套、H3 的 FL2VA/Ref2VA + `transformer_ref`。
   **权重存储要按「一次部署实际驻留几套」算，不是按单个 checkpoint 的文件大小**。
   这和 CLAUDE.md 的「参数量有三套口径」是同一类问题，生成类多了一根「变体」轴。

8. **`precision.inference_default`** —— Wan 全系与 HunyuanVideo-1.5 **落盘 FP32**（权重头实测），
   而厂商自己的脚本用 BF16 跑。LLM 侧没有这个现象（都是 BF16/FP8 落盘）。
   现有的 `precision.weights_release` 只能表达一半，照它算权重存储会高 2 倍。

9. **`audio: {in_channels, vae, sample_rate_in, sample_rate_out, latent_hz, channels}`** ——
   H3（DiT 内 32 通道音频入口 + 32 kHz / 40 Hz latent / 立体声双通道各跑一遍）与
   LTX-2（DiT 内 128 通道 + mel VAE 16 kHz + vocoder 输出 24 kHz）。
   **音频 token 和视频 token 拼在同一条注意力序列里**，所以序列长度要分两项记。
   注意 LTX 的输入 16 kHz 与输出 24 kHz 不是同一个数，单字段装不下。

10. **组件的可得性三连：`shipped_with_weights` / `gated` / `substitutable`** ——
    这是**能不能把模型跑起来**的硬门槛，不是元数据：
    HunyuanVideo-1.5 要凑 MLLM（官方未发布、用 Qwen2.5-VL-7B 替代）+ byT5 + SigLIP（从 gated 仓取）；
    H3 把 33B 的 Qwen3-VL-32B 整个随仓发；StepVideo 的 VAE 不在 HF 仓里。
    顶层的 `hf.gated` 已经有了，但那是**本仓**的 gate，不是依赖组件的。

11. **`parallelism`** —— 生成类的并行维是 Ulysses 序列并行 / USP / CFG 并行 / VAE tiling /
    group offloading，不是 TP/EP。`platforms[].parallel` 是自由字符串可以装，
    但如果要做平台矩阵的筛选，需要把这几个维度结构化。

12. **VAE 自身的参数量与解码形态** —— H3 的 VAE 是 5.2B 且解码器是 36 层 Transformer；
    LTX-2.3 的 VAE 是 0.73B 卷积。**VAE decode 在 H3 上不是边角料，是一次独立的大计算**。
    §3.2 把 VAE 当成一个压缩率标量，漏掉了这一整块。

13. **推导量（进 `derived`，由 `derive.py` 算，不手写）**：
    - `latent_tokens = (F / t_vae / t_patch) × (H / s_vae / h_patch) × (W / s_vae / w_patch)`
    - `tokens_total = latent_tokens + audio_tokens + text_tokens`（取决于 `text_conditioning`）
    - `flops_per_step_linear ∝ tokens_total × active_params`
    - `flops_per_step_attn ∝ tokens_total²`（稀疏时按 SSTA 的 tile/topk 折算）
    - `flops_end_to_end = Σ_stage(steps × forwards_per_step × flops_per_step) + 文本编码一次 + VAE decode 一次`
    - `weights_resident_bytes`：按 §5.2 第 7 条的变体口径算，不是单文件大小

14. **`license_restriction`** —— MiniMax H3 的许可要求美/欧/英/韩用户另行申请；
    HunyuanVideo-1.5 的卡有 `extra_gated_eu_disallowed: true`。
    这不是法务洁癖：它决定一个模型能不能进某个部署区域，属于 infra 可行性。
    LLM 侧现有的 `license` 是个字符串，够不够要在第 1 步定。

### 5.3 公开仓根本填不了的字段

按 CLAUDE.md，公开仓 CI 拒 `source.type: self` / `basis: measured` / `visibility: internal`。
下列字段在公开站上**注定是空的**，应当在页面模板上显式写明「本仓不收实测性能」，
而不是留一个空表格让人以为只是没查：

| 字段 | 为什么填不了 | 例外 |
|---|---|---|
| 单步时延（ms/step） | 实测量，没有权重能对账 | 无 |
| 端到端耗时 | 实测量；且多阶段流水线里有未开源的段（H3） | **StepVideo 的卡上给了**：50 步 743 s / 1232 s（4 卡），`official` + ref |
| 显存峰值 | 实测量，强依赖 offload / tiling / 并行配置 | **StepVideo 卡上给了 77.64 GB / 72.48 GB**；HunyuanVideo-1.5 只给了定性的「最低 14 GB（开 offload）」 |
| 吞吐（videos/hour）、并行扩展性 | 实测量 | 无 |
| 昇腾侧算子覆盖、融合建议、Cube/Vector 配比 | 进 `data-private/` | 无 |
| 不同 VAE tiling 策略的显存-时延权衡 | 实测量 | 无 |
| 质量指标（VBench 等） | 不是 infra 维度，各家口径不同 | 建议整体不收 |

还有一类**看起来能填其实不能**的：厂商宣称的相对加速比。
HunyuanVideo-1.5 的「1.87× vs FlashAttention-3」、步数蒸馏「4090 上提速 75%」、
Wan2.2 MoE「保持相同计算成本」—— 这些可以作为 `official` 文本进 `infra_signals` 或
`bottlenecks.quantified`，但**不能换算成绝对时延**，因为基线配置没公开。
这正是 model-audit 规则 8「免责声明只免掉它点名的那件事」的生成类版本。

**结论**：生成类记录的「性能」区在 12 份里大概只有 1 份（stepvideo）非空。
这不是数据没收集，是口径决定的，要在站点上说清楚。

### 5.4 LLM 侧可以直接复用的结构

**原样复用，一行不改：**

- `evidence`（点分路径 + 前缀继承 + `validate.py --list-judgements`）—— 生成类的来源分四档
  （模型卡/技术报告 > 厂商参考实现 > 引擎文档 > 社区转换），和 LLM 侧完全同构
- `sources[]` / `$defs.ref`（kind / url / locator / accessed / note）
- `lineage`（track / predecessor / successor / `changed_from_predecessor[].{axis, text, benefit}`）——
  视频线特别需要 `axis`，建议取值集扩成 `{vae, backbone, attention, text_encoder, sampling, modality, precision}`
- `platforms[]`（engine × hardware × precision × status × since × graph_mode × caveat × ref）
- `reviewed` / `reviewed_elsewhere` / `review[]` / `open_questions[]` —— 三道闸口的机制不变
- `infra_signals[]`（text / evidence / horizon / axis）
- `bottlenecks[]`（dimension / finding / quantified / fix）
- `hf`（repo / sha / last_modified / pipeline_tag / **gated**）—— `gated` 正好用来标 LTX-2.5
- `visibility` 与公开/内部隔离机制
- `precision.quantization` 与 `precision.per_subsystem` 的**实测机制**（`fetch_dtypes.py` 从权重头读）
- `arch.residual`（type / multiplier / coef_proj）与 `arch.attention.attn_sink`、`arch.attention.output_gate`
  —— 为 LLM 侧 mHC 建的这几个字段在 MAGI-2 上**原样成立**（权重实测：残差流 ×4 扩到 12288、
  逐头 sink `[1, 24]`、逐头输出门 `linear_g [24, 3072]`）。这是「LLM 侧结构正在往视频 DiT 迁移」的直接证据

**结构复用但语义要重新定义：**

- `params.total` / `breakdown` / `excludes` / `counting_note` —— 见 §5.2 第 7 条
- `precision.weights_release` —— 要配一个 `inference_default`，见 §5.2 第 8 条
- `arch.attention.sparse` / `arch.attention.hybrid` —— 扩到时空块，见 §5.2 第 4 条

**明确不复用（硬塞会造出假事实）：**

| 字段 | 为什么 |
|---|---|
| `context`（max / train_native / extension_method） | 没有「上下文长度」这回事。对应概念是「序列长度 = latent token 数」，由分辨率×帧数×压缩率推出，不是一个发布属性 |
| `arch.moe.{routed_experts, top_k, routing, balance, expert_groups}` | **对 Wan2.2 不适用**：它的「MoE」路由轴是**时间步**，没有 router、没有 top-k、没有 all-to-all。填进去会让站点显示一个不存在的 EP 通信量。它该进 `generation.expert_switching: {axis: timestep, boundary: 0.875, experts: 2}`。**但对 MAGI-2 完全适用** —— 那是真的 token 级路由（3072 专家 + 共享专家 + aux-loss-free bias）。所以这两个字段要共存，由 `routing_axis ∈ {token, timestep}` 区分 |
| `arch.decoding`（MTP / speculative） | 非自回归 |
| 整个 KV cache 体系（`kv_sharing` / `kv_component` / `derived.kv_memory` / `cache_basis`） | 非自回归，推理过程中没有跨步复用的 KV。**但不要把这些块从生成类里删掉** —— §2.4 的世界模型线是自回归的，KV cache 会回来 |
| `derived.active_params` | MoE 激活参数的概念不适用；Wan2.2 的「单步只用一套专家」是另一回事 |
| `derived.comm_per_token` | EP all-to-all 不存在；对应的通信是 Ulysses SP 的 all-to-all，口径完全不同，要另算 |
| `arch.retrieval` / `arch.residual` / `arch.vision` / `arch.audio` | `vision` / `audio` 在 LLM 侧描述的是**理解**侧编码器；生成侧的 VAE/vocoder 是另一回事，不要共用这两个 key |

### 5.5 对脚本的改动需求（供第 1 步统一做，本轮不动）

> 现有五条 P1 回溯已经在共享脚本上积了三处待整合改动（`derive.py` 的 bytes_basis、
> `subsystems.py`、`lib/dtypes.py` 的 INT4-in-INT32 打包）。下面是视频线要加的第四批，
> 请在那三处整合完之后一起做。

1. **`scripts/derive.py` 要按 `kind` 分支**。现在它完全不分 kind，三件推导（参数量/权重存储、
   KV、EP 通信）全是 LLM 口径。对 `kind ∈ {image-gen, video-gen}`：
   - `kv_memory()` / `active_params()` / `comm_per_token()` 必须**跳过而不是返回 0**。
     按 model-audit 规则 7（空结果和「真的没有」长得一样），跳过要留显式标记，
     否则站点会把「不适用」显示成「0 字节 KV」
   - 新增生成类推导：latent token 数、单步线性/注意力 FLOPs、端到端 FLOPs、常驻权重字节（§5.2 第 13 条）
   - `from_shapes()` 的参数量统计可以复用，但要能按**组件**（dit / text_encoder / vae / audio_vae /
     vocoder / upsampler / connector）先分一层，再在组件内分子系统

2. **`scripts/lib/subsystems.py` 的归类正则要扩一层**。现在是「张量名 → 子系统」一层映射，
   对生成类会错得很安静：
   - `text_encoder.model.language_model.layers.*.mlp.*` 会落进 `dense_ffn`，
     **和 DiT 自己的 FFN 混成一桶** —— 一个 33B 的外部文本编码器会被算进 DiT 的参数量
   - `vae.encoder.conv_in` 之类没有任何已有正则匹配，全落 `other`
   - LTX-2.3 的 `model.diffusion_model.video_embeddings_connector.*` 会按内部的 attn/ff 名字
     散进 `attention` / `dense_ffn`，1.6B 的 connector 看不出来
   - MiniMax-H3 的 `audio_proj_in` 恰好被 `audio` 正则接住了 —— 这是运气，不是设计

   建议：**先按顶层组件前缀分桶，再在桶内跑现有的子系统正则**。
   并在 `data/schema/subsystems.json` 增加生成类子系统：
   `dit_block` / `adaln` / `connector` / `patchify` / `vae_encoder` / `vae_decoder` /
   `audio_vae` / `vocoder` / `upsampler` / `text_encoder`。
   注意 `adaln` 要单列：H3 的 33B 里 13B 在这里，且卡说推理可预计算。

3. **`data/schema/model.schema.json`**：
   - `family` 是**封闭枚举**（deepseek/glm/minimax/kimi/qwen/stepfun/other），必须加 §6 的新值
   - `generation` 块按 §5.1 / §5.2 重建，`additionalProperties: false` 意味着现在填不进任何新字段
   - `$defs.dtype` 已有 FP32，够用
   - `platforms[].engine` 目前是自由字符串（好事），但 `status` 的枚举要确认够用

4. **`scripts/validate.py`**：要能对 `kind ∈ {image-gen, video-gen}` 跳过 LLM 专属的必填检查，
   并对生成类加新的必填（至少 `generation.vae.{spatial_compression, temporal_compression, latent_channels}`
   与 `generation.sampling.cfg.forwards_per_step` —— 这两个缺了就算不出任何负载）

5. **`scripts/fetch_hf.py` / `scaffold.py`**：生成类的权重不在一个 `config.json` 里，
   而是散在 `transformer/` `vae/` `text_encoder/` 等子目录，还有 LTX-2.3 那种单文件打包。
   取数脚本要能按子目录分别拉 config 与 index。

6. **`site/`**：生成类详情页的「性能」区按 §5.3 默认留空并显式说明；
   `data.ts` 的 `diff: 'same-basis'` 守卫要认识「VAE 压缩率 × patch」这个新的口径对。

---

## 6. family 与命名方案

### 6.1 family 切分

按已定的「生成类单开 family」，不挂到任何现有语言模型 family 下。
理由在 CogVideoX 上最明显：它和 GLM 是两条毫不相干的技术路线，混进 `glm` 的时间线
只会让「GLM-4 → GLM-5.3」这条主线多出两个无关节点。

需要给 `model.schema.json` 的 `family` 枚举新增：

```
wan, hunyuan-video, minimax-video, ltx, cogvideo, stepvideo, sana-video, magi
```

命名规则（与现有 `deepseek` / `qwen` / `glm` / `minimax` / `kimi` / `stepfun` 对齐）：
全小写、连字符分词、**用产品线名而不是厂商名**。
`minimax-video`、`hunyuan-video` 带后缀是为了和同厂的语言模型 family（`minimax`、将来可能的 `hunyuan`）
区分开；`wan` / `ltx` / `cogvideo` / `stepvideo` / `sana-video` 本身就不与任何语言 family 撞名。

> `minimax-video` 这个名字有点别扭，因为 MiniMax 自己把 H3 叫「通用全模态生成系统」而不是视频模型。
> 备选是 `minimax-h3`，但那是模型名不是产品线名，下一代就得改。建议用 `minimax-video` 并在
> family 描述里写清楚它涵盖音视频联合生成。

### 6.2 model_id 命名

现有记录是 `deepseek-v41-flash` / `qwen3.8-2.4t-a95b` / `glm-5.3` / `minimax-m3` / `kimi-k3` /
`step-5-preview` —— 全小写，允许点号，连字符分词，带规模后缀时放最后。沿用同一套：

| family | model_id |
|---|---|
| `wan` | `wan2.1-t2v-14b`、`wan2.2-t2v-a14b`、`wan2.2-ti2v-5b` |
| `hunyuan-video` | `hunyuanvideo-13b`、`hunyuanvideo-1.5` |
| `minimax-video` | `minimax-h3` |
| `ltx` | `ltx-2`、`ltx-2.3`（将来 `ltx-2.5`） |
| `cogvideo` | `cogvideox-1.5-5b` |
| `stepvideo` | `stepvideo-t2v-30b` |
| `sana-video` | `sana-video-2.0-5b` |
| `magi` | `magi-2-preview` |

两条约定：
- 任务后缀（`t2v` / `ti2v` / `i2v`）**只在同代有多个任务档且结构不同时才写进 id**。
  Wan2.2 需要（T2V-A14B 与 TI2V-5B 是两套 VAE），HunyuanVideo-1.5 不需要（11 个 checkpoint 一份记录）。
- MoE 档位沿用厂商写法 `a14b`（激活 14B），与 `qwen3.8-2.4t-a95b` 的 `a95b` 同构。
  但 Wan2.2 的 A14B **不是** MoE 激活参数的意思（见 §5.4），要在 `counting_note` 里写明这个撞名。

### 6.3 共用组件怎么指过去

同一厂商的语言模型和视频模型是两个 family，但组件会跨家族共用。建议在 `generation` 块里
每个组件都带一组可选的引用字段：

```
generation:
  text_encoders:
    - role: primary
      name: Qwen3-VL-32B
      ref_external: { hf_repo: Qwen/Qwen3-VL-32B }   # 本仓没有这条记录
      hidden_layer_taken: 50
      shipped_with_weights: true
  vae:
    name: LTX-2.3 VAE
    ref_model_id: ltx-2.3                             # 本仓已有记录，不抄数字
```

- **`ref_model_id`**：指向本仓已有的另一份记录。现成用例有两个，而且都是**跨厂商**的：
  `sana-video-2.0-5b` 的 VAE → `ltx-2.3`，`magi-2-preview` 的 VAE → `wan2.2-ti2v-5b`（它直接用 `Wan2.2_VAE.pth`）。
  这正是 CLAUDE.md「数字只出现在一处」的要求：SANA 的记录不应该再抄一遍 LTX 的 128 通道和 8×32×32。
- **`ref_external`**：本仓没录的外部模型（Qwen3-VL-32B、Gemma-3-12B、umT5-XXL、T5-XXL、
  SigLIP、byT5、Step-LLM）。只存 hf_repo + 参数量，不展开结构。

关于「文本编码器就是自家 LLM」这个情形：**本轮候选里其实一例都没有**。
H3 用的是 Qwen3-VL-32B（不是 MiniMax 自家的 M3），LTX 用 Gemma-3，SANA 用 Gemma-2。
最接近的是 StepVideo 用自家 Step-LLM 19.6B，但本仓只录了 Step-5 这一代，对不上。
所以这个字段现在填不满 —— **建议字段先建、一律用 `ref_external` 兜底**，
等出现真的跨家族复用（比如将来某家把自己的旗舰 LLM 当文本编码器）再用 `ref_model_id`。

### 6.4 tracking 名册草案

下面是建议追加到 `data/tracking.yaml` 的内容。**本轮不写进去**，只在这里留草案；
`policy` 一行是视频线专用的，与现有那条（「每个家族追最新旗舰」）并列而不是替换。

```yaml
# ↓ 建议追加的部分，Gate A 草案，尚未写入 data/tracking.yaml

video_policy: >-
  每个 family 追最新旗舰，外加一个差分基线；
  VAE 或注意力形态变了的同代档位单独录；
  纯分辨率/时长/任务（i2v、s2v、animate）变体不录，进记录内的变体表。

video_families:
  - family: minimax-video
    flagship: MiniMax-H3
    hf_repo: MiniMaxAI/MiniMax-H3
    released: 2026-07-28
    model_id: minimax-h3
    status: planned
    wave: 1
    note: 压力测试首录。音视频联合；文本编码器是随仓发布的 Qwen3-VL-32B 全权重；
          Ref2VA 侧带第二套 33B transformer_ref；Context-IR 与 Regenerate-2K 未开源

  - family: ltx
    flagship: LTX-2.5
    hf_repo: Lightricks/LTX-2.5
    released: 2026-07-23
    model_id: null
    status: blocked-gated
    note: 仓 gated，模型卡与 config 均取不到，不录。family 当前实际最新记录是 ltx-2.3
    members:
      - model_id: ltx-2
        hf_repo: Lightricks/LTX-2
        released: 2026-01-03
        status: planned
        wave: 1
      - model_id: ltx-2.3
        hf_repo: Lightricks/LTX-2.3
        released: 2026-03-04
        status: planned
        wave: 1
        note: 卡上无结构数字，全部靠权重头实测；单文件打包 DiT+VAE+audio VAE+vocoder

  - family: wan
    flagship: Wan2.2-T2V-A14B
    hf_repo: Wan-AI/Wan2.2-T2V-A14B
    released: 2025-07-24
    model_id: wan2.2-t2v-a14b
    status: planned
    wave: 1
    members:
      - model_id: wan2.2-ti2v-5b
        hf_repo: Wan-AI/Wan2.2-TI2V-5B
        released: 2025-07-18
        status: planned
        wave: 1
        note: 同代但换了 VAE（4×16×16/48ch），按「换 VAE 就是另一回事」单独录
      - model_id: wan2.1-t2v-14b
        hf_repo: Wan-AI/Wan2.1-T2V-14B
        released: 2025-02-25
        status: planned
        wave: 2
        note: 差分基线

  - family: hunyuan-video
    flagship: HunyuanVideo-1.5
    hf_repo: tencent/HunyuanVideo-1.5
    released: 2025-11-20
    model_id: hunyuanvideo-1.5
    status: planned
    wave: 1
    note: 11 个全量 checkpoint 进同一份记录；SSTA 稀疏权重只有 720p_i2v_distilled_sparse 一个
    members:
      - model_id: hunyuanvideo-13b
        hf_repo: tencent/HunyuanVideo
        released: 2024-12-01
        status: planned
        wave: 2
        note: 嵌入式引导（每步单次前向），与 1.5 代的真 CFG 形成对照

  - family: cogvideo
    flagship: CogVideoX1.5-5B
    hf_repo: zai-org/CogVideoX1.5-5B
    released: 2024-11-02
    model_id: cogvideox-1.5-5b
    status: planned
    wave: 2
    note: 2024 年锚点；该线自 2024-11 起无后续，zai-org 之后只发了任务模型

  - family: stepvideo
    flagship: Step-Video-T2V
    hf_repo: stepfun-ai/stepvideo-t2v
    released: 2025-02-14
    model_id: stepvideo-t2v-30b
    status: planned
    wave: 2
    note: 压缩率最激进（16×16×8 / 64ch）；卡上有峰值显存与端到端耗时，是唯一能填公开 performance 的一份

  - family: magi
    flagship: MAGI-2-preview
    hf_repo: sand-ai/MAGI-2-preview
    released: 2026-08-03
    model_id: magi-2-preview
    status: planned
    wave: 1
    note: 唯一的 token 级 MoE 视频 DiT（114B 总参 / 每 token 激活 6B，36 层各 3072 专家）；
          带 attention sink、逐头输出门、mHC 残差（×4）；视频 VAE 直接用 Wan2.2_VAE.pth，
          音频 VAE 用 stable-audio-open-1.0，另带一个蒸馏的 turbo 解码器。
          采用度低（HF 下载量 0、likes 122），入选理由是结构不是影响力

  - family: sana-video
    flagship: SANA-Video 2.0 5B 720p
    hf_repo: Efficient-Large-Model/SANA-Video_2.0_5B_720p
    released: 2026-08-20
    model_id: sana-video-2.0-5b
    status: planned
    wave: 2
    note: 唯一的线性注意力视频 DiT（75% gated linear + 25% dense anchor）；VAE 复用 ltx-2.3。
          若要缩减范围，砍这一条

video_watchlist:
  - name: LTX-2.5
    vendor: Lightricks
    reason: 有公开权重但仓 gated，模型卡与 config 取不到；第三方镜像不足以支撑记录
    action: 申请访问后转正式记录
  - name: Sora 2 / Veo 3.x / Kling / Seedance / Runway Gen-4 / Pika
    vendor: 多家
    reason: 无公开权重，任何数字都无法复算
  - name: H3-Context-IR / H3-Regenerate-2K
    vendor: MiniMax
    reason: MiniMax-H3 系统的两个模块未开源，端到端负载在公开仓算不全
  - name: HunyuanMLLM
    vendor: Tencent
    reason: HunyuanVideo-1.5 的官方文本编码器未发布，社区用 Qwen2.5-VL-7B-Instruct 替代
  - name: 世界模型 / 交互式视频线
    vendor: Hy-Nomas / Matrix-Game-4.0 / AlayaWorld / Cosmos3 / LongLive-2.0 / lingbot-world
    reason: 自回归逐块生成，KV cache 与流式时延回来了，负载模型与本线不同；建议单开一条线
  - name: Diffusion Forcing 长视频
    vendor: SkyReels-V2-DF / LongLive
    reason: 改的是采样调度不是主干，作为路线观察条目
  - name: 训练无关缓存加速
    vendor: deepcache / teacache / taylorcache / MagCache
    reason: 直接改变「步数 × 单步」的有效步数，是生成类特有的一类加速，进专题不进记录
```

---

## 7. 没查到或查不准的部分

这一节不省。下面每一条在第 1 步定 schema 或第 2 步录记录时都要先解决或显式留成 open question。

**取不到证据的**

1. `Lightricks/LTX-2.5` 的全部结构细节 —— 两个仓都 gated。能推的只有「22b 档」与
   新增六个组件目录（来自不受 gate 限制的 HF 文件清单 API）。需要申请访问。
2. HunyuanVideo-1.5 的官方 MLLM（HunyuanMLLM）未发布。记录里「文本编码器」填什么要先定口径：
   填「未发布，社区用 Qwen2.5-VL-7B-Instruct」还是填替代品本身？两种写法下
   `text_encoder.params` 的含义不同。
3. MiniMax-H3 的 **默认采样步数在模型卡上根本没写**。sglang cookbook 说 50，
   但那是引擎文档（优先级最低），只能标 `judgement`。第三方蒸馏 LoRA 做到 8 / 4 步，更不能当官方值。
   录的时候要么留 `unknown`，要么标 `judgement` 并写明来源是引擎文档。

**查到但没验到底的**

4. MiniMax-H3「33B 里约 13B 是 AdaLN 分支、推理不必加载」是卡的原话（`official`），
   但**实际能省多少显存没验**。要按张量名分桶统计 AdaLN 相关张量的 numel 才能给出数。
   在那之前，不要把「33B − 13B = 20B」写成任何一个字段 —— 那是两个口径相减。
5. Wan2.2 的两套专家在实际部署里是否必须同时驻留。`boundary` 只切换一次，理论上可以换入换出；
   厂商脚本用 `--offload_model True` 统一管理，没有单独说明。权重存储该按 28.6B 还是 14.3B 算，未定。
6. HunyuanVideo-1.5 的 SR 模块到底多大。`upsampler/720p_sr_distilled` 只有 0.09 GB，
   而 `transformer/720p_sr_distilled` 是 33.3 GB —— 看起来 SR 实际是整个 DiT 的另一套权重 +
   一个小 upsampler，但没有确认。录的时候要先搞清楚这两个目录的关系。
7. LTX-2.3 的 VAE / 文本编码器是否与 LTX-2 完全一致。2.3 的单文件里 VAE 是 0.726B，
   LTX-2 的 VAE 没有单独统计过，两边未逐张量比对。
8. LTX-2.3 注意力里的 `to_gate_logits [32, 4096]` 是逐头门控 —— 这是从张量形状读出来的
   （`judgement`），没有卡或论文印证具体语义（是输出门控还是注意力选择门）。
   arXiv 2601.03233 没读，只读了模型卡。
9. Wan 与 HunyuanVideo-1.5「落盘 FP32、推理 BF16」已由权重头实测确认，
   但**没有确认厂商是否在别处发过 BF16 版本**。如果有，`weights_release` 要写成数组。
10. StepVideo 的 VAE 具体参数与 latent 通道数。`in_channels: 64` 是 DiT 侧的入口，
    推 VAE latent 是 64 通道是合理推断（`derived`），但 VAE 本体不在 HF 仓里，没有直接证据。

**范围上的空白**

11. **昇腾侧状态只查到存在性**。确认有 Ascend-SACT（AtomGit）的 HunyuanVideo / HunyuanVideo-1.5
    适配条目，以及华为云 ModelArts 的「Wan2.1/Wan2.2/HunyuanVideo/CogVideo 系列基于轻量算力节点
    适配 NPU 推理指导」。但**没有逐个核对**支持哪些模型、到什么精度、什么时间、有没有文档化配方。
    值得注意的一点：这些适配走的是 ModelArts / SACT 的独立配方，**不是 vllm-ascend** ——
    和语言模型线的平台矩阵不是同一套入口。第 1 步定 `platforms[].engine` 枚举时要考虑这个差别。
12. 引擎侧只做了粗扫：确认 TensorRT-LLM 的 visual generation（beta）列了 Wan2.1/2.2、LTX-2、
    MiniMax-H3（其中 H3 的 Ref2VA 未开放、注意力限 SM100/SM103），vLLM-Omni 有 Wan2.2 I2V 配方，
    SGLang 有 diffusion cookbook。**没有逐个核对版本号与支持状态**，平台矩阵要在录记录时重做。
13. 本轮**只扫了 HuggingFace**。只发在 ModelScope（或只发在 GitHub release）的厂商模型可能漏掉。
    `data/tracking.yaml` 的双周采集如果要覆盖视频线，采集源要加 ModelScope。
14. Open-Sora v2 的 DC-AE 压缩率、Mochi 的 AsymmDiT 细节没深挖 —— 因为决定不录。
    如果后来要翻案，这两条要补。
15. 图像生成（`kind: image-gen`）**完全没看**。已定的三条里第 1 条说 schema 按 image-gen + video-gen
    两类一次建齐，但本轮只从视频候选里摸字段。建库时要留意：
    图像侧没有时间维（VAE 只有空间压缩、没有帧数/时长/fps），
    `generation.vae.temporal_compression` 之类字段在 image-gen 上必须允许缺省；
    反过来图像侧有视频侧没有的东西（多尺度/refiner/ControlNet 生态），本轮一条都没调研。
    **建议第 1 步先只落 video-gen 的字段，image-gen 留出扩展位但不猜字段** ——
    凭空设计会建一堆没人填的字段，这正是 Gate A 要避免的。

16. **扫描方法有个缺口，已经漏过一次。** 本轮的候选是按 HF `pipeline_tag` + 下载量/likes 排序扫出来的，
    而 `sand-ai/MAGI-2-preview` 的 HF **下载量显示为 0**（`sand-ai/MAGI-1` 也是 0），
    于是它没有出现在任何一张榜上，是用户追问才补上的。
    下载量为 0 的原因没查清 —— 可能是权重以非标准布局（`.pth`、`.ckpt`）发布导致计数器不统计。
    结论：**下载量不能当唯一的发现手段**，双周采集要同时按 `createdAt` 扫全量 video 类 pipeline_tag，
    并维护一份厂商 org 白名单（sand-ai 之前不在我的名单里）。
    同理可能漏掉的还有：只发 `.pth` / `.ckpt` 的其它仓、以及 §7 第 13 条说的 ModelScope 独发。

17. MAGI-2 的**采样参数全部未知**：默认步数、CFG、是否蒸馏、preview 与 refiner 各跑几步，
    模型卡上一条都没有（卡只讲了 114B/6B、两阶段、10 秒、8 卡 Hopper）。
    这些要去读 sand.ai 的技术博客或 GitHub 的推理代码。本轮没读博客，只读了卡与权重。

18. MAGI-2 的 MoE 口径有三个数没定：top-k 是多少、`split_linear` 把 3072 维切成 12 片 256 维
    之后每片各自路由还是共享路由、「每 token 激活 6B」是按几片算的。
    这三个不定，激活参数与 EP 通信量都算不出来。权重形状只能告诉我们专家总数和每个专家的形状。

19. MAGI-2 的视频 latent 48 通道是从 `pre_adapter.video_embedder.weight [12288, 48]` 读出来的，
    与 Wan2.2-VAE 的 48 通道一致，所以判断它用的就是 `Wan2.2_VAE.pth`（仓里确有这个文件）。
    但**没有核对这个 .pth 与 Wan-AI 官方发布的那份是否逐张量一致** —— 有可能是微调过的。
