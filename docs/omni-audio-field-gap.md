# 音频侧的字段缺口

`arch.audio` 需要哪些字段才能回答 infra 的问题。这是一份**需求说明**，不是建字段的提交 ——
本轮只在 `arch.vision` 下动了手（见那边新增的 `tokens` / `prefill` / `separability`），
音频侧一个字段都没加，因为**仓里现在没有任何一份记录能用来验证新字段**。

先摆事实，再说需求，最后给落地顺序。

## 一、现状：音频维度是空的，但不是「schema 里没有」

`arch.audio` **已经存在**，有五格。这一点和「schema 里一个音频字段都没有」的印象相反，
所以先核清楚，免得重复造一遍：

| 位置 | 字段 | 在用的记录数 |
|---|---|---|
| `arch.audio` | `encoder` / `params` / `tokens_per_second` / `fusion` / `output` | **0 / 47** |
| `generation.audio` | `in_channels` / `vae` / `vae_params` / `vae_hf_repo` / `sample_rate_in` / `sample_rate_out` / `latent_hz` / `channels` / `vocoder` / `note` | **0 / 47** |
| `kind` 枚举 | `omni` 这个值 | **0 / 47**（35 份 `llm` + 12 份 `vlm`） |

也就是说：**格子有一部分，数一个都没有**。缺的不是 schema 的全部，是
(a) 任何一份带音频的记录，(b) `arch.audio` 里 infra 真正要的那几格。

### 这 12 份 `kind: vlm` 里零份带音频

本轮逐份核过，两条独立判据都是零命中：

- **config**：12 份 `facts/<id>/cfg.json` 全文正则搜 `audio|speech|wav|mel|asr|tts|sound`（忽略大小写），
  **0 处命中**。没有 `audio_config`，没有音频相关的顶层键。
- **权重清单**：12 份 `facts/<id>/shapes.json` 合计 914,930 个张量，
  名字里含 `audio|speech|mel|wav` 的 **0 个**。

复现：

```bash
.venv/bin/python - <<'EOF'
import json, re, glob
ids = "deepseek-v41-flash ds-v4-flash-vision-exp kimi-k3 kimi-k2.5 kimi-vl-a3b minimax-m3 qwen3.5-27b qwen3.5-397b-a17b qwen3.8-27b qwen3.8-flash-next step-3 step-5-preview".split()
for i in ids:
    cfg = open(f'facts/{i}/cfg.json').read()
    hits = re.findall(r'audio|speech|wav|mel|asr|tts|sound', cfg, re.I)
    names = json.load(open(f'facts/{i}/shapes.json'))
    an = [n for n in names if re.search(r'audio|speech|mel|wav', n, re.I)]
    print(f'{i:24s} cfg={len(hits)} tensors={len(an)}/{len(names)}')
EOF
```

所以**这份文档不是从数据反推出来的需求**，是前瞻的。写它的理由是：
下一轮录 Omni 的人会面对和本轮视觉侧一模一样的几个口径坑，
而那几个坑在视觉侧已经踩完了（见下面第四节）。

### 仓里唯一的音频分析，结论是「填不出来」

`data/models/stepfun/step-1.yaml` 的 R6 和 `step-3.yaml` 的 R20 / `notes`
是目前仓内关于音频的全部内容，两条都指向同一件事：

- step-1 的模型卡自称「130 billion parameter **multimodal** LLM」，但 795 个张量里
  **没有任何音频编码器、适配器或 projector**；音频以**离散 token** 进 74,752 的词表，
  前端在 `stepfun-ai/Step-Audio-Tokenizer` **另一个仓**。
  结论是 `kind` 保持 `llm`、`arch.audio` 一格都填不出来 ——
  「记 omni 会让人期待一个不存在的编码器」。
- Step-Audio-2-mini / -Base / -Think（`model_type step_audio_2`）文本侧是
  Qwen2.5-7B 规格、vocab 158,720，被判为「名字像阶跃、主干却不是」而没有另起记录。

**这条已有的判断正是下面第二节那个分叉的来源**，不是本文新提的。

## 二、两种集成形态，现有 `arch.audio` 只建模了其中一种

| | **形态 A：连续编码器 + projector** | **形态 B：离散音频 token 进文本词表** |
|---|---|---|
| 主仓里有音频编码器张量吗 | 有 | **没有** |
| 音频怎么进序列 | 编码器输出 → projector → 主干 hidden | codec 把波形编成 token id，和文本共用 embedding 表 |
| 已知例子 | Qwen-Omni / Kimi-Audio 一类（本仓未录） | StepFun（step-1 R6 已查明） |
| `arch.audio.encoder` / `params` | 能填 | **填不出来**，编解码器在另一个仓 |
| 真正要记的是 | 编码器结构、下采样栈、projector | **词表切分**、codec 仓与版本、码本数、帧率 |

现有五格（`encoder` / `params` / `tokens_per_second` / `fusion` / `output`）
是按形态 A 设计的。对形态 B，前两格恒为空，而最该记的词表切分与 codec 仓没有位置。
`tokens_per_second` 更麻烦：它在两种形态下**不是同一个量**——
A 是编码器下采样后的输出帧率，B 是 `codec 帧率 × 码本数`。
同名不同义的字段会让跨形态比较直接出错，而且 `validate.py` 看不出来。

**建议：`arch.audio` 显式记形态**（例如 `integration: continuous-encoder | discrete-codec`），
其余字段按形态分别要求，而不是让一组字段兼顾两种。

## 三、需求：`arch.audio` 要补哪几组

按 PLAN §2「视觉/音频编码器、模态融合方式、prefill 负载、encoder 并行与分离部署」
和 §3.1 第 2 条「编码器结构与参数量、projector、token 压缩方式」对齐，
分四组。**每组都注明形态 A / B 是否适用**，以及为什么现有字段不够。

### 1. 编码器结构与参数量（形态 A）

| 建议字段 | 为什么 |
|---|---|
| `encoder`（已有） | 够用 |
| `params`（已有） | 够用，但要和视觉侧一样**实测自权重清单**、不抄模型卡 |
| `layers` / `hidden` / `num_heads` / `intermediate` | 视觉侧有这四格，音频侧没有。没有它们算不出 encoder FLOPs |
| `downsample_stack` | 音频编码器的下采样通常是**多级卷积 + 可选的池化**，不是视觉那种单一 patch 网格。要能逐级复算（本轮 StepFun 两档的两级跨步卷积就是这个形状，见 `arch.vision.tokens.compression_stage`） |

形态 B 这一组全部留空，改记 `codec_repo` / `codec_rev` / `codebooks` / `vocab_span`。

### 2. 采样率与帧率（两种形态都要）

**不要新造一套命名** —— `generation.audio` 已经把这件事做对了，直接借它的：

| `generation.audio` 已有 | 含义 | 理由（它的 description 原文就说明了） |
|---|---|---|
| `sample_rate_in` / `sample_rate_out` | 输入与输出采样率**分两格** | 两者可以不同（LTX-2 的 VAE 收 16 kHz、vocoder 出 24 kHz），「单字段装不下」 |
| `latent_hz` | latent 的时间分辨率 | 这正是理解侧要的「帧率」 |
| `channels` | 声道数 | 立体声且左右各跑一遍时要在 note 写明 |

理解侧再加两格：

- `window_ms` / `hop_ms`（或等价的 `frame_shift_ms`）—— 前端特征的窗长与帧移。
  **没有它推不出每秒多少帧**，而每秒多少帧是下一组的分母。
- `feature`（如 `log-mel-80`）—— 梅尔通道数会进编码器的输入维，算 FLOPs 要用。

**跨两处对齐的收益**：理解侧和生成侧用同一套采样率/帧率命名，
音视频联合生成那条线和 Omni 理解线的数才能放在一起比。各造一套就只能各说各话。

### 3. 一秒音频产出多少 token（两种形态都要，这是 infra 的基本量）

对应视觉侧刚建的 `arch.vision.tokens`。**单独一格 `tokens_per_second` 不够**，
理由和视觉侧一样：encoder 侧和主干侧的 token 数不是一个数。

| 建议字段 | 对应视觉侧 |
|---|---|
| `reference_duration`（如 `1s` / `30s`） | `reference_resolution` |
| `encoder_tokens_per_second` | `encoder_tokens` |
| `llm_tokens_per_second` | `llm_tokens` |
| `compression_ratio` | 同名 |
| `compression_stage` | 同名 —— 压在 projector 之前还是之后，次序决定 projector 的 FLOPs 落在哪个 token 数上 |
| `derivation` | 同名，逐因子写来源 |

形态 B 的 `llm_tokens_per_second` = `codec 帧率 × 码本数`，
而 `encoder_tokens_per_second` 不适用（没有连续编码器）—— 这又是要显式记形态的理由。

### 4. 融合方式、prefill 负载、可分离性

`fusion`（已有）够记 projector 结构。另外两件事**直接复用本轮在视觉侧建好的两个子树形状**，
不要另设一套：

- `arch.audio.prefill` ← 照 `arch.vision.prefill` 的七格
  （`encoder_linear_gflops` / `projector_linear_gflops` / `backbone_token_budget` /
  `backbone_linear_gflops_per_token` / `audio_as_text_tokens` / `attention_excluded` / `derivation`）。
  视觉侧那个换算问的是「一张图相当于多少个文本 token」，
  音频侧问「一秒音频相当于多少个文本 token」—— 同一个口径，换个物理量。
- `arch.audio.separability` ← 照 `arch.vision.separability` 的五格。
  音频侧还多一个问题：**Omni 有两座编码器**（视觉 + 音频），
  它们是否在同一个分片、能否各自独立拆出来。视觉侧的 `shard_isolation`
  三值（`dedicated` / `partial` / `mixed`）可以直接用，但要按模态各记一份。

`output`（已有，`understand-only` / `tts` / `duplex`）是理解侧没有而音频侧特有的一轴，保留。
**`duplex` 还会带出一个纯 infra 的问题**：双向流式下音频编码器和主干是并发的，
`prefill` 这个口径本身是否还成立。那是录第一份 duplex 记录时要回答的，现在不预设字段。

## 四、口径护栏：视觉侧踩过的坑，音频侧会原样重演

本轮在 12 份视觉记录上抓到四个口径问题。它们都**不是视觉特有的**，
录音频时会以同样的形式出现，所以先写下来：

1. **encoder token 数 ≠ 主干 token 数。** 视觉侧压缩比从 4 到 16 不等，
   两个数差一个数量级都可能。音频侧同理（下采样栈 + 可选的 token 合并），
   合成一格就答不出 prefill。
2. **主干那侧的分母要扣掉 `lm_head`。** `data/derived/<id>.json` 的
   `flops.linear_per_token` 是 `2 × active_params.total`，**含 `lm_head`**——
   那是 decode 口径（每步对全词表做 GEMM），而 prefill 只在最后一个位置算 logits。
   该用 `2 × (active_params.total − active_params.lm_head)`。
   本轮 12 份都用了后者。`lm_head` 在分母里的占比实测 **1.1%–15.0%**：
   主干越小、词表越大，这一项越不能忽略 —— kimi-vl-a3b 是 15.0%
   （A3B 主干 + 16 万词表），kimi-k3 与 qwen3.8-flash-next 只有 1.1%。
   音频侧的 Omni 多半挂在中小主干上，所以这一项要当成必做，不是精修。
3. **跨模型比较必须同「物理量」。** 视觉侧各家默认分辨率不同，
   所以本轮把 12 份全钉在 672×672（StepFun 两档固定 728、**明确标注不可换算**）。
   音频侧对应的是采样率与参考时长：**不同采样率之间不算倍数**，
   那是范畴错误（CLAUDE.md「不同口径之间不算倍数」的同一条）。
4. **「结构上可分」与「有实现支持」是两件事。** 本轮 12 份的
   `separability.weight_prefix` / `shard_isolation` / `cross_prefix_sharing`
   全部实测自 `model.safetensors.index.json` 的 `weight_map`，是硬事实；
   而 `runtime_support`（厂商参考实现 / 部署文档里 encoder 能否独立成服务）
   **12 份一份都没填** —— 没有一手来源就不填，不从「权重能分开」推「实现支持分离部署」。
   音频侧照此办理。

另有一条**本轮没能钉住、音频侧同样会遇到**的：encoder 自注意力的跨度
（整段 / 分窗 / 分块）。12 份里只有 kimi-k2.5 有一手来源（参考实现读过）。
音频的注意力跨度（整段 vs 分块流式）对长音频的 O(n²) 影响更大，
录的时候要当成一个必答项，而不是顺带。

## 五、建议的落地顺序

字段**不要先建**。按这个顺序走，每一步都有记录能验证新字段：

1. **先录一份形态 A 的 Omni**（主仓里带音频编码器张量的），用现有五格能填多少填多少，
   把填不出来的逐条记进 `review` 与 `open_questions`。
   这一步会暴露「哪些格子是真缺的」—— 比照着别家 schema 猜要准。
2. **按那份记录的缺口改 `arch.audio`**，一次只加能立刻填上的格子。
   采样率/帧率那一组直接借 `generation.audio` 的命名。
3. **再录一份形态 B**（StepFun 那条线），这时 `integration` 这一轴的必要性
   会由两份记录的对比自己证明，而不是靠本文论证。
4. **最后补 `prefill` / `separability` 两组**，形状照搬视觉侧，这样两个模态的数可比。

### 不建议现在就建的字段

- 把 `arch.audio` 一次扩到十几格：现在零份记录，建出来的格子没人能验，
  而空格子会被下一轮当成「已经考虑过了」而不再追问
  （`model-audit` 规则 9：注释不是证据，同理空字段不是覆盖）。
- 为 `duplex` 预设流式字段：口径本身还没定（见第三节末）。
- 给 `kind: omni` 改任何校验规则：schema 已经允许这个值，等第一份记录落地再看
  `check_evidence` 与站点渲染是否需要跟着动。

## 六、一句话结论

音频维度的缺口**不在 schema 的存在性上**（`arch.audio` 五格、`generation.audio` 十格都已就位，
`kind: omni` 也允许），**在于零份记录和 infra 那几格**：
每秒多少 token、prefill 负载、两座编码器各自能不能拆。
这三件事的字段形状本轮已经在 `arch.vision` 下做出来并在 12 份记录上验证过，
音频侧照搬即可 —— 前提是先有一份带音频的记录。
