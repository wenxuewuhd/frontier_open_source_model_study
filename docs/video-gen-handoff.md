# 视频生成线 · 交接

> 分支 `p2-video-gate-a`，worktree `/Users/yuan/Work/frontier-llm-tracking/wt-p2-videogen`。
> **未 push。** 基线：已合入 `main` = 37e9404。
> 这份是给接手的人看的状态与判断留痕；选题在 [video-gen-gate-a.md](video-gen-gate-a.md)，
> 管线改造清单在 [video-gen-step1-checklist.md](video-gen-step1-checklist.md)，
> 两份都在本分支上，**不要在这里重复它们的内容**。

---

## 1. 当前状态

### 三份产物

| 文件 | 是什么 |
|---|---|
| `docs/video-gen-gate-a.md` | Gate A 选题。12 份记录 / 8 个 family 的入选清单与逐条依据、不录的逐条依据、观察名单、每个候选的事实表、字段需求清单、family 与命名方案、没查到的部分 |
| `docs/video-gen-step1-checklist.md` | 第 1 步管线改造清单。§0 依赖、§A subsystems、§B schema、§C derive、§D 其它、§E 执行顺序、§F 第一批三份的分工、§G 本线之外的待办、§H 实现中发现的事 |
| `docs/video-gen-handoff.md` | 本文件 |

### 提交

| commit | 内容 |
|---|---|
| `fc254f2` | Gate A 初版（11 份 / 7 family） |
| `1958424` | 补 MAGI-2（→ 12 份 / 8 family），新增第 1 步清单 |
| `9bdcad3` | 修正清单 §0 的依赖结论 |
| `6a5859d` | 定第一批三份；A3 拆成 A3a / A3b |
| `beaa155` | 合 `main`（37e9404）进本分支 |
| `5557c6c` | **第一段管线落地**：schema + derive + subsystems 组件层 |

### ⚠️ 第 1 步的第一段已经做完了，不是「还没开始」

主会话在交接要求里写的是「step 1 等 P1 五条线整合完再开」。
**实际情况是：用户在此之前已经指示开始第一段，代码已落在 `5557c6c` 上。**
两边的信息不同步，接手的人以本分支的 git 状态为准。

已落地（`5557c6c`）：

- `data/schema/model.schema.json` —— B1 family 枚举 +8；B2 `generation` 块从 9 个标量重建成十组；
  B3 `arch.moe.routing_axis` / `precision.inference_default` / `license_restriction`；
  B4 `platforms[].engine` 的生成类说明；三个硬必填进 `allOf`
- `data/schema/components.json` —— 新文件，顶层组件前缀表
- `scripts/lib/subsystems.py` —— **A3a 组件层**：`component()` / `classify2()` / `COMPONENT_ORDER` / `COMPONENT_LABEL` / `SPLIT_INSIDE`
- `scripts/derive.py` —— C1 按 `kind` 分流、C2 `gen_workload()`、C3 参数量组件维

回归（每一项都实跑过）：

- LLM 侧 15 份记录的 `derive.py` **摘要逐字不变**、**JSON 逐字不变**（忽略 `generated_at`）
- `validate.py` 0 错 0 警
- A3a 组件归类 23 个用例全过，含「语言模型张量必须全部落 `backbone`」一组
- 生成类端到端冒烟：schema 校验通过、三个必填的反例各自被拒、`derive.py` 返回 0

未做：**A3b**（`dit` 桶内正则扩展）、`fetch_dtypes.py`、`site/`、以及录任何一份记录。

### 第一批录哪三份（用户已拍板）

`wan2.2-ti2v-5b` → `minimax-h3` → `magi-2-preview`。分工与各自的 open question 见清单 §F。
前两份只依赖已落地的部分，第三份依赖 A3b。

---

## 2. 依赖复核的完整结论（含我修正过的那一版）

### 结论变过一轮，两版都记在这里

**第一版（错的）**：我查到 `git diff main..<branch> -- scripts/lib/` 在五个 p1 分支上全为空，
于是写成「依赖比预想的轻，`subsystems.py` 与 `dtypes.py` 没有待整合项」。

**第二版（现行）**：查到的事实对，推出的结论错。
主会话明令禁止五条 P1 会话直接改共享文件，要求把需求**写进交接笔记报回主会话**统一整合。
所以那两处改动是以**待办**的形式存在的 —— git 里查不到，正是因为流程要求它们不该在 git 里。

> 拿 `git diff` 为空去证明「没有待整合项」，是 model-audit 规则 7 的同一个坑：
> **空结果和「真的没有」在代码里长得一样。**
> 这条值得留着 —— 它和本仓已有的几个错数字是同一类失误。

### 三项待整合的现状

| 待整合项 | 来自哪条线 | 现在在哪 | 状态 |
|---|---|---|---|
| `derive.py` 的 `bytes_basis` / 厂商公布 KV 字节（含 schema 的 `kv_sharing.declared_per_token_bytes` 与 `precision.kv_cache_components.source` 描述改写） | MiniMax 线提出，在 DeepSeek 记录上落地 | **`main`，commit `7e65c70`** | **已在 main**。原在 `p1-glm` 上叫 `bad7cf6`，已 cherry-pick；`p1-glm` 退回自己的 9 个提交（`0793fd1`）并搬进 `wt-p1-glm` |
| `scripts/lib/subsystems.py` 要补语言模型侧缺的子系统划分 | GLM 线（报「必改」） | 只在交接笔记里 | 未整合 |
| `scripts/lib/dtypes.py` 不认 INT4-in-INT32 打包 | Kimi 线 | 只在交接笔记里 | 未整合。`kimi-k2-thinking` / `kimi-k2.5` 各有 2 处 Gate B MISMATCH 卡在这上面没收口 |

核实命令（可重跑）：

```bash
git merge-base --is-ancestor 7e65c70 main && echo "7e65c70 已在 main"
git show main:scripts/derive.py | grep -c vendor_declared     # 3
git diff --stat 352c114..main -- scripts/lib/                 # 空
```

### `subsystems.py` 的三方撞车

这个文件现在有三拨人要改：

1. **GLM 线**（待整合）：补语言模型侧缺的子系统划分
2. **本线 A3a**（已落地）：加组件层 —— 新函数 + 新文件，**没碰 `SUBSYSTEM` 列表**
3. **本线 A3b**（未做）：往 `SUBSYSTEM` 列表里插条目

`SUBSYSTEM` 是一张**顺序即优先级**的正则表。1 和 3 改的是同一张表的同一个区域，
分别改完再合并几乎必然出错 —— 不是文本冲突，是语义冲突：
两边各自插了条目，合并后的优先级顺序谁也没验过。

---

## 3. A3a / A3b 的拆法 —— 两种意见都在，接手的人自己判断

### 拆法本身

| | 改什么 | 碰不碰 `SUBSYSTEM` 表 |
|---|---|---|
| **A3a 组件层** | 新加 `component()` / `classify2()` + 新文件 `components.json` | **不碰** |
| **A3b 正则表扩展** | 往 `SUBSYSTEM` 里插新条目 | 碰，且顺序敏感 |

### 本会话的理由（主张 A3a 可以先做）

组件级的账只靠 A3a 就能算，而第一批前两份记录要的正是组件级的账：

- `minimax-h3` 要分开 DiT 33B / 文本编码器 33B / VAE 5.2B
- `ltx-2.3` 要分开主干 18.56B / connector 2.0B / 文本投影 1.16B

只有 `magi-2-preview` 要的是 DiT **桶内**的专家 / mHC / 注意力拆分，那才是 A3b。
把 A3a 一起压到整合之后，等于让前两份记录陪着第三份一起等。

实现上也刻意压低了撞车面：只在文件顶部加函数，`SUBSYSTEM` 列表一行未动；
`classify()` 的签名与行为完全不变；语言模型张量全部落缺省桶 `backbone`，
15 份现有记录的 derive 输出逐字不变。

### 主会话的判断（主张 A3a 也应该等）

「加性」是**语义上**不冲突，但 A3a 和 P1 整合改的是**同一个 `subsystems.py` 文件**，
而 GLM 线报的那条「必改」大概率就落在归类表附近。
刚刚才花力气把三个 DeepSeek 提交从 `p1-glm` 上拆下来、cherry-pick 到 main，
不值得为省几小时再造一次同样的麻烦。

### 决定：不回退（用户，2026-10-30）

两种意见的分歧在**要不要回退**，不在拆法本身。用户拍板：**不回退**，
A3a 留在 `5557c6c` 上，P1 整合时按下面的办法叠。

本会话当时的建议（与决定一致）：不回退，理由是回退本身也有成本（schema 与 derive 的组件维依赖它），
而真实的合并风险可以这样消：**P1 整合时以 GLM 线的改动为准先落地，
再把本分支的 `component()` / `classify2()` 几个函数原样叠上去** ——
它们和 `SUBSYSTEM` 表没有交集，叠的时候只需确认 `classify()` 的签名没被 GLM 线改掉。

**整合时的具体做法**（已决定走这条）：

1. 先把 GLM 线对 `SUBSYSTEM` 表的改动落进 `main`
2. 再把本分支 `subsystems.py` 顶部的 `component()` / `classify2()` / `COMPONENT_ORDER` /
   `COMPONENT_LABEL` / `SPLIT_INSIDE` / `_COMPONENTS` / `_COMPONENT_PREFIXES` / `DEFAULT_COMPONENT`
   连同 `data/schema/components.json` 原样叠上去 —— 它们和 `SUBSYSTEM` 表没有交集
3. **唯一要确认的一点**：`classify()` 的签名（`name`, `mtp_from`）没被 GLM 线改掉。
   `classify2()` 直接调它，签名变了要同步
4. 叠完重跑 LLM 侧 15 份记录的 `derive.py`，摘要与 JSON 应逐字不变（本分支已验证过一次）

回退方案（已不采用，留档）：范围是 `scripts/lib/subsystems.py` 与 `data/schema/components.json`，
以及 `derive.py` 里 `classify2` / `params_by_component` / `weight_bytes_by_component` 三处引用；
schema 侧与 `gen_workload()` 不受影响。

---

## 4. 归类验收用例（整合 A 节时照这张表验）

用仓里的 `scripts/lib/subsystems.py` 实跑。**这不是推测，是实跑输出。**

- 「A3a 前」= 只有 `classify()` 时的结果
- 「A3a 后」= 当前 `classify2()` 的 `组件/子系统`，组件不细分时只显示组件名
- **17 例仍需 A3b**，全部落在 `dit` 桶内

| 张量名（真实） | 来自 | A3a 前 | A3a 后 | 应该归到 | |
|---|---|---|---|---|---|
| `block.layers.N.mlp.moe_mlp.W_up` | MAGI-2 | dense_ffn | dit/dense_ffn | dit/expert | **A3b** |
| `block.layers.N.mlp.moe_mlp.W_down` | MAGI-2 | dense_ffn | dit/dense_ffn | dit/expert | **A3b** |
| `block.layers.N.mlp.moe_mlp.W_gate` | MAGI-2 | dense_ffn | dit/dense_ffn | dit/expert | **A3b** |
| `block.layers.N.mlp.moe_mlp.gate` | MAGI-2 | router | dit/router | dit/router | ok |
| `block.layers.N.mlp.moe_mlp.router.expert_bias` | MAGI-2 | router | dit/router | dit/router | ok |
| `block.layers.N.mlp.shared_expert_fc1.weight` | MAGI-2 | shared_expert | dit/shared_expert | dit/shared_expert | ok |
| `block.layers.N.mlp.modality_specific_shared_expert_fc1.weight` | MAGI-2 | shared_expert | dit/shared_expert | dit/shared_expert | ok |
| `block.layers.N.mlp.split_linear.weight` | MAGI-2 | dense_ffn | dit/dense_ffn | dit/expert 投影 | **A3b** |
| `block.layers.N.mlp.merge_linear.weight` | MAGI-2 | dense_ffn | dit/dense_ffn | dit/expert 投影 | **A3b** |
| `block.layers.N.attention.linear_qkv.weight` | MAGI-2 | other | dit/other | dit/attention | **A3b** |
| `block.layers.N.attention.linear_proj.weight` | MAGI-2 | other | dit/other | dit/attention | **A3b** |
| `block.layers.N.attention.linear_g.weight` | MAGI-2 | other | dit/other | dit/attention | **A3b** |
| `block.layers.N.attention.sinks` | MAGI-2 | other | dit/other | dit/attention | **A3b** |
| `block.layers.N.mhc_alpha_post_attn` | MAGI-2 | attention | dit/attention | dit/residual | **A3b** |
| `block.layers.N.mhc_phi_fused_attn` | MAGI-2 | attention | dit/attention | dit/residual | **A3b** |
| `block.layers.N.mhc_bias_pre_mlp` | MAGI-2 | dense_ffn | dit/dense_ffn | dit/residual | **A3b** |
| `block.layers.N.mhc_norm.weight` | MAGI-2 | norm | dit/norm | dit/norm | ok |
| `pre_adapter.video_embedder.weight` | MAGI-2 | other | dit/other | dit/patchify | **A3b** |
| `pre_adapter.audio_embedder.weight` | MAGI-2 | audio（巧合） | dit/audio | dit/patchify | **A3b** |
| `post_adapter.final_linear_video.weight` | MAGI-2 | other | dit/other | dit/patchify | **A3b** |
| `text_encoder.…layers.N.mlp.down_proj.weight` | MiniMax-H3 | dense_ffn | text_encoder | text_encoder | ok |
| `text_encoder.…layers.N.self_attn.q_proj.weight` | MiniMax-H3 | attention | text_encoder | text_encoder | ok |
| `vae.decoder.proj_in.weight` | MiniMax-H3 | other | vae | vae | ok |
| `vae.encoder.conv_in.weight` | MiniMax-H3 | other | vae | vae | ok |
| `audio_proj_in.weight` | MiniMax-H3 | audio（巧合） | **backbone/audio** | dit/patchify | **A3b**（见下） |
| `model.diffusion_model.transformer_blocks.N.attn1.to_q.weight` | LTX-2.3 | attention | dit/attention | dit/attention | ok |
| `model.diffusion_model.video_embeddings_connector.…ff.net.0.proj.weight` | LTX-2.3 | other | connector | connector | ok |
| `model.diffusion_model.adaln_single.linear.weight` | LTX-2.3 | other | dit/other | dit/adaln | **A3b** |
| `vocoder.conv_pre.weight` | LTX-2.3 | other | vocoder | vocoder | ok |
| `audio_vae.encoder.…conv1.weight` | LTX-2.3 | audio（巧合） | audio_vae | audio_vae | ok |
| `text_embedding_projection.0.weight` | LTX-2.3 | other | text_encoder | text_encoder | ok |

复现：

```python
import sys; sys.path.insert(0, 'scripts')
from lib import subsystems as S
print(S.classify2('block.layers.5.mlp.moe_mlp.W_up'))   # ('dit', 'dense_ffn') —— 待 A3b
```

### 三条值得单独说的

1. **MAGI-2 的 36 层专家堆栈（约占全模型 95% 的参数）整体落进 `dense_ffn`。**
   `expert` 的正则是 `\.experts?\.\d+\.|\bexperts\.|\.moe\.(gate|up|down)_proj\b`，
   `moe_mlp.W_up` 一条都不匹配；而 `dense_ffn` 的 `mlp` 命中了 `moe_mlp` 里的 `mlp`。
   **不报错，只是算错。**

2. **mHC 系数落进 attention / dense_ffn** —— 正是 `subsystems.py` 注释里已经警告过的那个坑，
   只因为 MAGI-2 写作 `mhc_` 而现有正则写的是 `^hc_`。这些系数是 F32，
   混进 BF16 的注意力桶会把精度表带偏（规则 6：可信度不同的东西不要乘进同一个字段）。

3. **`audio_proj_in.weight` 落进 `backbone`，不是 `dit`。**
   它是 MiniMax-H3 diffusers 布局里 `transformer/` 子目录下的顶层张量，名字不带任何组件前缀。
   `fetch_hf.py` 取数时是否给它补上 `transformer.` 前缀，决定了它落哪个桶。
   **这是 A3b 之外的一个 open question**：要么取数时统一补前缀，
   要么 `component()` 要能按「这份 shapes 来自哪个子目录」来判，而不只看张量名。

---

## 5. 两条潜在新线

两条都**不建议并进视频线**，理由是负载模型不同 —— 并进来会让「单步 × 步数」
这个基本量在相当一部分记录上失效。候选清单在 Gate A §2.4。

### 世界模型 / 交互式视频

`tencent/Hy-Nomas`、`Skywork/Matrix-Game-4.0`、`AlayaLab/Evoke` + `AlayaWorld`、
`robbyant/lingbot-world-*`、`nvidia/Cosmos3-*`、`Efficient-Large-Model/LongLive-2.0-5B`。

它们是**自回归逐块生成**：KV cache 回来了、要算每帧时延与首帧延迟、
动作与相机是逐帧条件。一次性生成一整段的 DiT 那套「单步 × 步数」在这里不成立。
schema 可以共用生成类的 VAE / 文本编码器块，但采样与负载块要另建 ——
所以它该是一条线，不是视频线里的几份记录。

**时机**：建议等第一批三份录完。现在开 Gate A 就是凭空猜 `generation` 块能复用多少，
而「凭空设计一定会漏字段或建一堆没人填的字段」正是这次视频线 Gate A 要避免的事。

### 图像生成

PLAN §3.2 要求 schema 按 `image-gen` + `video-gen` 一次建齐。
**本轮一个图像模型都没调研** —— 现在的 `generation` 块是纯粹从视频候选里摸出来的。

图像侧没有时间维（无帧数 / 时长 / fps，`vae.temporal_compression` 必须允许缺省），
但有视频侧没有的东西（多尺度、refiner、ControlNet 生态）。
建议做一次 **2–3 个模型的小勘探**，目的只有一个：验证现有字段结构留的扩展位够不够，
而不是把 image-gen 的字段也一次建完。

扫视频时撞见过几个线索，**未经查证，只是名字**：
`Qwen-Image-2.1` / `Qwen-Image-2512`、`Z-Image-Turbo`（以上阿里）、
`HunyuanImage-2.1`（腾讯）、`FLUX.1-Redux-dev`（BFL，HunyuanVideo-1.5 的 SigLIP 就从这个仓取）。
要做得先按视频线那套方法正经扫一遍 HF，不能照这几个名字开工。

---

## 6. LTX-2.5 的访问申请

**这件事已转给用户，本会话与其它会话都不代为申请、不代为接受条款。**

### 要申请什么

两个仓都是 gated，都要申请：

| 仓 | 页面 | 用途 |
|---|---|---|
| `Lightricks/LTX-2.5` | https://huggingface.co/Lightricks/LTX-2.5 | 原始权重与模型卡 |
| `Lightricks/LTX-2.5-Diffusers` | https://huggingface.co/Lightricks/LTX-2.5-Diffusers | diffusers 布局的各组件 config |

在仓页面顶部的门禁框里点同意并提交；批准后用同一个 HF 账号的 token 访问。
`LTX-2.5-Diffusers` 的门禁标记是 `gated: auto`（自动批准），`LTX-2.5` 是需要人工批准的那种。

### 门禁条件

许可是 **LTX-2 Community License Agreement**（非 Apache/MIT）。
同意条款这件事有法律含义，所以必须由用户本人做。

### 为什么非要它不可

`Lightricks/LTX-2.5` 是当前下载量最高的开源视频模型（2.27M 下载、9879 likes）。
未授权时**连 `README.md` 都取不到** —— 不是只缺权重，是连模型卡都读不了。

能从不受门禁限制的 HF 元数据 API 推出来的只有两件：
参数规模仍是 22b 档（由 `Lightricks/LTX-2.5-22b-IC-LoRA-*` 的仓名推）；
Diffusers 版相对 LTX-2 新增了 `transformer_full/`、`diffusion_decoder/`、`duration_head/`、
`prompt_enhancer/`、`temporal_latent_upsampler/`、`processor/` 六个组件目录。

社区有 `unsloth/LTX-2.5-FP8`、`comfyicu/LTX-2.5`、`Abiray/LTX-2.5-Distilled-GGUF` 等镜像，
**照它们读等于用第三方转换推翻厂商口径**，正是 model-audit 规则 10 禁止的那条链。

### 影响

批下来之前 `ltx` family 的最新记录只能停在 `ltx-2.3`，名册里挂 `status: blocked-gated`。
`hf.gated` 这个 schema 字段已经有了，正好用它。

### 进展记录

| 日期 | 进展 |
|---|---|
| 2026-10-30 | 已转给用户，尚未申请 |
| | |

---

## 7. 我没能覆盖到的部分

### 取不到证据的

1. **LTX-2.5 的全部结构细节** —— 见 §6。
2. **HunyuanVideo-1.5 的官方 MLLM（HunyuanMLLM）未发布。** 记录里「文本编码器」填什么要先定口径：
   填「未发布，社区用 Qwen2.5-VL-7B-Instruct」还是填替代品本身？两种写法下
   `text_encoders[].params` 的含义不同。schema 里已经留了 `substitutable` 字段，但口径要人定。
3. **MiniMax-H3 的默认采样步数模型卡上根本没写。** sglang cookbook 给 50，但那是引擎文档
   （优先级最低），只能标 `judgement`。第三方蒸馏 LoRA 做到 8 / 4 步，更不能当官方值。

### 查到但没验到底的

4. **MiniMax-H3「33B 里约 13B 是 AdaLN 分支、推理不必加载」** 是卡的原话（`official`），
   但实际能省多少显存**没验**。要按张量名分桶统计 AdaLN 相关张量的 numel 才能给数。
   在那之前**不要把 33 − 13 = 20 写进任何字段** —— 那是两个口径相减。
5. **Wan2.2 的两套专家在实际部署里是否必须同时驻留。** `boundary` 只切换一次，
   理论上可以换入换出；厂商脚本用 `--offload_model True` 统一管理，没有单独说明。
   权重存储算 28.6B 还是 14.3B，未定。schema 里已留 `expert_switching.both_resident`，值待定。
6. **HunyuanVideo-1.5 的 SR 模块到底多大。** `upsampler/720p_sr_distilled` 只有 0.09 GB，
   而 `transformer/720p_sr_distilled` 是 33.3 GB —— 看起来 SR 实际是整个 DiT 的另一套权重
   加一个小 upsampler，但没有确认。
7. **LTX-2.3 的 VAE / 文本编码器是否与 LTX-2 完全一致。** 2.3 的单文件里 VAE 是 0.726B，
   LTX-2 的 VAE 没有单独统计过，两边未逐张量比对。
8. **LTX-2.3 的 `to_gate_logits [32, 4096]`** 是逐头门控 —— 从张量形状读出来的（`judgement`），
   没有卡或论文印证具体语义（输出门控还是注意力选择门）。arXiv 2601.03233 没读，只读了模型卡。
9. **Wan 与 HunyuanVideo-1.5「落盘 FP32、推理 BF16」** 已由权重头实测确认，
   但**没有确认厂商是否在别处发过 BF16 版本**。若有，`weights_release` 要写成数组。
10. **StepVideo 的 VAE 参数与 latent 通道数。** `in_channels: 64` 是 DiT 侧的入口，
    推 VAE latent 是 64 通道是合理推断（`derived`），但 VAE 本体不在 HF 仓里，没有直接证据。
11. **MAGI-2 的采样参数一条都没有** —— 默认步数、CFG、是否蒸馏、preview 与 refiner 各跑几步，
    卡上全无。要去读 sand.ai 技术博客或 GitHub 推理代码。
12. **MAGI-2 的 MoE 口径有三个数没定**：top-k 是多少、`split_linear` 把 3072 维切成 12 片
    256 维之后每片各自路由还是共享路由、「每 token 激活 6B」是按几片算的。
    这三个不定，激活参数与 EP 通信量都算不出来。
13. **MAGI-2 仓里的 `vae/Wan2.2_VAE.pth` 与 Wan-AI 官方那份是否逐张量一致**，没核对，可能是微调过的。

### 范围上的空白

14. **昇腾侧只查到存在性。** 确认有 Ascend-SACT（AtomGit）的 HunyuanVideo / HunyuanVideo-1.5
    适配条目，以及华为云 ModelArts 的「Wan2.1/Wan2.2/HunyuanVideo/CogVideo 系列基于轻量算力节点
    适配 NPU 推理指导」。但**没有逐个核对**支持哪些模型、到什么精度、什么时间、有没有文档化配方。
    关键差别：这些适配走 ModelArts / SACT 的独立配方，**不是 vllm-ascend** ——
    和语言模型线的平台矩阵不是同一个入口。
15. **引擎侧只做了粗扫**：TensorRT-LLM 的 visual generation（beta）列了 Wan2.1/2.2、LTX-2、
    MiniMax-H3（H3 的 Ref2VA 未开放、注意力限 SM100/SM103），vLLM-Omni 有 Wan2.2 I2V 配方，
    SGLang 有 diffusion cookbook。**没有逐个核对版本号与支持状态**，平台矩阵要在录记录时重做。
16. **只扫了 HuggingFace。** 只发在 ModelScope 或只发在 GitHub release 的厂商模型可能漏掉。
17. **扫描方法有个缺口，已经漏过一次。** 候选是按 HF `pipeline_tag` + 下载量/likes 排序扫的，
    而 `sand-ai/MAGI-2-preview` 的 HF **下载量显示为 0**（`MAGI-1` 也是 0），
    一张榜都没进，是用户追问才补上的。下载量为 0 的原因没查清，
    可能是权重以 `.pth` / `.ckpt` 这类非标准布局发布导致计数器不统计。
    **下载量不能当唯一的发现手段**：双周采集要同时按 `createdAt` 扫全量 video 类 pipeline_tag，
    并维护厂商 org 白名单（sand-ai 之前不在名单里）。
18. **图像生成完全没看** —— 见 §5。
19. **Open-Sora v2 的 DC-AE 压缩率、Mochi 的 AsymmDiT 细节没深挖** —— 因为决定不录。
    若要翻案，这两条要补。

### 本次实现里留下的

20. **`audio_proj_in` 一类无前缀的 DiT 顶层张量落进 `backbone` 而不是 `dit`** —— 见 §4 第 3 条。
    取数时补前缀、还是让 `component()` 认得「这份 shapes 来自哪个子目录」，两条路都没定。
21. **`gen_workload()` 的注意力 FLOPs 是稠密全注意力口径。**
    块稀疏（SSTA）与线性注意力（SANA）要按各自形态折算，函数没有替人折 ——
    折算比例是记录里的事实，不是公式里的常数。录 `hunyuanvideo-1.5` 与
    `sana-video-2.0-5b` 时要先定这个口径。
22. **多阶段端到端是粗估。** 各阶段按同一个单步 FLOPs 算，而 SR / 上采样阶段的实际 token 数
    不同。输出里标了 `note`，但录记录时不要把它当精确值报出去。
