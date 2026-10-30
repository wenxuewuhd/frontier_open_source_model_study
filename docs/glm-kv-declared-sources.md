# GLM 九代模型卡正文：KV 字节的查找结果与声明参数量

> **这是为后续「KV 落盘精度清扫」准备的工作笔记，不是口径定义文档。**
> 口径定义在 [kv-cache-accounting.md](kv-cache-accounting.md)，案例史在
> [../README.md](../README.md) 的「用例」一节。
> 这份文件的作用是让接手清扫的人**不必把九份卡再查一遍**；
> 内容整合进各记录（或并进 README 的案例史）之后可以删掉。
>
> 写于 2026-10-30，`p1-glm` 分支。

## 0. 一句话结论

**九份 GLM 模型卡的正文里没有任何厂商公布的每 token KV 字节。** 零命中。
唯一沾边的数字是 `glm-5.2` 卡里的 per-token **FLOPs** 2.9×，它不是缓存口径、不能当锚（见 §3）。
下一步该读的是 arXiv **2602.15763**（GLM-5 技术报告）与 **2603.12201**（IndexShare），
两份本仓都没读（见 §4）。

## 1. 查法：搜了哪些正则，覆盖哪几份卡

九份 `README.md` 全部按 pinned sha 抓取，逐行读过，并做全文正则检索。
sha 取自各自 `facts/<id>/repoinfo.json`，与记录 `hf.sha` 一致：

| 本仓 id | HF repo | sha | 卡正文体量 |
|---|---|---|---|
| `glm-4-9b` | `zai-org/glm-4-9b-chat` | `bd8234fe5e0c09c48637a92abb0c797cb5fa0e73` | 7,772 B / 173 行 |
| `glm-4-32b-0414` | `zai-org/GLM-4-32B-0414` | `077b5c2f5c43bd3239fd605a0600229e8facbd4a` | 42,936 B / 397 行 |
| `glm-4.5` | `zai-org/GLM-4.5` | `cbb2c7cfb52fa128a9660cb1a7a78e017899e115` | 9,622 B / 186 行 |
| `glm-4.6` | `zai-org/GLM-4.6` | `be72194883d968d7923a07e2f61681ea9a2826d1` | 3,044 B / 59 行 |
| `glm-4.7` | `zai-org/GLM-4.7` | `602d01efcdd332c5238ca4bcede555defbe83eb7` | 13,582 B / 219 行 |
| `glm-4.7-flash` | `zai-org/GLM-4.7-Flash` | `7dd20894a642a0aa287e9827cb1a1f7f91386b67` | 8,156 B / 172 行 |
| `glm-5` | `zai-org/GLM-5` | `c183ef8c61faee82855eca1ed9bb3a9a7ce3b0b2` | 12,708 B / 147 行 |
| `glm-5.2` | `zai-org/GLM-5.2` | `cf457fa734ab149ffef225f80893eb38c6ff5cdc` | 10,908 B / 108 行 |
| `glm-5.3` | `zai-org/GLM-5.3` | `ba32f8c19c8738e2c57c71798c601d477a7bf844` | 14,163 B / 90 行 |

抓法（可复现）：

```
https://huggingface.co/<repo>/raw/<sha>/README.md
```

检索的正则，三组：

```
# 缓存字节
kv.?cache | per.token.*(byte|KB|MB|GB) | (byte|KB|MB|GB).*per.token
cache (size|capacity|memory|footprint) | 缓存.*(字节|KB|MB|GB)

# 参数量
[0-9]+ ?(billion|B)\b.{0,20}param | param.{0,20}[0-9]+ ?(billion|B)\b
total param | active param | scales from | 355B | 744B | 106B | 30B

# 训练精度与上下文
trained in (fp8|bf16) | training (precision|in fp8) | fp8 (training|mixed)
quantiz | MXFP4 | NVFP4 | INT4 | W8A8
[0-9]+ ?[KkMm] (token|context) | context (window|length) of
```

另外还搜了 `DSA|indexer|MLA|latent|压缩|compress|倍|×|HBM|显存|footprint` 做兜底。

**两类假命中要排掉**（搜 `kv`/`cache` 时一定会撞上）：

- `kvcache-ai` —— KTransformers 的 GitHub org 名，九份卡里多处出现（部署章节的链接）
- `--gpu-memory-utilization 0.85`、`--cpu-offload-gb 16` —— vLLM / SGLang 的命令行参数

## 2. 零命中这个结论的边界

**查过的**：上表九份卡正文，逐行加正则。

**没查的**：

- **技术报告**：arXiv 2602.15763（GLM-5）、2603.12201（IndexShare）、2508.06471（GLM-4.5）
- **z.ai 博客**：`/blog/glm-4.5`、`/blog/glm-4.6`、`/blog/glm-4.7`、`/blog/glm-5.2`
- **智谱飞书技术文档**（`glm-4.5` / `glm-4.6` 卡里引的那个 wiki 链接）
- 各代的 `*-FP8` / `*-BF16` 变体仓的卡（本仓都没收录这些仓）
- `GLM-5.3-Flash`（`glm5_next`，超出 GLM-5.3 这条线的范围）

所以**「厂商从未公布每 token KV 字节」这句话本仓现在说不了** ——
能说的只是「九份卡正文里没有」。各记录的 `reviewed.method` 里写明了这个作用域。

**一处容易被搜到后误采的**：`glm-4.5` 卡第 68 行

> "All are executed on devices that natively support FP8 inference, ensuring both weights
> and **cache** are in FP8 format."

它在 benchmark 复现条件那一节，说的是**评测时的部署配置**，不是发布权重的 KV 落盘格式。
全九份里只有这一处把 cache 和某个精度名放在一起，**不能当 `precision.kv_cache` 的一手来源**。

## 3. 沾边但不可用：`glm-5.2` 的 per-token FLOPs 2.9×

- **原话 + 定位**：`glm-5.2` README @ `cf457fa734ab`，**第 36 行**
  > "We propose [IndexShare](https://arxiv.org/abs/2603.12201), which reuses the same indexer
  > across every four sparse attention layers, **reducing per-token FLOPs by 2.9× at a 1M
  > context length**. We also improve GLM-5.2's MTP layer for speculative decoding,
  > increasing the acceptance length by up to 20%"

- **它的基准**：per-token **FLOPs**；上下文 **1M**；精度未提；**分母那一代没点名**
  （按内容应是 GLM-5 / GLM-5.1 的逐层 indexer，但卡上没写）。不含任何缓存口径。

- **为什么不能当缓存锚**：它和本仓算出来的两个数是三个不同的量 ——

  | 量 | GLM-5 → GLM-5.2 | 出处 |
  |---|---|---|
  | index K 元素 / token | 9,984 → 2,688 = **3.71×** | 本仓逐层实测（`indexer.wk` 输出维 128 × 层集合） |
  | KV 元素 / token（含全量 MLA） | 54,912 → 47,616 = **1.133×** | 同上 |
  | **per-token FLOPs** | **2.9×** | 卡第 36 行 |

  按 CLAUDE.md 的「不同口径之间不算倍数」，它最多进 `open_question`，
  不能填进任何字节或元素字段。`glm-5.2` 记录里就是这么处理的。

- **它确实有用的地方**：厂商自己说的 "reuses the same indexer across every four sparse
  attention layers" **独立印证了机制** —— 本仓原来只有两路证据
  （config 的 `indexer_types` 与清单里 `indexer.wk` 的层集合），`sparse.mechanism` 标 `judgement`。

- **但它印证不了层集合**：卡上说的是 "every four"，而实测层集合是
  `[0,1,2,6,10,…,74]`、间隔 `1,1,4,4,…`。**前三层的例外卡上没提**，
  按卡上这句字面推仍会得到 20 层（`glm-5.3` 的 R3 记的就是这个坑）。
  所以这条厂商说法**印证机制、不印证层集合**。

## 4. 下一步落点：两篇 arXiv，以及可能有什么

| arXiv | 是什么 | 被哪些卡引 | 判断里面可能有什么 |
|---|---|---|---|
| **2602.15763** | GLM-5 技术报告 | `glm-5` 第 88 行、`glm-5.2` 第 19 / 106 行 | 最可能有 KV 数字的一份。`glm-5` 卡第 32 行对 DSA 只给了定性的 "largely reducing deployment cost" —— 这种卖点通常在报告里配一个具体数或跨代图。DeepSeek 那边就是从技术报告的跨代图里查出 V4 一代高了 1.7 倍的 |
| **2603.12201** | IndexShare | `glm-5.2` 第 36 行 | 2.9× 那个 FLOPs 数的出处。IndexShare 的卖点是省 indexer，所以它大概率同时给 index K 的缓存账；若给了，正好能校验本仓的 128 宽度与 21 层层集合 |
| 2508.06471 | GLM-4.5 技术报告 | `glm-4.5` / `4.6` / `4.7` / `4.7-flash` | GLM-4.5 一代是 GQA、没有稀疏卖点，所以 KV 数字的可能性低。但它可能有「355B/32B」那两个声明值的精确定义（见 §5），能把 §5 的口径推断从判断升级成官方 |

`glm-5.3` 的卡最薄（90 行）且**不引任何技术报告**，所以 GLM-5.3 这一代
目前没有任何厂商侧的结构或缓存数字可查。

**若报告里真有每 token KV 字节，它还能顺手定下另一件事**：
`arch.attention.head_dim` 的跨族系语义分歧（`glm-4.7-flash` R12 / `glm-5` R11 /
`glm-5.2` R11，三条挂着 `resolution: open`）。`ds-v3` 存的 `head_dim` 是 nope+rope = QK 全宽，
GLM 四份存的只是 nope 段；`derive.py` 在**无** `kv_lora_rank` 时会拿
`num_kv_heads × head_dim` 当 latent 宽度，所以这个分歧在那条分支上有数值后果。
一个厂商公布的 KV 字节能当第三方裁判。

## 5. 顺带查出来的：四份卡里有声明参数量

这不是 KV 的事，但它是同一次检索的产物，而且推翻了记录里的一句话
（原来五份写着「厂商未声明总参数或激活参数」）。已落到提交 `4a63146`。

| 记录 | 卡里的声明 | 定位 |
|---|---|---|
| `glm-4.5` | "355 billion total parameters with 32 billion active parameters" | README @ `cbb2c7cfb52f` 第 27 行 |
| `glm-5` | "scales from 355B parameters (32B active) to **744B parameters (40B active)**" | README @ `c183ef8c61fa` 第 32 行 |
| `glm-4.7-flash` | "a **30B-A3B** MoE model" | README @ `7dd20894a642` 第 27 行 |
| `glm-4-32b-0414` | "featuring **32 billion parameters**" | README @ `077b5c2f5c43` 第 14 行 |

另五份（`glm-4-9b` / `4.6` / `4.7` / `5.2` / `5.3`）卡里**确实没有**参数量数字，逐行核过。

### 5.1 总参数：声明值 = 实测 − draft，三代同时成立

| 记录 | 声明 | 实测 total | 偏差 | 实测 − draft | 偏差 |
|---|---|---|---|---|---|
| `glm-4.5` | 355B | 358,337,791,296 | +0.940% | 352,797,829,024 | **−0.620%** |
| `glm-4.7-flash` | 30B | 31,221,488,576 | **+4.072%**（超 2% 容差） | 29,943,393,920 | **−0.189%** |
| `glm-5` | 744B | 753,864,139,008 | +1.326% | 743,911,218,432 | **−0.012%** |

三代档位差 20 倍而同一个口径同时成立，所以是规律不是巧合。
`glm-4.7-flash` 是最硬的判例：不扣 draft 就超容差、扣了就几乎逐位。

### 5.2 激活参数：第三套口径，推翻了 Qwen 一轮结论的普适性

| 记录 | 声明 | 带宽口径（含 lm_head） | 含输入词表 | **带宽口径 − lm_head** |
|---|---|---|---|---|
| `glm-4.5` | 32B | 32,856,319,904（+2.676%） | 33,632,266,144（+5.101%） | 32,080,373,664（**+0.251%**） |
| `glm-5` | 40B | 40,833,146,112（+2.083%） | 41,784,728,832（+4.462%） | 39,881,563,392（**−0.296%**） |

厂商的 "active parameters" **把输入词表与 lm_head 两头都排除**，只算 transformer 主干。
而 Qwen 一轮的结论是「官方激活参数**含**输入词表，与带宽相关的激活参数差 4%」——
方向相反。所以**「官方激活参数含什么」是逐厂商的，不能跨族系套用**。

记录里**没有填** `active_decode` / `active_prefill`：卡上只说 "active parameters"，
没区分 prefill 与 decode，填了会断言一个卡上没有的口径。

### 5.3 `glm-4.7-flash` 的 A3B 对不上，照实记

名字与卡里的 A3B 即声明激活 3B，而带宽口径 3,579,571,840 对它是 **+19.3%**、
带宽口径扣 lm_head 的 3,262,377,600 是 **+8.7%** —— 两套都差得远。
`glm-4.5` / `glm-5` 的 active 声明值都能对到千分之几，本份对不上，
所以 A3B 更像档位命名的取整而不是参数量声明，记录里不当可对账的声明值用。

### 5.4 `glm-4-32b-0414` 的 +1.77% 为什么比另三个大（整合时单独看的那一份）

**结论：它偏得多是因为这一代根本没有可扣的东西，+1.77% 就是厂商那边的取整幅度。**

- 另三份的 ±0.62% 以内是**扣掉 draft 之后**的结果。`glm-4-32b-0414` 是
  GLM 线上最后一代稠密模型，**没有 MTP 草稿块**（config 无 `num_nextn_predict_layers`，
  层号止于 60），所以 `excludes` 是空的 —— 声明值与实测是同一组张量，
  +1.77% 不含任何口径差。
- **穷举过还能扣什么**（schema 的 `excludes` enum 只允许
  `draft/vision/audio/retrieval/embed/expert`，这一代只有 `embed` 存在）：

  | 扣掉 | 结果 | 偏差 |
  |---|---|---|
  | 不扣 | 32,566,081,536 | **+1.769%** |
  | `embed` | 31,634,946,048 | −1.141% |
  | `expert` | 32,566,081,536（这一代没有专家） | +1.769% |

  扣 `embed` 的 |−1.14%| 确实比 |+1.77%| 小，但**没有任何理由扣它**：
  另三代的口径是扣 draft，这一代没有 draft；而「声明值不含输入词表」在本家族
  没有第二个判例支持。为了把偏差做小去挑一个无依据的口径，正是本仓反复踩的那种错
  （Qwen 一轮的「扣哪几项仍是判断」）。所以记录里不扣。
- **同族稠密一代的取整幅度对照**，支持「这就是命名取整」：

  | 记录 | 实测 | 名字 / 卡 | 偏差 |
  |---|---|---|---|
  | `glm-4-9b` | 9,399,951,392 | 9B | **+4.444%** |
  | `glm-4-32b-0414` | 32,566,081,536 | 32B | **+1.769%** |

  `glm-4-9b` 的卡正文**没有**参数量数字，名字里的 9B 两种口径都超容差
  （全量 +4.44%、扣输入词表 −2.45%），所以那一份的 `total` 仍是实测值、不填声明值。
  `glm-4-32b-0414` 的 32B 则是卡正文明写的，而且 32.57 → "32B" 与
  9.40 → "9B" 是同一种向下取整风格 —— 稠密一代厂商报的是**档位名**，
  不是独立算出来的参数总量。
- 所以整合时的判断是：这一份的 Gate B「参数总量」`+1.77%` 在容差内但**不构成强对账**，
  它的价值只在于把来源从「只能拿名字凑」换成了厂商正文声明。
  真要收紧，得去 arXiv 2508.06471 看 GLM-4 一代有没有给精确参数量。

## 6. 其它：卡里还有的一手说法（已采用）

| 记录 | 卡里的说法 | 定位 | 本仓怎么用 |
|---|---|---|---|
| `glm-4.6` | "The context window has been **expanded from 128K to 200K tokens**" | 第 29 行 | 128K / 200K 是 131072 / 202752 的取整（202752 / 1024 = 198）。收窄了「上下文怎么来的」那条 open question —— "expanded" 与能力提升的措辞倾向于不止改一个上限，但卡上没说做了什么 |
| `glm-5` | "a maximum context length of **202,752 tokens**" | 第 66 行（HLE-with-tools 评测设置） | **九份卡里唯一给出精确值而非取整的上下文数字**，与 config 的 `max_position_embeddings` 逐位相同 |
| `glm-5.2` | "a **solid 1M-token context**" | 第 33–34 行 | 取整说法，与 config 的 1048576 一致 |
| `glm-4-9b` | 「我们还推出了支持 1M 上下文长度（约 200 万中文字符）的模型」 | 第 30 行 | 厂商自己把 1M 档当成**另一个模型**，印证了不收 `glm-4-9b-chat-1m` 的判断（那份的 `multi_query_group_num` 是 4 而不是 2，缓存宽度翻倍） |

**九份卡里都没有训练精度。** 所有记录的 `precision.training` 继续不填，
`open_questions` 里的措辞已从「模型卡正文未读」改成「已读、里面没有」。

## 7. 复现这份笔记需要的东西

抓卡正文不需要本仓的任何脚本，九条 raw URL 在 §1。
本文件里所有实测数字都能从 `data/derived/glm-*.json` 的 `from_shapes` 与 `active_params`
读出来，或用 `scripts/derive.py --id <id> --model data/models/glm/<id>.yaml` 重算。

> **在 worktree 里跑 derive 的坑**：`facts/` 与 `.venv/` 都在 `.gitignore` 里、
> 只存在于主 checkout。`--facts-dir <主 checkout>/facts` 会把**绝对路径**写进
> `data/derived/*.json` 的 `inputs.facts_dir`，那个不该提交；
> 临时 `ln -s <主 checkout>/facts facts` 再不带 `--facts-dir` 跑，
> derived 里就仍是相对的 `facts/<id>`。注意这个符号链接**不被 `.gitignore` 的
> `facts/` 匹配**（带斜杠只匹配目录），`git status` 里是 `??` —— 别用 `git add -A`。
