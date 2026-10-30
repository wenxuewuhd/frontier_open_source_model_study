# VLM infra 维度补录 · 交接

分支 `p3-vlm-infra`，16 个本地提交（未 push）。只给 12 份 `kind: vlm` 补 infra 那一维，
**没有录新模型**。

## 做了什么

`arch.vision` 下新增三个子树，12 份全部填完：

| 子树 | 回答的问题 | 字段数 |
|---|---|---|
| `tokens` | 一张图/一帧产出多少 token、压缩比、压在哪一步 | 10 |
| `prefill` | encoder 的 FLOPs、一张图相当于多少个文本 token | 7 |
| `separability` | 权重前缀、分片隔离、有没有跨边界共享 | 5 |

schema 同步改了（`additionalProperties` 仍是 `false`）。
**`arch.vision` / `arch.audio` 之外一个字节没动**，`scripts/` 全程没碰。

### 12 份的结果

同一张 672×672 的图（StepFun 两档固定 728，**不可换算**，见下）：

（下表已按并入 `main` 的 `e394525` 重算 —— 那个提交改了子系统归类，
`qwen3.8-flash-next` 的占比因此变动 9.5 倍，详见本页「并 main 之后的修正」一节。）

| id | enc tok | 主干 tok | 压缩比 | ≈文本 tok | 塔占比 | 分片 |
|---|---|---|---|---|---|---|
| step-3 | 2705 | 169 | 16 | 484 | **2.86** | mixed |
| step-5-preview | 2704 | 169 | 16 | 352 | 2.08 | **dedicated** |
| kimi-vl-a3b | 2304 | 576 | 4 | 1007 | **1.75** | mixed |
| ds-v4-flash-vision-exp | 2304 | 256 | 9 | 332 | 1.30 | mixed |
| qwen3.8-flash-next | 1764 | 441 | 4 | 564 | 1.28 | mixed |
| deepseek-v41-flash | 2304 | 256 | 9 | 320 | 1.25 | partial |
| minimax-m3 | 2304 | 576 | 4 | 647 | 1.12 | mixed |
| qwen3.5-397b-a17b | 1764 | 441 | 4 | 490 | 1.11 | mixed |
| qwen3.5-27b / qwen3.8-27b | 1764 | 441 | 4 | 472 | 1.07 | mixed |
| kimi-k2.5 | 2304 | 576 | 4 | 608 | 1.06 | **dedicated** |
| kimi-k3 | 2304 | 576 | 4 | 585 | **1.02** | **dedicated** |

## 三个最关键的发现

### 1. 「视觉塔贵不贵」描述的是主干，不是塔

塔占比从 1.02 到 2.85，横跨 2.8 倍 —— 但**最大的变量不是塔**。
三组同塔对照把这件事钉死了：

- **kimi-vl-a3b 对 kimi-k2.5**：`vision_tower` 逐张量相同，
  占比 1.75 对 1.06（塔值 431 对 32 个位置当量）。差 7.7 倍，**差的全是主干**
  （4.49 对 61.03 GFLOPs/token）。
- **Qwen3.5/3.8 四档**：333 个 `model.visual.` 张量跨四档逐名、逐形状、逐 dtype
  相同（只有 `merger.linear_fc2` 的出口宽度不同），encoder 线性项 1456.5 GFLOPs
  四档一模一样；占比 1.03 到 1.11。
- **DeepSeek 两代**：259 个塔张量形状完全一致、encoder 线性项逐位相同
  （1,896,855,109,632），占比 1.30 对 1.25。

**所以「要不要把 encoder 拆出去」不能按模型族系答，要按主干大小答。**
值得拆的只有 kimi-vl-a3b（塔占一张图 prefill 的 43%）和 StepFun 两档（65% / 52%）；
kimi-k3 那一档拆出去几乎没有收益（塔只值 9 个位置当量）。

### 2. 压缩比是个权衡，不是谁做得好

压缩比 4（九份）/ 9（DeepSeek 两档）/ 16（StepFun 两档）。压得越狠，
主干位置预算越省、塔的占比越高：

- step-3 的 16:1 把主干压到 169 个 token（全批最低），
  代价是一座 63 层 / 4.3B 参数的塔，encoder 线性项 23.1 **T**FLOPs
  （第二位 minimax-m3 是 2906 GFLOPs，差 8 倍），塔占 65%。
- kimi-k2.5 的 4:1 让主干收 576 个 token，塔只占 5.3%。

两者主干位置差 3.4 倍、塔占比差 12 倍，**方向相反**。

### 3. 压缩在 projector 之前还是之后，差一个压缩比的钱

**minimax-m3 是 12 份里唯一「先投后压」的一份**，张量形状逼出这个次序：
`multi_modal_projector.linear_1` 是 `[6144, 1280]`，输入维 1280 是 encoder hidden，
跑在**未合并**的 token 上；`patch_merge_mlp.linear_1` 是 `[6144, 24576]`，
24576 = 4 × 6144，合并的是**已投影过**的 token。

于是两层 projector 的 GEMM 跑 2304 次而不是 576 次，
projector 线性项 **427.62 GFLOPs = Qwen 同口径 39.54 的 10.8 倍**；
反过来，按别家「先压后投」的口径算 minimax 只有 106.9，**会少算 4 倍**。

这就是新增 `tokens.compression_stage` 这一格的理由 —— 次序原来只是 `fusion`
里的一句描述，现在是一个能复算的数。

## 口径上的三个坑（都写进了字段，不只是 commit）

1. **主干分母不能用 `data/derived/<id>.json` 的 `flops.linear_per_token`。**
   它 = `2 × active_params.total`，**含 `lm_head`** —— 那是 decode 口径
   （每步对全词表做 GEMM），而 prefill 只在最后一个位置算 logits。
   该用 `2 × (total − lm_head)`。`lm_head` 占分母实测 **1.1%–15.0%**，
   kimi-vl-a3b 是 15.0%（A3B 主干 + 16 万词表）。
   **这一条对每一份 vlm 都成立**，不是个别记录的事。
2. **`temporal_patch_size: 2` 不等于一张图两个时间 patch。**
   Qwen / MiniMax 的 `patch_embed` 是 Conv3d、时间核 2，
   单张静态图**要补成 2 帧才喂得进去**，所以一张图恰好 1 个时间 patch。
   按「乘 2」读会把 encoder token 数和 encoder FLOPs 同时高估一倍。
3. **固定分辨率的记录不能换算到参考分辨率。**
   StepFun 两档 `image_size` 728、位置表 `[2705/2704, …]` 钉死了网格，
   和另外十份比 token 数是范畴错误。两份记录的 `scope_note` 里明写了这一条，
   只允许比压缩比与占比。

另外**改掉了一个我自己写错的最高/最低值**：`qwen3.8-flash-next` 一度写成
「12 份里塔开销占比最低」，算到 Kimi 才发现 kimi-k3 的 1.016 更低。
已收回成「同代四档最低、跨族系第二低」（提交 `0a00fbe`）。
报最高/最低之前要把比较范围说出来 —— 和「结构规律钉代际」是同一条。

**后续**：并入 `main` 之后这条结论又被推翻了一次 —— `qwen3.8-flash-next`
现在是四档里塔**最重**的一档（1.279），不是最轻。见下一节。

## 顺手补上的空格子

- `qwen3.8-27b`：`arch.vision.params`（460,730,096）与 `intermediate`（4304）原来空着。
  补的过程拿到一个跨代判据：**它的塔与 qwen3.5-27b 的 333 个张量
  逐名、逐形状、逐 dtype 完全相同，0 处差异**。
- `deepseek-v41-flash`：`arch.vision.params`（485,268,480）原来空着，
  `fusion` 只有一句「aligner MLP 投影到主干 hidden」，连 3×3 合并都没写出来。
- `kimi-k3`：`arch.vision.spatial_merge_size`（2，来自 config 的 `merge_kernel_size`）原来空着。
- 本轮按 `shapes.json` 重算的 `params` **复现了另外九份已有的值、逐位相同**，
  这给了新增 FLOPs 数字一次独立的分区校验（分区错了这九个数不可能全对）。

## 查不到的（都留在 open_questions / 字段里，没有填推测值）

1. **`separability.runtime_support` 12 份一份都没填。** 「权重结构上可分」
   和「实现支持分离部署」是两件事，后者要一手来源（厂商参考实现里 encoder
   是否独立成服务、部署文档里有没有这条路径），本轮没有，**所以没填**，
   没有从「权重能分开」推「支持分离部署」。
2. **encoder 自注意力的跨度（整图一段 / 分窗 / 分块）只有 1 份有一手来源。**
   12 份的 `vision_config` 里都没有 `window_size` / `fullatt_block_indexes`
   （前代 Qwen2.5-VL 有这两个键），只有 kimi-k2.5 本仓读过参考实现
   （按 t×h×w 切 `cu_seqlens` 的整段注意力）。
   所以 `prefill` 每一格**只装线性项**，`attention_excluded` 里写了
   被排除项的形式与影响量级。两处影响不可忽略：
   - **kimi-vl-a3b**：按整图一段算约 660 GFLOPs，是线性项的 35%，
     会把 1007 再推高约 147 —— **1007 是下界**。
   - **minimax-m3**：每块各成一段是 217 GFLOPs、四块连一段是 869 GFLOPs，
     差 4 倍，36 块（2016²）时差 36 倍。
3. **厂商默认的「一张图」分辨率。** 动态分桶规则在
   `preprocessor_config.json` 的 `min_pixels` / `max_pixels` 里，
   那个文件**不在 `facts/`**。所以 672×672 是本仓选的参考分辨率
   （能被 14/16/28/32/42/48 整除，且正好是 MiniMax 的一个分桶），
   `tokens` 下的数只能和同样钉在 672 的记录比，**不是部署时的实际单图 token 数**。
4. **落盘层面的 tied weights。** 12 份的张量名在视觉/主干两个前缀间都不重名，
   但那只排除了**重名**；真正的存储共享要看 safetensors 头里的数据偏移是否重合，
   本轮没读偏移。（三份 `dedicated` 的记录额外强一点：两侧连分片都不重合，
   所以连「同片内共享缓冲区」也排除了。）
5. **step-3 的 `patch_token_len: 81` 背后的像素几何。** 81 = 9²，
   但 9 反推到输入边长要知道 crop 尺寸，预处理器配置不在 `facts/`。
   影响：**step-3 / step-5 的 169 只是全局视图**，单图是 `169 + 81 × patch 数`，
   所以那两份的 `prefill` 做预算会低估。`scope_note` 里明写了。
6. **step-5-preview 两级卷积的 stride 是推出来的、不是读来的。**
   两端有官方值（输入 52² 由位置表钉住、输出 169 = `image_token_len`），
   两个 3×3 卷积把 52 变 13 只有「各 stride 2」一条路。
   上一代 step-3 的 stride 有实现行号可引（`modeling_step3.py` 686–698），
   本代参考实现没读 —— 结论相同但**依据弱一档**，padding 没有来源，
   `derivation` 因此标了 `judgement`。

## schema 加了哪些字段

只动 `arch.vision`，三个子树共 22 格，全部带 `description`（口径写在 description 里）：

- `tokens`：`reference_resolution` / `patch_grid` / `encoder_tokens` / `llm_tokens` /
  `compression_ratio` / `compression_stage` / `dynamic_resolution` /
  `max_llm_tokens_declared` / `scope_note` / `derivation`
- `prefill`：`encoder_linear_gflops` / `projector_linear_gflops` /
  `backbone_token_budget` / `backbone_linear_gflops_per_token` /
  `image_as_text_tokens` / `attention_excluded` / `derivation`
- `separability`：`weight_prefix` / `shard_isolation`（枚举
  `dedicated` / `partial` / `mixed`）/ `shard_note` / `cross_prefix_sharing` /
  `runtime_support`

**`arch.audio` 一个字段没加** —— 理由见 [omni-audio-field-gap.md](omni-audio-field-gap.md)。

## 音频缺口（详见 omni-audio-field-gap.md）

- **先纠正一个前提**：schema 不是「一个音频字段都没有」。`arch.audio` 已有 5 格、
  `generation.audio` 已有 10 格、`kind` 枚举已允许 `omni` ——
  缺的是**零份记录在用**（0 / 47）和 `arch.audio` 里 infra 要的那几格。
- **这 12 份里零份带音频**：12 份 config 全文搜
  `audio|speech|wav|mel|asr|tts|sound` 命中 **0 处**；
  12 份权重清单合计 914,930 个张量里音频张量 **0 个**。
- 关键分叉：**形态 A** 连续编码器 + projector（与视觉同构）对
  **形态 B** 离散 codec token 进文本词表、主仓里没有音频编码器张量
  （StepFun 这条线，step-1 的 R6 已查明）。现有 5 格只建模了 A；
  `tokens_per_second` 在两种形态下**不是同一个量**，同名不同义而 `validate.py`
  看不出来。
- 两条跨线建议：采样率与帧率**直接借 `generation.audio` 的命名**
  （它已经把输入输出采样率分成两格）；`prefill` / `separability`
  **照搬本轮验证过的形状**，「一张图相当于多少文本 token」换成「一秒音频」。
- **明确建议字段不要先建**：先录一份形态 A、缺口进 `review` 与 `open_questions`，
  再按那份记录改 schema。零份记录时建的格子没人能验，
  而空格子会被下一轮当成「已经考虑过了」。

## 闸口状态

```
validate.py                     exit 0，47 份记录 0 错 0 警（判断类 450 → 506）
derive.py（12 份逐份带 --model 与原 --ep）  11 份 exit 0
site/npm run build              exit 0，59 页
```

**例外一处：`step-5-preview` 的 `derive.py` 返回 1。** 这是它在 `main` 上
就有的状态，不是本轮引入的：外部 HF tally 对不上（604,339,504,832 对
602,972,927,296），根因已由该记录的 R8/R9 查清 —— 18 个跨文件重名张量，
HF 逐文件相加、本仓按 `weight_map` 去重。
**已验证**：改动前后 derive 输出除时间戳外逐字相同
（拿 `git show HEAD:…` 的版本跑过对照）。

`--ep` 全部沿用已提交 JSON 里的原值，重跑后 12 份的 `ep_size` 逐份核对未变：
`1 1 4 8 8 1 1 1 1 1 4 4`（deepseek-v41-flash / ds-v4-flash-vision-exp /
kimi-k3 / kimi-k2.5 / kimi-vl-a3b / minimax-m3 / qwen3.5-27b /
qwen3.5-397b-a17b / qwen3.8-27b / qwen3.8-flash-next / step-3 / step-5-preview）。

## 并 main 之后的修正（`e394525` 落地后）

把 `main` 的三个提交（`e394525` 归类整合、`4391992` head_dim、`cc4cf38`
公布值档次）并进本分支后，`e394525` 改了 `scripts/lib/subsystems.py` 的归类与
`derive.py` 的常驻名单，**12 份里有 5 份的主干分母跟着变**，
`arch.vision.prefill` 的两格按新值重算过。

| id | 分母 GF/tok | ≈文本 tok | 占比 | 变动来源 |
|---|---|---|---|---|
| qwen3.8-flash-next | 114.47 → **12.07** | 454 → **564** | 1.029 → **1.279** | PLE 查表 512 亿从 `other`（计激活）移到 `retrieval`（不计） |
| step-3 | 74.65 → **73.99** | 481 → **484** | 2.846 → **2.864** | 两级 `vit_downsampler` 3.31 亿从 `other` 移到 `vision` |
| deepseek-v41-flash | 30.23 → **30.31** | 320 | 1.250 → 1.249 | `residual` 计进常驻（+0.26%） |
| ds-v4-flash-vision-exp | 25.42 → **25.49** | 332 | 1.296 → 1.295 | 同上（+0.27%） |
| kimi-k3 | 206.03 | 585 | 1.016 | 同上，+0.001%，取整后无变化 |

### `qwen3.8-flash-next`：错了 9.5 倍，结论整个反过来（该记录 R13）

旧分母 114.47 把 **PLE 查表的 51,200,245,795 个参数当成了常驻**——
`subsystems.py` 认不出 PLE、归进 `other`，而 `other` 在 `derive.py` 的常驻名单里。
修正后 `active_params.total` 从 578.7 亿掉到 **66.7 亿**，分母 12.07。
于是它从「四档里塔最轻、12 份里第 2 低」变成「四档里塔最**重**、12 份里第 8」，
塔占一张图 prefill 的 **22%** 而不是 3%。

两点方法上的话都写进了 R13：

- **这一条是上一轮自己预告的。** 当时把偏差写进 `open_questions`、
  并特意把 `prefill.derivation` 标成 `judgement` 让它进 Gate C，
  理由就是「正则修好后要重算」。偏差没被当成精修放过去，所以正则一落地就被抓回来 ——
  这是把已知不确定写成字段而不是埋在数里的直接收益。
- **但预告的方向写错了。** 当时写「114.47 是上界、454 是下界」，幅度估成「略」,
  实际 9.5 倍。根因是只看到「PLE 被错分」这个事实，没去看它占
  `active_params.total` 多少（512 亿 / 578.7 亿 = **88%**）。
  **知道一个因子错了，不等于知道它错多少** —— 下次报上界/下界要同时给幅度。

### `step-3`：`arch.vision.params` 现在有三路独立闭合（该记录 R24）

这是主会话点名想要的交叉验证，结论是**修正后的归类对了**：

1. 本轮为算 encoder FLOPs 自己从 `shapes.json` 重分区：
   encoder 4,285,292,032 + projector 390,082,560 = **4,675,374,592**；
2. 修正后 `derive.py` 的 `params_by_subsystem.vision` = **4,675,374,592**
   （修正前是 4,208,158,720，少 4.67 亿，还凭空多出 `draft` 1.36 亿 ——
   **本记录是没有 MTP 的模型，draft 桶本不该存在**）；
3. 厂商声明：总量 320,969,912,832 − 4,675,374,592 = **316,294,538,240**，
   对上模型卡的「Total Parameters (LLM) 316B」，总量对上「(VLM) 321B」。
   这一路完全不依赖本仓代码。

**一句方法上的话**：本轮做 encoder 参数量时**没有引 derived JSON 的 `vision` 桶、
而是自己从 `shapes.json` 重分区**，所以两边的差额才暴露出来。
图省事引那个桶的话，encoder FLOPs 会跟着少算 4.67 亿参数的塔，
而 YAML 里那个正确的 `params` 就成了孤证。

### 三桶巡检结果（主会话点名要的）

`_TOWER` 旁路只在阶跃四份上逐张量验过，风险面是「某一代的 MTP 块张量名里带
vision/vit/audio 字样」。12 份逐份比过 `draft` / `vision` / `other` 三桶：

- **只有 step-3 的三桶变了**（`vision` +4.67 亿、`draft` 与 `other` 归零），
  方向正确、且三路闭合。
- `qwen3.8-flash-next` 的 `other` 从 512 亿降到 3,280 万（PLE 移走），
  `draft` 2,607,150,848 **未变**。
- 其余 10 份三桶**一个字节没动**，包括 `draft` 非零的 5 份
  （qwen3.5-27b / qwen3.8-27b 各 424,699,392、qwen3.5-397b-a17b 6,595,568,128、
  deepseek-v41-flash 14,225,362,530、ds-v4-flash-vision-exp 19,845,851,751、
  step-5-preview 2,515,858,560）。
- **没有发现把真 MTP 张量误移出 `draft` 的情况** —— 即担心的风险面在这 12 份上没出现。

另报一处**不是我的改动范围、但顺手发现的**：`main` 里
`data/derived/deepseek-v41-flash.json` 相对它自己的 `derive.py` 是**旧的**——
重跑会新增 `cc4cf38` 引入的 `declared_tier: tech_report` 与
`declared_covers: per_token` 两个字段。我重跑后把它们带上了。
其余 11 份重跑只差时间戳。

## 一个踩到的坑：`.gitignore` 的 `facts/` 挡不住符号链接

worktree 里没有 `.venv` 和 `facts/`（都在 gitignore 里、不随分支走），
我按任务说明把它们从主 checkout 链过来 —— 用的是**符号链接**。
结果第一次 `git add -A` 就把这两个链接**提交进去了**，
而且一路带了 16 个提交才被发现。

根因：`.gitignore` 里写的是 `facts/` 和 `.venv/`，**带尾斜杠只匹配目录**，
而符号链接在 git 眼里不是目录，是一个 `120000` 的 blob。
提交进去的内容是指向我本地主 checkout 的**绝对路径**，对别人完全无效。

已用 `git filter-branch --index-filter` 从全部 17 个提交里清掉，
清后核对：全部提交里 `.venv` / `facts` 条目总数 = 0，
改动范围正好是预期的 27 个文件；12 份 derive 与 validate 重跑结论不变
（derived JSON 除时间戳外零差异）。
本地用 `.git/info/exclude` 挡住（不动共享的 `.gitignore`）。

**建议**：`.gitignore` 里把这两行改成不带尾斜杠的 `facts` 与 `.venv`，
或者各加一行无斜杠的。这样每个用 worktree 的会话都不会再踩 ——
我没动 `.gitignore`，它不在本轮的改动范围里。
拷贝而不是链接也能绕开，但 `facts/` 有好几个 GB。

## 建议主会话接着做什么

按价值排序：

1. ~~`qwen3.8-flash-next` 的 `prefill` 两格要重算~~ —— **已做**，
   见「并 main 之后的修正」。`subsystems.py` 的正则在 `e394525` 修好后
   立刻重算了，结论反转（1.029 → 1.279）。
2. **站点还没有渲染这三个子树。** `site/src/lib/data.ts:417` 的 `vision` 这一行
   只取 `layers` 与 `hidden`。新字段 `validate.py` 全覆盖、`npm run build`
   也过（59 页不变），但页面上看不到。
   最有价值的一块是**跨模型的「同分辨率下一张图几个 token」对照表**——
   口径护栏（固定分辨率不可换算）得一起渲染，否则表会误导。
   `site/` 不在我的改动范围里，没动。
3. **按 `omni-audio-field-gap.md` 的落地顺序录第一份 Omni**（形态 A），
   先用现有 5 格填、缺口进 `review`，再改 schema。
4. **encoder 注意力跨度专项。** 12 份里 11 份没有一手来源，
   而它在 kimi-vl-a3b（线性项的 35%）与 minimax-m3（分块与否差 4–36 倍）
   上不可忽略。读参考实现的视觉塔部分就能收一批 ——
   注意 `model-audit` 规则 10：这一族的 `inference/` 常自称 demo，
   能回答布局与分段，但不能推翻卡上的数。
5. **`separability.runtime_support` 专项**（可以和上一条并一轮做）：
   哪几家的部署文档/参考实现真的支持 encoder 独立成服务。
   现在 12 份全空，而 `shard_isolation` 已经告诉我们
   kimi-k2.5 / kimi-k3 / step-5-preview 三份在**权重落盘层面**最容易拆。
6. **`kv_table.py` 不用跑** —— 本轮没碰任何缓存字段。
