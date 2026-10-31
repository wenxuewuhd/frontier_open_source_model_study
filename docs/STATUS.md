# 当前状态与待办

> 2026-10-30 盘点。这份是活文档，供接手的人（或压缩上下文之后的自己）一眼看清
> 做到哪了、什么没做、哪些坑已经踩过。数字都是盘点当天实测的，不是估的。

## 一眼看清

```
记录            54 份     六个语言模型家族 + M3.1-Flash 支线 + **视频生成 6 份**
视频线          6 / 12    wan ×3、minimax-h3、ltx ×2；hunyuan-1.5 在录
逐层产物        6 份      六个旗舰全有
注意力架构图    5 张      差 minimax-m3 一张
闸口            六道      全绿（validate 0 错 0 警 / 回归 39 / kv_table / 站点 70 页 /
                          check_rendered / **字段消费点棘轮**）
```

**验闸口要在临时 worktree 上跑。** 主 checkout 同时有好几个会话在写未提交的东西，
直接跑五道闸全是别人半成品的噪声，会把真错盖住（已经发生过两次，我因此误判过一次归因）：

```bash
R=$PWD; T=<scratch>/verify
git worktree add --detach $T HEAD -q
ln -s $R/.venv $T/.venv; ln -s $R/facts $T/facts; ln -s $R/site/node_modules $T/site/node_modules
(cd $T && python scripts/validate.py; ...; cd site && npx astro build)
git worktree remove --force $T
```

软链要用**绝对路径**（worktree 在 scratchpad 里，不是仓的兄弟目录），
`site/node_modules` 那个尤其容易错：相对目标按软链自己所在目录解析。

**告警归零了**（原先那 2 条 basis 标签表的告警见下面「做完了的」最后一行）。

## 做完了的（别重做）

| | 结果 |
|---|---|
| **P0 旗舰** | 六家族旗舰 + 注意力演进专题 + 站点部署 |
| **P1 历史回溯** | 五族共 32 份历史记录，15 → 47 份 |
| **三道人工闸** | 47 份全部过 Gate A/B/C |
| **KV 落盘精度复核** | 五族逐份查过卡与技术报告，**DeepSeek 之外零命中**，详见 `glm-kv-declared-sources.md` |
| **`weights_release` 口径统一** | 40 份补 `weights_release_scope`，闸口从 warning 收紧成 error |
| **缺值三态** | 32 列 + 逐子系统模板声明「空格子是哪一种空」，六个家族页裸 `—` 从 524 → **0** |
| **散文渲染** | 行内 Markdown，字面 `**` 从 3,665 → **0**，加第一道产物侧闸口 |
| **理解类多模态** | 12 份 `kind: vlm` 全部补 `tokens` / `prefill` / `separability` 三组，共 22 格 |
| **视频生成 Gate A** | 12 份记录 / 8 个 family 定稿，schema 与 derive 的生成类分流已落地 |
| **basis 标签表收口（原 B3）** | 两份表措辞本来就不一样（「按厂商公布值」vs「厂商公布」），收到 `data.ts` 一处，闸口从 warning 收紧成 **error**，并补了一条探针用例确认它真会红 |
| **E2 脚手架** | `styles/attn-figure.css` + `AttentionFigure.astro` 分发器 + DeepSeek 参照图两张。**加一张图只要新建 `attention/<id>.astro`**，不改分发器也不改 `[id].astro` |
| **E1 逐层产物** | `extract_layers.py` 推广到六族，五个旗舰补齐 `.layers.json`，deepseek 那份重跑逐字节相同 |
| **E2 四张图** | glm-5.3 / kimi-k3 / qwen3.8-2.4t-a95b / step-5-preview，四个家族会话各画各的，**没有一张套模板** |
| **视频线 C1 / C2** | 取数认四种布局；**拒绝把几个模型合并成一个**（LTX-2 实测 7,132 个同名张量跨文件 dtype 不同，旧代码按线程完成顺序静默覆盖） |
| **视频线 policy 与名册** | 8 个 family / 12 份进 `tracking.yaml`，与语言模型那条并列 |
| **首录 wan2.2-ti2v-5b** | 本仓第一份 `kind: video-gen`，Gate B 两路 0 处不一致。三条口径判断定在这份记录的头注释里 |
| **`weights_release_scope` 拆字段** | `measured` 一个布尔压着「取没取过数」与「进没进精度表」两件事，M3-MXFP8 只能填 false，**字段在说反话**。拆成 `manifest_taken` / `in_precision_table`，42 份 84 个条目，四个负例实跑都红 |
| **视频线首批 6 份** | wan ×3、minimax-h3、ltx ×2。Gate B 两路对账全部 0 处不一致 |
| **生成负载进页面** | derive 一直在算 `generation_workload`，页面一个字没读。序列长度 / 单步 FLOPs 拆分 / 逐因子推导表 / 两个时间压缩率现在都看得见 |
| **`latent_frames` 声明优先** | H3 是 17n+5（厂商 VAE 实现 + 14 个官方样片验），公式给 61.5、真值 72。token 低 17% 经 O(n²) 放大成注意力低 31% |
| **读 config 收口** | `lib/cfgio.py` 一个入口。量化声明**跟着模态走**：纯文本在顶层、VLM 嵌 `text_config` 下，所以只读顶层的新代码每多一份 VLM 就多漏一份 |
| **第六道闸：字段消费点棘轮** | schema 的字段站点上得有人读。当前欠 45 个（基本是整个 `generation` 块）。**这个数是低估**，判据会被字符串里的同名词顶掉，见 C8 |
| **名册对账（`check_roster`）** | `tracking.yaml` 的 `video_families` 陈旧过一整条线：九条 `status: planned` 对着已落地的记录，外加两条 `released` 口径错、一条写下时就错的 note。**原因是没有消费点** —— 站点只读 `families`，那张表没人看过。现在站点两张都读（总览与家族演进按线分表），status / released 按记录对账，分线判据再与记录有没有 `generation` 块互核 |
| **生成线逐子系统精度（九份）** | 此前 `precision.quantization` 在 48 份语言模型记录里全有、九份生成记录里一份都没有，`--list-gaps` 的「未录」名单 12 份。事实其实在手（`facts/*/shapes.json` 每个张量都带 dtype），是填的动作没做。六个 family session 各填一族，名单降到 3 份 |
| **子系统词表加生成三桶** | `modulation` / `time_embed` / `io_proj`，并给 `dense_ffn` 补 `(?:^\|\.)ff\.` —— **f-f 不是 ffn 的子串**，五份记录的整个 FFN 原先在 `other` 里。h3 的 `other` 从 75.7% 降到 0。三条新词条一律排在已有规则之后：只许从 `other` 里捞，排前面会从 `vision` / `audio` / `dense_ffn` 抢走 13 份语言模型记录与 LTX 的张量（实测过） |
| **cogvideox 的 AdaLN 在错桶里** | 它把调制实现成 `normN.linear`，名字带 norm 就被 `norm` 规则领走：**7.94 亿参数（14.26%）标成「归一化」，而这一份的全部意义就是它是「逐层放矩阵」那组的代表**。`modulation` 因此在 `SUBSYSTEM` 里出现两次 —— 这条排 `norm` 之前，尾部那条不能上移 |
| **M3.1-Flash（原 B1）** | 录为支线。顺带修掉三个页面早就印着的字面量 `undefined`（五道闸当时全绿） |
| **LayerMap 三修** | 零 MTP 块的轨道错位（`repeat(0,…)` 被钳成 1）、`DT` 补 `MXFP4`、**`recurrent_state` 进表**。第三条原被当成「图的口径限制」，实际是 kimi 93 层里 69 层的缓存去向在表上不存在、而每 token 合计看着像覆盖全部层 |

## 待办

### A. 立刻能做，无依赖

- ~~A1 `p4-meaning` 作废分支清理~~ —— **已拆**（分支与 worktree 都没了；
  它唯一的未提交项是个 `node_modules` 软链，没有内容）。
- **A2 已并分支与 worktree 清理** —— **现在可以拆了**：E2 四条与 p1-minimax 都已并入，
  九条 `p1-*` / `p2-video-gate-a` / `p3-vlm-infra` / `p4-units` / `p4-fieldstate`
  全部领先 main 0，连同它们的 worktree 一起拆。
  **拆之前先查每个 worktree 的 `facts/`**：它们各有一套独立的（不是软链），
  里头可能囤着主库没有的实测清单。这一轮查出 `wt-p1-minimax` 独有两份 ——
  `minimax-m3-mxfp8`（45,838 条）与 **`minimax-m3.1-flash`（89,729 条，
  已发布记录的清单来源）**。照单拆会让那条记录无法复算。两份已拷进主库 `facts/`。
- **A3 `--ep` 没有记录在任何地方** —— `derive.py` 默认 1，而已提交的 derived 里有 1 / 4 / 8 三种，
  重跑全靠人记得读 HEAD。**一天内踩了四次**（我自己两次把别人的值改掉，
  p1-minimax 又发现 minimax-m3.json 的 ep_size 被默认值 1 覆盖过、已恢复成 4）。
  修法：把 ep 记进记录或 derived 的输入段，让重跑不依赖记忆。

### B. 排队中

- ~~B1 M3.1-Flash 录入~~ —— **已录并并入**（4 个提交）。**我给的前提有一条是反的**：
  说「MTP 被去掉了」，实际 `dspark/` 子目录里有 97 条张量、1,958,493,697 参数的
  DSpark 草稿模块，这次真发了权重。派活时给的背景也要当待验证的东西，不是事实。
- **B2 MiniMax M2.1 / M2.5 / M2.7 三个仓本仓一份没录** —— **M2.5 下载量 1,392,282，是旗舰 M3 的 7 倍**。
  需 `p1-minimax` 回看自己 Gate A 的排除依据还成不成立。
- ~~B3 站点侧两份 basis 标签表~~ —— **已收口**，见上面「做完了的」。
- **B6 minimax-m3 记录里有一句现在是错的** —— 第 360 行「MXFP8 变体**本仓没有取过它的清单**」。
  清单其实取了（45,838 条，原先搁在 `wt-p1-minimax/facts/`，已拷进主库）。
  粗看 dtype 分布：`F8_E4M3` 22,422 + `U8` 22,422（1:1，U8 是 e8m0 scale 容器）
  + `BF16` 879 + `F32` 115。所以那句不只是过时，它挡着一件能实测的事。
  归 `p1-minimax`，它取的数它最清楚。
- **B4 音频 / Omni** —— `docs/omni-audio-field-gap.md` 已给出需求与落地顺序，字段还没建。归 `mm-infra`。
- **B5 两道闸口的设计题（p1-minimax 留给主会话的）** —— 两件事同一个形状：
  **产物错了而五道闸全绿**，而且补闸的难点都是「怎么分开『印出了 X』与『在讲 X』」。
  1. **字面量 `undefined`**：`check_rendered` 不查它。p1-minimax 加过判据、拿注入负例
     验过会报，但它随即在讲这个 bug 的核查散文上误报，已回退。
  2. **落单的 `**`**：`check_rendered` 的 docstring 第 22 行**有意豁免**
     （`head_dim**-0.5` 是幂运算）。而一条散文里写裸 `**` 会吃掉后面的强调配对，
     正好只留一个落单的 `**` —— 落在豁免里。M3.1-Flash 的记录就栽在这上面，
     **它讲的恰恰是这类缺陷**。
  可能的判据：先剔掉 `\w\*\*-?\d` 这种幂运算写法，再按散文字段数 `**` 的奇偶。
  没验过，只是思路。

### C. 视频生成线（Gate A 已完，step 1 做了一半）

- ~~C1 取不到 config~~ / ~~C2 多 index 仓的分桶与静默覆盖~~ —— **都已修**，见上面「做完了的」。
  原文里「Diffusers 布局没有顶层 config.json」这个前提**不普遍成立**：实扫 12 个仓是四种布局。
- **C5 两份仓里没有 safetensors** —— `tencent/HunyuanVideo` 是 4 个 `.pt`、
  `SANA-Video_2.0_5B_720p` 是单个 `.pth`，读头拿清单这条路不成立。
  已在 tracking.yaml 标 `blocked-no-safetensors`。
  **录它们之前要先定「没有清单的记录怎么过 Gate B」**。
  而且这不是边缘情况：首录 wan2.2-ti2v-5b 的 VAE 与文本编码器也都是 `.pth`，
  **DiT 在 safetensors、VAE 与文本编码器是 pickle，是这条线的通例**。
- **C6 usedStorage 在这条线上不能直接当权重总字节** —— wan2.2-ti2v-5b 实测差
  19,999,532,688 B，约等于多算了一份 DiT（仓有两次 `Upload folder`，
  旧修订版的 LFS 对象仍计入）。要逐文件取 HTTP 大小。
- ~~C1 原文~~ `fetch_hf.py` 对生成类仓取不到 config —— Diffusers 布局没有顶层 `config.json`，
  配置散在 `transformer/` `vae/` `audio_vae/` 等各组件目录里。不加按组件取 config 的模式，
  **12 份视频记录全部没有 config 侧，Gate B 的双路对账只剩一路**。
- **C2 `component()` 按顶层前缀分桶在多 index 仓上不成立** —— LTX-2.5 有 7 个独立 index，
  名字跨组件重叠（`transformer` ∩ `transformer_full` 4091 个同名）。组件维度得来自
  分片路径 / `weight_map`，而那个信息现在没进 `shapes.json`。
  顺带：**扁平字典装同名张量会静默覆盖**，这次因为两份是相同副本所以没丢数，是运气。
- **C7 残差与 Q 低秩两列共 52 格说「未录」** —— 残差 39 格、Q 低秩 13 格。
  **不能靠 derived 自动判**：清单里没有残差张量推不出「普通残差相加」，
  MiniMax 那两份就是没有权重的非常规残差（系数是 config 标量）。
  判据要两路合一（清单无张量 **且** config 无残差系数键），而规则 DSL
  读不到 config（`src` 只有 rec / derived）。Q 低秩那 13 份建议逐族系确认。
- **C8 整个 `generation` 块没有消费点** —— 第六道闸报的 45 个欠账基本都在这里：
  VAE 规格、音频、文本编码器、采样口径、尺寸约束、checkpoint 矩阵全录了，
  页面上只渲染了由它们**推出来的**负载。还掉之后把棘轮收紧。
  **而这 45 个是低估。** 棘轮按叶字段名 grep 标识符，正则不分代码与字符串字面量，
  于是别处的同名词会把字段顶成「已读」：`generation.text_encoders` 整块零消费点，
  13 个叶字段里七个（role / name / hf_repo / params / hidden / note / gated）
  是被撞出来的「已读」。诚实的数是 51 左右。`_MASKED_UNREAD` 现在只钉住 `gated`
  一个（名册状态标签 `blocked-gated` 一分词就撞上它，是本仓代码自己造成的），
  其余六个要连同预算一起改 —— **把预算从 45 调到 51 看起来像放松，实际是把量测改准**，
  所以单独做、单独说明。先还 `text_encoders` 的消费点更划算：那一块本身就该渲染
  （「部署时要不要额外塞一个大模型进显存」「拿不拿得到」）。
- **C3 第一批三份录入** —— `wan2.2-ti2v-5b` → `minimax-h3` → `magi-2-preview`，第三份依赖 A3b。
- **C4 视频线的 policy 与语言模型线不同** —— 「每个 family 追最新旗舰 + 一个差分基线；
  VAE 或注意力形态变了的同代档位单独录」。要写进 `tracking.yaml` 的 policy，
  这是本仓第一次出现两套选题标准。

### E. 旗舰模型的两张图（用户 2026-10-30 点名）

参照物是 `deepseek-v41-flash` 模型页现有的逐层实现图，以及那份「V4.1 注意力分层图」
里的**图 1 生产侧 / 图 2 消费侧**。两件事分开做，第二件不依赖第一件。

- ~~E1 另外五个旗舰补逐层实现产物~~ —— **已做完并并入**（`e306638`）。
  六个旗舰现在都有 `.layers.json`，deepseek 那份重跑逐字节相同。
  值得记的一条：第一次试跑传了 `--facts-dir` 绝对路径，产物的 `inputs.shapes`
  就记成绝对路径、和 HEAD 对不上 —— **不是内容差异，是记录输入路径的那一行**。


- **E2 注意力架构图** —— **五张已落地**（deepseek 参照 + glm-5.3 / kimi-k3 /
  qwen3.8-2.4t-a95b / step-5-preview）。**差 `minimax-m3` 一张**，`p1-minimax`
  现在空出来了，可以派。四张都没套模板，各自的引导句：
  GLM「index K 与 MLA latent 不同源」、Kimi「两路存的不是同一种东西，也不在同一个刻度上」、
  Qwen「两类缓存对象在 3,192 个 token 处交叉」、Step「三层滑窗配一层全局，
  于是全部的缓存增长都压在那 1/4 的层上」。

  **约定不在这份文档里，在两份能跑的代码里** —— 写成散文会被重新发明，写成代码不会：
  - `site/src/styles/attn-figure.css` —— 四套语义配色与全部类名。
    颜色另起 `--fig-*`，**不复用 `--official` / `--derived` / `--judgement`**：
    那三个在站内已表示证据档位，图里再拿来表示「KV」「压缩」就是同色两义，
    正是 B3 修掉的那个毛病。
  - `site/src/components/attention/deepseek-v41-flash.astro` —— 参照实现。
  - `AttentionFigure.astro` 按记录 id 查文件名分发，**新建一个文件就自动出现在模型页**，
    不改分发器也不改 `[id].astro`。这是有意的：共享文件上并写必撞。

  下面这份约定保留做速查，但以上面两份代码为准：
  - **内联 SVG**，无 JS；`viewBox` 约 940×330–380，`width:100%` + `min-width:760px`，
    外层 `.figbox` 允许横向滚动
  - **横向分带**，每带一个 `①②③` 编号标题（`.lbl-s`），带与带之间用虚线
    （`.arr-d`）分隔。生产侧三带：query / 滑窗 KV / 压缩 KV；消费侧两带：indexer / 稀疏注意力
  - **四套语义配色**：teal = KV 与 cache 数据流、ochre = 压缩与打分、
    violet = indexer 与 index key、rose = 候选与 mask；中性灰框是普通算子。
    盒子 `.box/.box-t/.box-o/.box-v/.box-r`，箭头 `.arr/.arr-t/.arr-v/.arr-d`
  - **每个盒子都带量**：投影形状（`wkv 5120→512`）、头数与维度（`64×512`、`32×128`）、
    dtype 与块（`FP4 b16+e4m3`、`MXFP4 b32+e8m0`）、缓存宽度（`512 维正文` / `128 维摘要`）、
    条目字节（`512 条 × 288 B`）。**没有量的框不要画**，那就退化成示意图了
  - **`<figcaption>` 用加粗引导句说出这张图唯一最该记住的那件事**
    （图 1 是「index_k 与 compress_kv 同源于 latent」，图 2 是「indexer 只产索引，
    真正的 KV 数据流在第 ⑤ 带」）
  - 每张图 `role="img"` + 一句 `aria-label` 概括全图；marker id 每张唯一（`pa/pb`、`qa/qb`）
  - 深浅色都要能看：颜色走 `:root` 的 CSS 变量，`prefers-color-scheme` 与
    `[data-theme]` 两处都覆盖

  **这六张图的内容差异极大**，别套模板：MLA+DSA（glm-5.3）、
  MSA block-sparse（minimax-m3）、KDA+MLA 混合（kimi-k3）、
  SWA/global 3:1 + DSA（step-5-preview）、混合线性注意力（qwen3.8-2.4t-a95b）
  各自的「谁产谁用」结构根本不同 —— 这正是画图的价值所在。
  数据来源用 E1 的 `.layers.json` 加各自记录的 `arch.attention`，**不要凭印象画**。

### D. 已知缺陷，记着没修

- **D1 `derive.py` 的算术强度按参数量比例折算 decode 权重字节** —— 对混合精度模型不成立
  （kimi-k3 偏 2.34×）。
- **D8 组件层的桶在逐子系统精度表上没有列** —— `classify2()` 是两层的：
  非 `SPLIT_INSIDE` 的组件（vae / connector / text_encoder / vocoder / audio_vae / upsampler）
  整块算一桶，桶名就是组件名。而逐子系统精度表的列来自 `data/schema/subsystems.json`，
  那里**没有任何组件键** —— 于是这些桶的精度在那张表上无处可放。
  当前只有 ltx-2.3 一份踩到：5 个组件桶共 **4,080,667,243 参数（占它 23.07B 的 17.7%）**
  在表上不可见（audio_vae / connector / text_encoder / vae / vocoder）。
  另八份生成记录的取数都收窄到单个组件（`--include 'transformer/*'` 之类），所以无列桶为 0。
  模型页的「按子系统」表直接渲染 derived 的桶，**那里看得见**；缺列的是家族页与精度演进页
  那张逐子系统**精度**表。不是填表能解决的，要么给组件立列、要么说清这张表只覆盖主干。
- **D7 逐子系统精度表的行不带来源维度** —— 每行隐含「测自 `hf.repo`」
  （`columns.json` 的 `$subsystem` 模板里 `scope_from: hf.repo`）。
  于是厂商另发的量化变体（M3 的 MXFP8，已实测 45,838 条）**进不了那张表** ——
  只能在 `weights_release_scope` 里说一句。
  **别问「一份记录能不能带两套精度表」，那个问法会引出「比哪张」的死胡同**；
  该问的是「那张表的行要不要带来源维度」。带了就能共存于一张表而不混。
  要改 columns.json 的模板、站点渲染、以及 42 份的隐含口径，不是顺手的事。
- **D5 `kv_dtype` 那一格不走 Cell** —— 它的取值器 `kvCachePrecision()` 把**缺值标签当成值**
  返回，于是 `cell()` 看到非空字符串直接当文本渲染，那一格丢掉 `absent` 类与解释用的
  `title`。**三态契约的招牌字段反而不按契约渲染**，49 份全部如此。既有缺陷，不是视频线引入的。
- **D6 `[family].astro` 的 `norm_eps` 是手写行** —— 绕过 Cell，硬编码
  `<span class="dash">—</span>`。它不是 TRACKED_FIELD、也不在 columns.json，
  所以「key 必须一一对应」那道闸查不到它。家族页上唯一一个裸 `—` 就是它。
- **D3 `derive.py` 没法把「已查清的口径差」标成良性** —— step-5-preview 的外部参数量对账
  永远报 MISMATCH，而记录第 106–114 行早已查清：HF 的 tally 含两份 index 外的同名副本
  （`vit.safetensors`、`model-mtp3-00001`），是口径差不是算错。
  **每次重跑都红一次而每次都该忽略，这会训练人不读红字。**
  要么让记录能声明「这条 MISMATCH 已解释，依据在 X」，要么对账时认得 index 外副本。
- **D4 `derive.py` 的 CLI 默认值会静默改坏产物** —— 不只是 `--ep`（A3），`--model` 也一样：
  不给就把 `inputs.model_yaml` 记成 `null`，并丢掉 `attention_type`、`status: measured`。
  两个都是「不传就得到一个看起来正常的错产物」。A3 的修法要把两个一起管。
- **D2 选题漂移没有任何闸口查** —— 五道闸全在查已录记录的内部一致性，
  没有一道问「该录的录全了没有」。B2 就是实例。

## 踩过的坑（重复出现的才列）

1. **`--ep` 不在任何地方记着** —— 见 A3。**一天五次**（我自己三次）。
   而且不只 `--ep`：`--model` 不给会把 `inputs.model_yaml` 记成 null 并丢掉
   `attention_type`，同样是「不传就得到一个看起来正常的错产物」。见 D4。
2. **用名字当锚，锚到别处同名词** —— 三次：抠键锚到注释、证据行锚到 `platforms` 里的
   `precision: FP8_E4M3`、层号捷径咬到视觉塔。判据要按结构不按词。
3. **worktree + 符号链接** —— 四次：`facts` 不随分支走；`.gitignore` 带尾斜杠挡不住同名软链
   （已修，去掉尾斜杠）；`site/node_modules` 软链会让 dev server 服务另一个 checkout 的代码；
   **第四次是反向的** —— 有些 worktree 的 `facts/` 是独立目录而不是软链，于是在那边取的
   清单**永远到不了主库**，而且因为 `facts` 在 gitignore 里，git 完全看不见这件事。
   拆 worktree 等于删数据。先 `comm -23` 对一遍再拆。
4. **闸口的盲区** —— 见 `.claude/skills/model-audit/SKILL.md` 规则 11。
   锚在内侧只守得住内侧；闸全锚在数据和代码上时，产物可以三千多处出错而全绿。
5. **生成物冲突不要手合** —— 从合并后的源重算。
6. **未跟踪文件会卡住 merge** —— git 直接拒绝（「会被覆盖」），平时看不见，只在合并那刻发作。
7. **一个字段说反话，页面会照着它说反话** —— `weights_release_scope.measured` 压了两件事，
   M3-MXFP8「取了清单、没进精度表」只能填 `false`，于是 `precision.astro` 把它渲染成
   「未取」、tooltip 写「厂商发了，本仓没取清单」。**读者看到的是一句通顺的、具体的、错的话**，
   比字段本身错更难发现。写字段的人还照着那个 false 复述过一次。
   **一个布尔只装一件事**；装两件时，第三方消费点会替你选一个含义，而且多半选错。
8. **估出来的落地面不要当范围说出口** —— 这次估「影响 10 份记录」（按带 note 的那些数），
   实际是 **42 份 81 个条目**；估「要动三处」，实际四处（漏了 `precision.astro`）。
   漏掉的那一处恰恰是错得最显眼的那一处。
9. **核查编号各分支自己递增，分支一多必撞** —— 两次：R8、R9 都是两条分支各加一条、
   git 自动合并不报冲突（两段文本不相邻），靠 `validate` 的「编号重复」才抓到。
   **合完一条带 `review:` 改动的分支，一定跑一遍 validate**，别只看 merge 的退出码。
10. **删一处重复，要同时退掉盯着它的所有闸** —— basis 那份重复表被**三处**盯着：
   `validate.py` 的 stray-label 检查、回归用例、以及 `kv_table.py` 里一道
   「两份表不许漂」的守卫。只退了前两处，`kv_table --check` 就红了。
   更要紧的是那道守卫**写明「只对键，不对值」**，注释里还列着两处不同的措辞 ——
   **它一直知道两份措辞不一样，并且按设计放过了**。防「少一个键」的闸，
   防不了「同一个取值两种读法」。
11. **改完要五道闸一起跑** —— B3 那个提交只跑了四道（漏了 `kv_table --check`），
   带着一道红闸进了 main，下一次全跑才发现。五道的名字写在本文件开头，照着跑。
12. **散文里写裸 `**` 会吃掉后面的强调配对** —— M3.1-Flash 的核查里写了一句
    「只查没解开的 ** 与反引号」，那个 `**` 开了一段强调，一路吃到两行之后的
    `**我没碰过的 kimi-k3**`，产物上剩一个落单的 `**`。**一条讲「字面 `**` 是缺陷」
    的记录自己制造了一个。** 要写字面量就用行内代码包起来。
    闸口按设计放过它（见 B5）。
13. **派活时给的背景也要当待验证的东西** —— 我给 p1-minimax 的三条前提里有一条是反的
    （说 M3.1-Flash 的 MTP 被去掉，实际发了 1.96B 的 DSpark 草稿模块）。
    它查出来并顶了回来，这是对的。**brief 里的「已知」不比记录里的断言更可信。**
14. **字段加了、没人读** —— **一天四次**：`weights_release_scope` 一处不渲染、
    `bottlenecks.fix` 不走 Prose、`generation_workload` 整块没消费点、
    `group_size` 加完没有列。**第四个是我诊断完前三个之后自己造的。**
    加字段时只想「schema 收得下」，没问「谁读它」。
    已做成第六道闸（棘轮），但闸只挡新增，存量 45 个要人去还 —— 而那 45 个本身是低估。
15. **断言「A 当且仅当 B」只验了一个方向** —— p1-kimi 查「有 derived 残差信号的是不是
    都填了字段」（7/7 成立）就报「两边 100% 一致、零例外」，反方向没查，
    而反例（填了字段却无张量的两份 minimax）**就在它自己上一条命令的输出里**。
    那两份是「没有权重的非常规残差」，照那条规则会把 39 格标成「确认无」。
    **可执行的办法**：`sorted(A - B)` 与 `sorted(B - A)` 两行都跑、都打印，缺一行不许说「零例外」。
16. **写死名单的回归用例** —— **一天三条**（KV 两条 + 「声明了而清单里没有」那条）。
    每进一份新记录就红，而每次只能把数字或 id 加进去，用例什么也没守住。
    **断言规律，不要断言人口普查**：按 kind / 按结构推期望，名单变了用例不用改。
17. **「唯一 / 第一 / 零例外」这类断言有保质期** —— 两类要分开：
    **过期**（写时对，后续记录让它失效，如我那句「全 F32 全仓独一份」）与
    **当时就没核**（手边有全量数据却没去数，如 Qwen 的「INT4 首发」、
    Gate A 的「Wan 是唯一宽松许可」）。后者更该防，它不需要任何事情发生就已经是错的。
    写这类句子一律带**作用域与时点**。
18. **小 diff 不等于对的 diff** —— 我拒绝 `json.dumps` 回写（会重排 1016 行），
    改用定点插入得到 12 行，**结果八条里五条插到了别的列上**。固定窗口越过了列边界，
    而没有任何闸口比对「规则的理由属不属于这一列」。
    改用花括号配对定位，并**读回来双向验**（每列有自己的、且没有别列的）。
19. **拿过期的状态去断言别人手上的情况** —— 我照一份旧的 `git status` 告诉 v-wan
    「闸口失败是你那份半成品造成的」，而它早已提交并修好。同一轮还把一个
    `preview_start` 的 cwd 问题归因成软链断了（纯猜，它用 lsof 坐实了真因）。
    **别人手上的状态要现查，或者说清「截至我看到的那一刻」。**
20. **脚本「什么都没产出」却返回 0** —— `extract_layers.py` 对五个旗舰各产出 0 层，
   只打一行 `!!` 就算成功（见 E1）。坑 #4 的同一张脸：**闸口和脚本都要能红**。
   新写一道检查时，先故意喂它一个坏输入，确认它真会非零退出 —— 这回给
   `check_stray_labels` 补的探针用例就是干这个的。
21. **把「我没有能看见的列」当成「它没变」** —— 家族页的「字段零变化」段原文写着
   「相对前代连一个规格字段都没动，**能力变化来自后训练或权重刷新**，算子清单可直接
   沿用前代」。这是从缺值推出的结论，而三态契约的全部意义就是不许这么推。
   wan2.2-T2V-A14B 在 48 个可比字段里只有 **6 个**两侧都有值，真正变了的那个量
   （整模型参数量 14.3B → 28.6B，两套专家按时间步切换）**没有列** ——
   于是一个双模型架构被写成了权重刷新。ltx-2.3 同形。
   **而这不是生成线独有的**：glm-4.7 比过 24 / 48 个字段、也落进「零变化」，
   可它与 glm-4.6 的 `params.total` 差 1,551,892,480。`params` 根本不是跟踪字段。
   修法不是加一句免责 —— 是**把对比口径和结论一起报**：现在每个版本后面带
   「比过 N / M 个字段」，一个字段都没比过的不进这一段，
   而「矩阵看不见什么」按线取词（生成线提 VAE 与采样步数，语言模型线提权重存储口径），
   否则免责声明自己就成了另一条悬空断言。
22. **共用工作区里 `git add <paths>` 不够，`git commit` 也必须带 pathspec** ——
   六个 family session 共用同一个 checkout 时撞上：cogvideo 那个 session 的 add 写的是
   两个具体文件路径（没有目录、没有通配、没有 `-A`），commit 也没有 `-a` ——
   但 **commit 没带 pathspec，提交的是整个 index**，而 wan 那个 session 在它 add 之后、
   commit 之前把自己的三份记录 add 进了同一个 index。于是 `048954f` 的说明只写
   cogvideox，内容里混进了 wan 三份。内容没丢（wan 补了空提交 `0de757c` 留痕），
   但历史上那个提交说的不是它做的事。**「只 add 指定路径」这条规矩只管住了一半。**
   正确做法是 `git commit -- <paths>`，或者先 `git stash` 别人的改动 —— 而后者在
   共用工作区里恰恰是**禁止**的：`git stash` 会把别的 session 未提交的改动一起藏走
   （我自己为了做棘轮基线对比踩过一次，7 个文件被藏，pop 回来才没出事）。
   要做基线对比就另开 `git worktree`，或者在内存里算。
   **第三面：共用 checkout 里只有一份历史，所以任何一个 session 的 `git push`
   会把别人「已提交未推送」的工作一起推出去。** v2-hy15 的 `74ee4e0` 就是这么搭着我的
   `66b3b18` 出去的 —— 它以为自己还控制着什么时候公开，其实不控制。
   好的一面是它因此也进了我那次干净 worktree 的六道闸复验（`74ee4e0` 是 `66b3b18` 的父提交）；
   坏的一面是**反过来也成立：没准备好的提交会被别人推走**。
   所以共用 checkout 里「本地提交、不要 push」这条约束只约束得住主动推送的人，
   约束不住被动带出去 —— 要真的不出去，提交前就得确认它已经可以出去。
23. **锚不住就要说出来，不能静默通过** —— validate 的「`resolution: open` 必须有
   对应 open_question」那道检查，判据是 `field.split(".")[-1] in open_q`，
   而 `field: '-'` 时 key 就是 `-`，**一个「——」就有两个破折号**，于是 `in` 恒真。
   stepvideo 的 R4 挂了两轮 `open` 而这道检查从没管过它。坑 #2 的又一例，
   但这回撞的不是另一个同名词，是散文里到处都有的标点 —— 判据按结构不按词，
   「词」里也包括标点。改成锚不住就明说「这一条要人看」之后，
   **立刻报出 4 条此前不可见的条目**（cogvideox 3 条、hunyuanvideo-1.5 1 条）。

## 会话地图

| 会话 | 分支 | 状态 |
|---|---|---|
| `p1-minimax` | `p1-minimax` | B1 已收工并入；**B2 还没做，minimax-m3 那张图也还没派** |
| `p1-glm` `p1-qwen` `p1-stepfun` | 同名 | 已收工并入 |
| `p1-kimi` | `p1-kimi` | 已收工并入（量化作用域专题、三态诊断） |
| `v-wan` `v-ltx` | 同名 | 已收工并入（wan ×3 / ltx ×2） |
| `gatea-video` | `v-minimax-h3` | **在录 hunyuan-video**；minimax-h3 已并 |
| `e1-layers` | `main`（共用主 checkout，没单开 worktree） | 已收工（E1 直接提交在 main） |
| `mm-infra` | `p3-vlm-infra` | 已收工并入，B4 归它 |
| `presentation` | `p4-units` | 已收工并入 |
| `p4-render-gate` | `p4-fieldstate` | 已收工并入（B3 已由主会话做掉） |
| `gatea-video` | `p2-video-gate-a` | 已收工并入，C 组归它 |

> 多会话共用一个 checkout 时，分工按**路径**切，并明说「不许切分支 / stash /
> `checkout -- .`」。这一轮 `e1-layers` 和主会话同在 `main` 上写，没出事。
