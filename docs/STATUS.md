# 当前状态与待办

> 2026-10-30 盘点。这份是活文档，供接手的人（或压缩上下文之后的自己）一眼看清
> 做到哪了、什么没做、哪些坑已经踩过。数字都是盘点当天实测的，不是估的。

## 一眼看清

```
main            2f262ec   已推，未推 0
记录            47 份     六个语言模型家族，2024 起的历史代际已补齐
闸口            五道      全绿（validate 0 错 2 警 / 回归 26 / kv_table / 站点 59 页 / check_rendered）
```

`validate` 那 2 条告警是**设计内的**：`data.ts` 与 `[id].astro` 各有一份 basis 标签表，
等站点侧统一入口收口。详见下面「待办 · B3」。

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

## 待办

### A. 立刻能做，无依赖

- **A1 `p4-meaning` 作废分支清理** —— 领先 main 4 个提交、1,910 行，已判整条不并
  （四个文件两留两丢，留的两个已以重写版落地）。连 `wt-p4-meaning` 一起拆。
- **A2 九条已并分支与 worktree 清理** —— `p1-*` 五条、`p2-video-gate-a`、`p3-vlm-infra`、
  `p4-units`、`p4-fieldstate` 全部领先 main 0，可拆。
- **A3 `--ep` 没有记录在任何地方** —— `derive.py` 默认 1，而已提交的 derived 里有 1 / 4 / 8 三种，
  重跑全靠人记得读 HEAD。**一天内踩了三次**（含我自己两次把别人的值改掉）。
  修法：把 ep 记进记录或 derived 的输入段，让重跑不依赖记忆。

### B. 排队中

- **B1 M3.1-Flash 录入** —— 已派给 `p1-minimax`。支线（`lineage.track: side`），不改旗舰：
  org 下只有 Flash 档、没有完整 M3.1。五处结构差异与三个主线问题见那条会话的 brief。
- **B2 MiniMax M2.1 / M2.5 / M2.7 三个仓本仓一份没录** —— **M2.5 下载量 1,392,282，是旗舰 M3 的 7 倍**。
  需 `p1-minimax` 回看自己 Gate A 的排除依据还成不成立。
- **B3 站点侧两份 basis 标签表** —— `data.ts:414` 的 `KV_BASIS_LABEL` 是正宗位置，
  `[id].astro:28` 的 `KV_BASIS` 是重复。**注意闸口的提示文本已与设计脱节**：
  它叫人「从 `columns.json` 读」，而 basis 那一节已被有意摘掉；又说「`columns.ts` 落地后收紧」，
  而它已经落地。提示该改成「合并到 `data.ts`」。
- **B4 音频 / Omni** —— `docs/omni-audio-field-gap.md` 已给出需求与落地顺序，字段还没建。

### C. 视频生成线（Gate A 已完，step 1 做了一半）

- **C1 `fetch_hf.py` 对生成类仓取不到 config** —— Diffusers 布局没有顶层 `config.json`，
  配置散在 `transformer/` `vae/` `audio_vae/` 等各组件目录里。不加按组件取 config 的模式，
  **12 份视频记录全部没有 config 侧，Gate B 的双路对账只剩一路**。
- **C2 `component()` 按顶层前缀分桶在多 index 仓上不成立** —— LTX-2.5 有 7 个独立 index，
  名字跨组件重叠（`transformer` ∩ `transformer_full` 4091 个同名）。组件维度得来自
  分片路径 / `weight_map`，而那个信息现在没进 `shapes.json`。
  顺带：**扁平字典装同名张量会静默覆盖**，这次因为两份是相同副本所以没丢数，是运气。
- **C3 第一批三份录入** —— `wan2.2-ti2v-5b` → `minimax-h3` → `magi-2-preview`，第三份依赖 A3b。
- **C4 视频线的 policy 与语言模型线不同** —— 「每个 family 追最新旗舰 + 一个差分基线；
  VAE 或注意力形态变了的同代档位单独录」。要写进 `tracking.yaml` 的 policy，
  这是本仓第一次出现两套选题标准。

### E. 旗舰模型的两张图（用户 2026-10-30 点名）

参照物是 `deepseek-v41-flash` 模型页现有的逐层实现图，以及那份「V4.1 注意力分层图」
里的**图 1 生产侧 / 图 2 消费侧**。两件事分开做，第二件不依赖第一件。

- **E1 另外五个旗舰补逐层实现产物** —— `glm-5.3`、`minimax-m3`、`kimi-k3`、
  `step-5-preview`、`qwen3.8-2.4t-a95b`。现在**只有 `deepseek-v41-flash` 有
  `data/derived/*.layers.json`**，站点的逐层图就是读它。
  生成：`scripts/extract_layers.py --id <id> --model <yaml>`（只吃完整清单，
  零差异孪生版的空 `shapes.json` 会直接退出，这是有意的）。
  **先确认五份的 `shapes.json` 都是完整清单**；`facts/` 不随分支走，缺了要先拷。

- **E2 每个旗舰画一张注意力架构图**，照图 1 / 图 2 的两段式：
  **生产侧**（谁产 KV、产什么形状、什么 dtype）+ **消费侧**（谁读、怎么选、怎么拼）。

  那两张图的可复制约定，抄下来免得每张重新发明：
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
- **D2 选题漂移没有任何闸口查** —— 五道闸全在查已录记录的内部一致性，
  没有一道问「该录的录全了没有」。B2 就是实例。

## 踩过的坑（重复出现的才列）

1. **`--ep` 不在任何地方记着** —— 见 A3。一天三次。
2. **用名字当锚，锚到别处同名词** —— 三次：抠键锚到注释、证据行锚到 `platforms` 里的
   `precision: FP8_E4M3`、层号捷径咬到视觉塔。判据要按结构不按词。
3. **worktree + 符号链接** —— 三次：`facts` 不随分支走；`.gitignore` 带尾斜杠挡不住同名软链
   （已修，去掉尾斜杠）；`site/node_modules` 软链会让 dev server 服务另一个 checkout 的代码。
4. **闸口的盲区** —— 见 `.claude/skills/model-audit/SKILL.md` 规则 11。
   锚在内侧只守得住内侧；闸全锚在数据和代码上时，产物可以三千多处出错而全绿。
5. **生成物冲突不要手合** —— 从合并后的源重算。
6. **未跟踪文件会卡住 merge** —— git 直接拒绝（「会被覆盖」），平时看不见，只在合并那刻发作。

## 会话地图

| 会话 | 分支 | 状态 |
|---|---|---|
| `p1-minimax` | `p1-minimax` | 在做 B1 / B2 |
| `p1-kimi` `p1-glm` `p1-qwen` `p1-stepfun` | 同名 | 已收工并入 |
| `mm-infra` | `p3-vlm-infra` | 已收工并入 |
| `presentation` | `p4-units` | 已收工并入 |
| `p4-render-gate` | `p4-fieldstate` | 已收工并入，B3 在它那边 |
| `gatea-video` | `p2-video-gate-a` | 已收工并入，C 组归它 |
