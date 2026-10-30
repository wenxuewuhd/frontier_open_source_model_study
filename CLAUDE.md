# 本仓约定

从 infra 视角跟踪开源大模型。数字只出现在 `data/` 的 YAML 里，可校验、可复算；分析引用数据，不复制数字。
仓库结构、流程闸口、脚本用法见 [README.md](README.md)，分期见 [PLAN.md](PLAN.md)。

**要审一个模型、核一个数、录一个新族系旗舰，先读 `model-audit` skill**（`.claude/skills/model-audit/`）。
这几轮踩出来的坑都在那里，从零开始最容易重犯一遍。

## 跑

用 `.venv/bin/python`。系统 `python3` 没装 jsonschema，`validate.py` 会报一串假错误。
没有 Makefile 或统一入口，脚本用法在 README。

## 改数据

- **手写记录用定点编辑**（Edit / sed），改完跑 `validate.py`。`data/models/` 下的 YAML 带注释与折叠块，
  `yaml.safe_dump` 回写会把它们全部重排
- **config 与权重冲突时，权重赢。** config 描述模块配置，不描述落盘形状
- **第三方转换只能作旁证，不能当来源。** 社区把厂商的 `.pt` 转成 safetensors 之后，
  清单读得到了 —— 但它不在下面那条「来源优先级」的阶梯上。能做的是**为官方数字作证**：
  HunyuanVideo 13B 的社区镜像 tally 12,821,012,544 × 2 B = 25,642,025,088 B，
  与官方 `.pt` 的 25,642,314,002 B 只差 288,914 B（pickle 开销），
  这说明那次转换没有增删张量 —— 于是厂商声明的参数量多了一个独立支点。
  **但那个数不能进 `params.total`，也不能进 `per_subsystem`。**
  进 `sources` 时 `kind: repo` 并在 note 里写明是第三方转换。
- **只剩一路的记录照录，但那个事实要显式。** `derive.py` 的 `gate_b` 自动报还剩几路
  （仓里零 safetensors → 仅 config；仓里没有 config → 仅清单），模型页在交叉核对那节
  横幅说明。**不要让它只活在 `reviewed.method` 的散文里** —— 写不写全看人记不记得，
  而下面那张核对表少一路时会安静地变短，看起来和「核过、都一致」一样。
  这类记录的 `params` 标 `official` 并在 `counting_note` 写明「厂商声明，本仓未实测」。
- **来源优先级：技术报告 > 模型卡 > 厂商参考实现 > 引擎实现。** 精度与模型结构一律按厂商发布的说法填。
  厂商随仓的 `inference/` 常常自称 demo，会用**伪量化**代替打包存储，照它读会把 FP4 读成 BF16；
  vllm / sglang 只在算子表里查 —— 引擎可能滞后，也可能自己另选一套。坑的全貌见 `model-audit` skill 规则 10
- **技术报告与模型卡不同档。** 报告带节号 / 表号 / 图号，常能在源内部自校验（DeepSeek 那张
  Figure 1(b) 的图注比值与四个数据标签逐个对上，这才是它能当外部锚的理由）；模型卡正文是宣传语相邻位置，
  「KV 降低 10×」这类**没有基准的倍数不能用**，最多进 `open_question`
- **引一个厂商公布值，必须同时给定位和口径。** 定位到图号 / 表号 / 节号，否则无法复核；
  口径要写清覆盖什么、在什么上下文长度和精度下。`declared_per_token_bytes` 把这两件事做成了必填字段
  （`tier` / `covers` / `basis`），`covers != per_token` 的值只登记、不进推导 ——
  本仓 47 份里 17 份带「每序列固定」分量，它和「随上下文增长」不是一个量
- **实测自权重的字段，evidence 写 `judgement` 或 `derived`**。落在 `arch: official` 的前缀继承里就不会
  出现在 `validate.py --list-judgements`，绕过 Gate C 的人工复核
- **抽样与定向取数的产物**（`subsystem_dtypes.json`、`kv_widths.json`）**只回答它被瞄准的那个问题**：
  不并入 `shapes.json`、不用于参数量。形状为空的清单会让 `derive.py` 把参数量静默算成 0，而不是报错

## 口径

- 参数量有三套（参数量 / 权重存储 / 激活参数），缓存容量另有一根 basis 轴
  （`theoretical_floor` / `dtype_fallback`）。**不同口径之间不算倍数** —— 那是范畴错误，不是精度误差。
  定义与一份完整示例见 [docs/kv-cache-accounting.md](docs/kv-cache-accounting.md)
- **一个乘式逐因子验。** 报出一个乘积前把每个因子的来源列出来；只验一个，剩下的照样能错一个量级
- **免责声明只免掉它点名的那件事**，不会传播到引用它的派生页面
- **结构规律写进记录时钉住代际。** 「零例外」在下一代通常就不成立了

## 内部数据

算子表、自测性能、硬件相关分析只进 `data-private/`（在 gitignore 里），公开仓永不包含。
`validate.py` 公开模式拒 `visibility: internal` / `source.type: self` / `basis: measured`；
脱敏做不干净，所以**不做字段级脱敏，整条记录拒收**。内部站靠 `OVERLAY_DIR` 指向私有目录构建。

## 提交

```
Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>
```
