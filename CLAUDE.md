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
