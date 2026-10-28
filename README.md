# 前沿开源大模型演进跟踪

从 infra 视角跟踪主流开源大模型与新型范式的演进。每次模型变化都回答三个问题：
改了什么、对负载意味着什么、下一代 infra 要准备什么。

方案与分期见 [PLAN.md](PLAN.md)。当前进度：**P0-A 完成**（schema + 脚本 + 首个压力测试用例）。

## 这个仓库是什么

数据和叙述分开：数字只出现在 `data/` 的 YAML 里，可校验、可复算；分析引用数据，不复制数字。

```
data/
  schema/           JSON Schema：模型记录、算子表
  models/<家族>/    手写事实。每个字段带证据口径
  derived/          推导值，由脚本生成，不手写
facts/              取数缓存（不进仓库，可按 pinned sha 复现）
scripts/
  fetch_hf.py       HF 仓库 -> facts/（只读 safetensors 头，不下权重）
  derive.py         config + 权重清单 -> 负载特征 + 交叉核对
  validate.py       schema + 证据覆盖 + 防泄漏
  mk_ops_table.py   算子表 YAML -> xlsx
tests/fixtures/     渲染用的合成样例
data-private/       内部 overlay（不进仓库）：算子表、自测性能、硬件相关分析
```

## 证据口径

每个字段都要能在 `evidence` 块里找到归属，支持前缀继承（`arch: official` 覆盖整棵子树，更深的路径优先）：

| 口径 | 含义 |
|---|---|
| `official` | 官方披露 |
| `derived` | 由已知量算出 |
| `judgement` | 解释判断 |
| `unknown` | 未知 |

`validate.py --list-judgements` 列出所有判断类字段。**判断类必须人看过，`official`/`derived` 自动过** —— 判断类才是会错的。

## 流程与人工 review 点

```
S0  选题        本期追哪些模型/范式                      ◀ Gate A 人
S1  取数  自动  fetch_hf.py
S2  推导  自动  derive.py，双路对账                      ◀ Gate B 人
S3  深挖  agent 读推理代码+技术报告 -> arch/精度/瓶颈
S4  算子表 skill npu-op-table -> rows/params/groups+锚点
S5  核查  agent 对抗式逐条核源码 -> 修正留痕
S6  校验  自动  validate.py
S7  渲染  自动  公开站；叠 overlay 出内部站
S8  简报        双周报草稿                               ◀ Gate C 人
```

人只进三个闸口，多了就没人真看。**Gate B 是最关键的一个**：`derive.py` 走两条路——从 config 套公式、从权重清单逐张量求和——两条都能算的量一律交叉核对，不一致就报，不替人选一个。

## 用法

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/fetch_hf.py <org/repo> --id <模型id>
.venv/bin/python scripts/derive.py --id <模型id> --model data/models/<家族>/<模型id>.yaml --ep 4
.venv/bin/python scripts/validate.py
.venv/bin/python scripts/mk_ops_table.py <算子表.yaml>
```

`derive.py` 在有不一致项时返回非零，`validate.py` 在有错误时返回非零，都可以直接给 CI 用。

## 三个口径不能混

这是本仓最容易出错的地方，工具里已经固化：

| 口径 | 含不含 scale | 低位打包 | 用途 |
|---|---|---|---|
| 参数量 | 排除 | **还原**（numel 是容器数，不是参数数） | 对外说「多大的模型」 |
| 权重存储 | 包含 | 不还原（容器就是真实存储） | 显存与带宽规划 |
| 激活参数 | 排除 | 还原 | decode 带宽需求 |

## 公开与内部

公开站和内部站是同一套代码构建两次：

- **公开仓**：架构、精度、公开引擎支持、公开 benchmark、新范式观察。
- **内部 overlay**（`data-private/`）：算子表、自测性能、硬件相关的瓶颈分析。

`validate.py` 默认拒绝 `visibility: internal`、`source.type: self` 和 `basis: measured`。脱敏做不干净，所以这里不做字段级脱敏，而是整条记录拒绝；内部构建加 `--allow-internal` 放行。

## 首个用例

`data/models/deepseek/deepseek-v41-flash.yaml` 是 P0-A 的 schema 压力测试用例。选它是因为字段覆盖最宽：MLA、细粒度 MoE、跨层 KV 共享、稀疏选块、残差流改造、外置查表、自带草稿模型、混合精度打包。

跑通过程中 schema 和脚本被它撑出四处修改，都是会静默出错的那类问题：

1. **低位打包**：FP4 权重两个一字节装进 INT8 容器，`numel` 是容器数。直接求和把参数量少算一半（485B vs 实际 756B）。
2. **跨层 KV 共享**：只有 4 层产出 KV、其余层跨层读取，indexer 另装在 8 层上。按总层数乘高估近十倍。
3. **抽样快照**：权重清单只覆盖部分层时，形状可用但参数量不可用。少算 63% 且不报错 —— 现在会拦住。
4. **口径差异**：声明的 748B 不含草稿模型部分。不声明口径就会报出一个假的 2% 差异。

修完后对账闭合到 0.13%，且推导出的激活参数 16.5B 独立对上了模型卡声明的 16B。

P0 的退出条件不是「做完 6 家」，而是**再加一个结构朴素的旗舰也能装进同一套 schema** —— 只有一个特殊用例，schema 会被它带偏。
