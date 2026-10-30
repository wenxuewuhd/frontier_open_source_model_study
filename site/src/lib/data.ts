/**
 * 数据层：读 data/ 下的手写事实与推导值，合并成渲染用的记录。
 *
 * 两条原则，和仓库其余部分一致：
 *   1. 不在这里定第二套 schema。校验是 scripts/validate.py 的事，
 *      这里只负责读取与渲染，字段缺失要能优雅降级而不是崩。
 *   2. 推导值与手写事实分开存放，合并只发生在渲染期，
 *      所以页面上能明确区分「谁说的」。
 *
 * 内部 overlay：设 OVERLAY_DIR 指向 data-private，其记录按 id 深合并，
 * 数组字段追加。公开构建不设这个变量。
 */
import fs from 'node:fs';
import path from 'node:path';
import yaml from 'js-yaml';

/**
 * 统一的 YAML 读取。
 *
 * 必须用 CORE_SCHEMA：js-yaml 默认走 YAML 1.1，会把 2026-10-12 这样的 ISO 日期
 * 解析成 Date 对象，而我们的 schema 声明的是 string + format:date。
 * 这和 Python 侧 scripts/lib/yamlio.py 关掉 timestamp 解析器是同一件事 ——
 * 两侧必须一致，否则同一份文件在校验时和渲染时得到不同的类型。
 */
function loadYaml(file: string): any {
  return yaml.load(fs.readFileSync(file, 'utf8'), { schema: yaml.CORE_SCHEMA });
}

const REPO = path.resolve(process.cwd(), '..');
const DATA = process.env.DATA_DIR ?? path.join(REPO, 'data');
const OVERLAY = process.env.OVERLAY_DIR ?? null;

export type EvidenceTag = 'official' | 'derived' | 'judgement' | 'unknown';

export const EVIDENCE_LABEL: Record<EvidenceTag, string> = {
  official: '官方披露',
  derived: '计算推导',
  judgement: '解释判断',
  unknown: '未知',
};

export interface Model {
  id: string;
  family: string;
  kind: string;
  display_name: string;
  released?: string;
  license?: string;
  visibility?: string;
  hf?: Record<string, any>;
  lineage?: Record<string, any>;
  params?: Record<string, any>;
  context?: Record<string, any>;
  arch?: Record<string, any>;
  generation?: Record<string, any>;
  precision?: Record<string, any>;
  platforms?: any[];
  performance?: any[];
  bottlenecks?: any[];
  ops_table?: Record<string, any>;
  infra_signals?: any[];
  evidence: Record<string, EvidenceTag>;
  sources?: any[];
  notes?: string;
  open_questions?: string[];
  /** scripts/derive.py 的产物，渲染期挂进来 */
  derived?: Record<string, any> | null;
  /** scripts/extract_layers.py 的产物：逐层实现。只有拉过全量清单的模型才有 */
  layers?: Record<string, any> | null;
}

export const FAMILY_LABEL: Record<string, string> = {
  deepseek: 'DeepSeek',
  glm: 'GLM',
  minimax: 'MiniMax',
  kimi: 'Kimi',
  qwen: 'Qwen',
  stepfun: '阶跃星辰',
  other: '其他',
};

/** 家族展示顺序。追踪范围固定，所以写死而不是按字母排。 */
export const FAMILY_ORDER = ['deepseek', 'qwen', 'glm', 'minimax', 'kimi', 'stepfun', 'other'];

function walkYaml(dir: string): string[] {
  if (!fs.existsSync(dir)) return [];
  return fs.readdirSync(dir, { withFileTypes: true }).flatMap((e) => {
    const p = path.join(dir, e.name);
    if (e.isDirectory()) return walkYaml(p);
    return e.name.endsWith('.yaml') || e.name.endsWith('.yml') ? [p] : [];
  });
}

function isPlainObject(v: unknown): v is Record<string, any> {
  return typeof v === 'object' && v !== null && !Array.isArray(v);
}

/** overlay 深合并：对象递归，数组追加，标量以 overlay 为准。 */
function merge(base: any, over: any): any {
  if (Array.isArray(base) && Array.isArray(over)) return [...base, ...over];
  if (isPlainObject(base) && isPlainObject(over)) {
    const out: Record<string, any> = { ...base };
    for (const [k, v] of Object.entries(over)) out[k] = k in base ? merge(base[k], v) : v;
    return out;
  }
  return over;
}

function readDerived(dir: string, id: string, suffix = ''): Record<string, any> | null {
  const p = path.join(dir, 'derived', `${id}${suffix}.json`);
  if (!fs.existsSync(p)) return null;
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch {
    return null;
  }
}

let _cache: Model[] | null = null;

export function allModels(): Model[] {
  // dev 下不缓存：data/ 是外部文件，不走模块图，HMR 不会让这个缓存失效。
  // 缓存住会让刚跑完 derive.py 的结果在页面上看不到，排查起来很费时间。
  if (_cache && !import.meta.env.DEV) return _cache;
  const byId = new Map<string, Model>();

  for (const dir of [DATA, OVERLAY].filter(Boolean) as string[]) {
    for (const f of walkYaml(path.join(dir, 'models'))) {
      const doc = loadYaml(f) as Model | null;
      if (!doc || !doc.id) continue;
      const prev = byId.get(doc.id);
      byId.set(doc.id, prev ? merge(prev, doc) : doc);
    }
  }
  for (const m of byId.values()) {
    // 推导值优先取 overlay 的（内部构建里可能含实测修正）
    m.derived = (OVERLAY && readDerived(OVERLAY, m.id)) || readDerived(DATA, m.id);
    m.layers = (OVERLAY && readDerived(OVERLAY, m.id, '.layers')) || readDerived(DATA, m.id, '.layers');
  }

  _cache = [...byId.values()].sort((a, b) => {
    const fa = FAMILY_ORDER.indexOf(a.family) - FAMILY_ORDER.indexOf(b.family);
    if (fa !== 0) return fa;
    return (b.released ?? '').localeCompare(a.released ?? '');
  });
  return _cache;
}

export function modelById(id: string): Model | undefined {
  return allModels().find((m) => m.id === id);
}

export function families(): { key: string; label: string; models: Model[] }[] {
  const seen = new Map<string, Model[]>();
  for (const m of allModels()) {
    if (!seen.has(m.family)) seen.set(m.family, []);
    seen.get(m.family)!.push(m);
  }
  return FAMILY_ORDER.filter((k) => seen.has(k)).map((k) => ({
    key: k,
    label: FAMILY_LABEL[k] ?? k,
    models: seen.get(k)!,
  }));
}

/**
 * 字段级证据口径，带前缀继承：arch 覆盖 arch.attention.type，更深的路径优先。
 * 规则必须和 scripts/validate.py 的 evidence_for 保持一致 ——
 * 两边不一致会让校验通过的记录在页面上显示错的口径。
 */
export function evidenceFor(
  fieldPath: string,
  ev: Record<string, EvidenceTag> | undefined,
): EvidenceTag | undefined {
  if (!ev) return undefined;
  let best: EvidenceTag | undefined;
  let bestLen = -1;
  for (const [k, v] of Object.entries(ev)) {
    if ((fieldPath === k || fieldPath.startsWith(k + '.')) && k.length > bestLen) {
      best = v;
      bestLen = k.length;
    }
  }
  return best;
}

// ---------------------------------------------------------------- 格式化

export function fmtCount(n: number | undefined | null): string {
  if (n === undefined || n === null || !isFinite(n)) return '—';
  for (const [unit, div] of [
    ['T', 1e12],
    ['B', 1e9],
    ['M', 1e6],
    ['K', 1e3],
  ] as [string, number][]) {
    if (Math.abs(n) >= div) {
      const v = n / div;
      return `${v >= 100 ? v.toFixed(0) : v.toFixed(v >= 10 ? 1 : 2)}${unit}`;
    }
  }
  return String(n);
}

export function fmtBytes(b: number | undefined | null): string {
  if (b === undefined || b === null || !isFinite(b)) return '—';
  for (const [unit, div] of [
    ['TiB', 2 ** 40],
    ['GiB', 2 ** 30],
    ['MiB', 2 ** 20],
    ['KiB', 2 ** 10],
  ] as [string, number][]) {
    if (b >= div) return `${(b / div).toFixed(2)} ${unit}`;
  }
  return `${Math.round(b)} B`;
}

/** 极小或极大的数用科学计数法。norm_eps 这类取值写成小数会变成一串 0。 */
export function fmtSci(n: unknown): string | undefined {
  if (typeof n !== 'number' || !isFinite(n)) return n === undefined ? undefined : String(n);
  if (n !== 0 && (Math.abs(n) < 1e-3 || Math.abs(n) >= 1e7)) {
    return n.toExponential().replace('e', 'e');
  }
  return String(n);
}

export function fmtInt(n: number | undefined | null): string {
  return n === undefined || n === null ? '—' : n.toLocaleString('en-US');
}

/** 站内链接。base 由 astro.config 给出，手拼会在部署后全错。 */
export function url(p: string): string {
  const base = import.meta.env.BASE_URL.replace(/\/$/, '');
  return `${base}/${p.replace(/^\//, '')}`;
}

/** 总览用：把所有记录的 infra 信号摊平，按 horizon 分组。 */
export function allSignals() {
  return allModels().flatMap((m) =>
    (m.infra_signals ?? []).map((s: any) => ({ ...s, model: m })),
  );
}

/** 时间线：按发布日期倒序，只取有日期的。 */
export function timeline(): Model[] {
  return allModels()
    .filter((m) => m.released)
    .sort((a, b) => (b.released ?? '').localeCompare(a.released ?? ''));
}

export const BUILD_IS_INTERNAL = Boolean(OVERLAY);

/** infra 信号的时间尺度。schema 的 horizon 枚举改了这里也要改，所以只存一份。 */
export const HORIZON_LABEL: Record<string, string> = {
  now: '当下就要动',
  'next-gen': '下一代要准备',
  watch: '继续观察',
  settled: '已定型，无需动作',
};

export const HORIZON_ORDER = ['now', 'next-gen', 'watch', 'settled'];

/** 演进维度。schema 的 lineage.changed_from_predecessor[].axis 枚举，顺序固定便于逐版本对读。 */
export const AXIS_LABEL: Record<string, string> = {
  attention: '注意力',
  moe: 'MoE / 路由',
  residual: '残差',
  retrieval: '检索',
  precision: '量化精度',
  context: '上下文',
  modality: '模态',
  decoding: '解码 / 投机',
  scale: '规格',
  training: '训练 / 后训练',
};

export const AXIS_ORDER = [
  'attention', 'moe', 'residual', 'retrieval', 'precision',
  'context', 'modality', 'decoding', 'scale', 'training',
];

export const SEVERITY_LABEL: Record<string, string> = {
  wrong: '事实错误',
  unsupported: '缺依据',
  imprecise: '表述误导',
  missing: '漏字段',
  stale: '已过时',
  overreach: '过度推论',
  miscategorized: '口径标错',
};

export const RESOLUTION_LABEL: Record<string, string> = {
  corrected: '已改',
  refuted: '质疑不成立',
  confirmed: '已确认',
  open: '未解决',
  deferred: '本期不处理',
};

/** 核查状态。未核查的记录要在列表里看得出来，否则「没问题」和「没人看过」分不开。 */
export function reviewStatus(m: Model) {
  const findings = (m as any).review ?? [];
  return {
    reviewed: Boolean((m as any).reviewed),
    date: (m as any).reviewed?.date as string | undefined,
    total: findings.length as number,
    open: findings.filter((f: any) => f.resolution === 'open').length as number,
    corrected: findings.filter((f: any) => f.resolution === 'corrected').length as number,
  };
}

/**
 * 跨版本跟踪的字段。
 *
 * 每个字段挂一个演进维度（axis）和一张表（group），家族页的「变化矩阵」「规格对照」
 * 「精度对照」三块都由这一份定义驱动 —— 加一行只改这里，不用动三个页面。
 *
 * 取值一律从记录里读，读不到返回 undefined。**不回退到猜测**：空白表示该版本没有这一项，
 * 和「有但没录」要能分开，所以取不到就是取不到。
 */
/**
 * 子系统的顺序与中文名，和 scripts/lib/subsystems.py 读同一个文件。
 * 在两种语言里各写一份，迟早会在加子系统时只改一边。
 */
export const SUBSYSTEMS: { key: string; label: string }[] = JSON.parse(
  fs.readFileSync(path.join(DATA, 'schema', 'subsystems.json'), 'utf8'),
).subsystems;

export interface TrackedField {
  key: string;
  label: string;
  axis: string;
  group: 'spec' | 'precision';
  /**
   * 进不进变化矩阵：
   *   true / 省略      照常比
   *   false            不比。自由文本（措辞差异不是架构变化）
   *   'both-measured'  只在两边都测过时才比。实测字段缺值是「没测」不是「取消」，
   *                    照常比会把调研进度显示成模型变化
   *   'same-basis'     只在两边口径（basis）相同时才比。KV 字节一侧按落盘精度算（理论下限）、
   *                    一侧按模型 dtype 兜底时，from → to 的倍数里混着「换了一套精度假设」，
   *                    读者会把它整个算成架构的功劳
   */
  diff?: boolean | 'both-measured' | 'same-basis';
  /** diff 为 'same-basis' 时的口径。取不到或为 mixed 都不比：mixed 两边可能是不同的混法 */
  basis?: (m: Model) => string | undefined;
  get: (m: Model) => string | number | undefined;
}

const dtype = (v: unknown) => (typeof v === 'string' ? v.toUpperCase() : undefined);

/**
 * KV 缓存精度的显示值 —— 把「取不到」拆成两种说法。
 *
 * 以前取不到就渲染 `—`，而那一个符号同时背着「厂商没公布」和「这一代不建 KV」。
 * 对 infra 决策这是两个结论：前者说明字节那一列是本仓按模型 dtype 兜底填的占位值，
 * 真实落盘精度可能更省，容量规划要留余量；后者说明根本没有这笔账要规划。
 *
 * 判据只认记录里已经有的东西，不从别处推：
 *   记录里有值                                        → 照报
 *   没值，且缓存账已经算过、每 token 与每序列字节都是零 → 不适用（没有 KV 要落盘）
 *   其余（含没 derive、status unknown）                → 未公布
 *
 * 「没 derive」归到「未公布」而不是单列一类，是因为这一格说的是**厂商公布了什么**，
 * 而不是本仓算到了哪一步 —— 调研进度由字节那一列的 basis 和 derive 状态承担。
 *
 * 当前 47 份里 45 份走「未公布」、0 份走「不适用」：五个家族逐份查过模型卡与技术报告，
 * DeepSeek 之外没有任何厂商公布过 KV 落盘精度。「不适用」这一支现在是空的，
 * 留着是因为判据得先立住 —— 真出现一份不建 KV 的记录时，它不能和那 45 份共用一个符号。
 */
export function kvCachePrecision(m: Model): { text: string; declared: boolean } {
  const v = dtype(m.precision?.kv_cache);
  if (v) return { text: v, declared: true };
  const kv = m.derived?.kv_cache;
  const accounted = kv?.status && kv.status !== 'unknown';
  if (accounted && !kv!.per_token_bytes && !kv!.per_sequence_bytes) {
    return { text: '不适用', declared: false };
  }
  return { text: '未公布', declared: false };
}

export const KV_BASIS_LABEL: Record<string, string> = {
  theoretical_floor: '理论下限',
  dtype_fallback: '精度兜底',
  declared_dtype: '声明精度',
  mixed: '口径混合',
};

export const TRACKED_FIELDS: TrackedField[] = [
  { key: 'layers', label: '层数', axis: 'scale', group: 'spec', get: (m) => m.arch?.layers },
  { key: 'hidden', label: 'hidden', axis: 'scale', group: 'spec', get: (m) => m.arch?.hidden },
  { key: 'vocab', label: '词表', axis: 'scale', group: 'spec', get: (m) => m.arch?.vocab },
  { key: 'attn_type', label: '注意力类型', axis: 'attention', group: 'spec', get: (m) => m.arch?.attention?.type },
  { key: 'heads', label: '头数 / KV 头', axis: 'attention', group: 'spec',
    get: (m) => (m.arch?.attention?.num_heads
      ? `${m.arch.attention.num_heads} / ${m.arch.attention.num_kv_heads ?? '—'}` : undefined) },
  { key: 'head_dim', label: 'head_dim', axis: 'attention', group: 'spec', get: (m) => m.arch?.attention?.head_dim },
  { key: 'kv_lora', label: 'KV latent 秩', axis: 'attention', group: 'spec', get: (m) => m.arch?.attention?.kv_lora_rank },
  { key: 'q_lora', label: 'Q 低秩', axis: 'attention', group: 'spec', get: (m) => m.arch?.attention?.q_lora_rank },
  { key: 'o_lora', label: '输出低秩 / 组', axis: 'attention', group: 'spec',
    get: (m) => (m.arch?.attention?.o_lora_rank
      ? `${m.arch.attention.o_lora_rank} / ${m.arch.attention.o_groups ?? '—'}` : undefined) },
  { key: 'swa', label: '滑窗', axis: 'attention', group: 'spec', get: (m) => m.arch?.attention?.sliding_window },
  { key: 'sink', label: 'attention sink', axis: 'attention', group: 'spec',
    get: (m) => (m.arch?.attention?.attn_sink ? '有' : undefined) },
  { key: 'sparse', label: '稀疏 index 头 / topk', axis: 'attention', group: 'spec',
    get: (m) => (m.arch?.attention?.sparse?.index_n_heads
      ? `${m.arch.attention.sparse.index_n_heads} / ${m.arch.attention.sparse.index_topk ?? '—'}` : undefined) },
  { key: 'kv_src', label: '产出 KV 的层', axis: 'attention', group: 'spec',
    get: (m) => {
      const k = m.arch?.attention?.kv_sharing?.source_layers;
      return k ? `${k.length} / ${m.arch?.layers ?? '?'} 层` : undefined;
    } },
  { key: 'idx_src', label: '产出 index K 的层', axis: 'attention', group: 'spec',
    get: (m) => {
      const k = m.arch?.attention?.kv_sharing?.index_kv_source_layers;
      return k ? `${k.length} 层` : undefined;
    } },
  // 跨代比 KV 结构用元素数，不用字节：字节 = 元素 × 精度，而各代的精度来源不一
  // （V4 按 BF16 兜底，V4.1 查到了落盘格式）。V4-Flash → V4.1 按字节是 6880 → 890，7.7 倍，
  // 属于结构的只有元素那 2.15 倍。只计随上下文增长的分量，滑窗与线性状态是每序列常量
  { key: 'kv_elems', label: 'KV 元素 / token', axis: 'attention', group: 'spec',
    get: (m) => {
      const kv = m.derived?.kv_cache;
      if (!kv?.per_token_elems) return undefined;
      // 公式路径里层数或宽度是上界的，数字前标 ≤，比较时读者能看出哪一侧不精确。
      // 单位跟着数字走，不只写在表头：裸的「1,600」被当成字节读过一次 ——
      // 和它并排的字节列是「890 B（理论下限）」，两个数差 1.8 倍，看起来像哪一边算错了，
      // 而实际是 1600 元素 × 0.5563 B/元素 = 890 B，两个口径
      return `${kv.per_token_elems_bound === 'upper_bound' ? '≤ ' : ''}${fmtInt(kv.per_token_elems)} 元素`;
    } },
  { key: 'experts', label: '路由专家 / top-k', axis: 'moe', group: 'spec',
    get: (m) => (m.arch?.moe?.routed_experts
      ? `${m.arch.moe.routed_experts} / ${m.arch.moe.top_k ?? '—'}` : undefined) },
  { key: 'shared', label: '共享专家', axis: 'moe', group: 'spec', get: (m) => m.arch?.moe?.shared_experts },
  { key: 'moe_inter', label: '专家中间维', axis: 'moe', group: 'spec', get: (m) => m.arch?.moe?.intermediate },
  { key: 'dense_k', label: 'dense 前缀层', axis: 'moe', group: 'spec', get: (m) => m.arch?.moe?.first_k_dense },
  // 自由文本，措辞差异不是架构变化 —— 路由的实际改动由逐维度演进的叙述承担
  { key: 'routing', label: '打分 / 选路', axis: 'moe', group: 'spec', diff: false,
    get: (m) => m.arch?.moe?.routing },
  { key: 'hash', label: 'hash 路由层', axis: 'moe', group: 'spec', get: (m) => m.arch?.moe?.hash_layers?.count },
  { key: 'residual', label: '残差', axis: 'residual', group: 'spec',
    get: (m) => (m.arch?.residual?.type
      ? `${m.arch.residual.type}${m.arch.residual.multiplier ? ` ×${m.arch.residual.multiplier}` : ''}` : undefined) },
  { key: 'retrieval', label: '检索层', axis: 'retrieval', group: 'spec',
    get: (m) => (m.arch?.retrieval?.layers ? `第 ${m.arch.retrieval.layers.join('、')} 层` : undefined) },
  { key: 'mtp', label: 'MTP 层', axis: 'decoding', group: 'spec', get: (m) => m.arch?.decoding?.mtp_layers },
  { key: 'spec_method', label: '投机方法', axis: 'decoding', group: 'spec',
    get: (m) => m.arch?.decoding?.speculative?.method },
  { key: 'vision', label: '视觉塔', axis: 'modality', group: 'spec',
    get: (m) => (m.arch?.vision ? `${m.arch.vision.layers ?? '?'} 层 / hidden ${m.arch.vision.hidden ?? '?'}` : undefined) },
  { key: 'ctx', label: '上下文 / 原生', axis: 'context', group: 'spec',
    get: (m) => (m.context?.max
      ? `${fmtInt(m.context.max)} / ${m.context.train_native ? fmtInt(m.context.train_native) : '—'}` : undefined) },

  // —— 精度：单拉一组。量化是独立于规格的一条演进线，混在规格表里会看不出节奏。
  //
  // 这里只留 **dtype 看不出来的** config 声明项。
  // quant_method（整体 FP8）、weight_format（fmt e4m3）、expert_dtype（FP4）三项
  // 都被下面的逐子系统实测覆盖了，留着是同一件事说两遍；其中 weight_format 还会
  // 在 V4.1 显示成「e4m3 → 取消」（config 不再声明这个键），读起来像放弃了 E4M3。
  //
  // scale_format 留着，因为它和实测**不一致**，而那个不一致正是要说的事：
  // V3.1 起声明 ue8m0，但 scale 张量实测到 V4 才真的变成 8 bit。
  // 标明「config 声明」：这个值对 FP4 专家是错的，实测值在下面的逐子系统行里。
  // 不标的话两处并排显示 [128,128] 和 [1,32]，读者无从判断信哪个
  { key: 'quant_block', label: '量化块（config 声明）', axis: 'precision', group: 'precision',
    get: (m) => {
      const b = m.precision?.quantization?.weight_block_size;
      return b ? `[${b.join(',')}]` : undefined;
    } },
  { key: 'scale_fmt', label: 'scale 格式（config 声明）', axis: 'precision', group: 'precision',
    get: (m) => m.precision?.quantization?.scale_format },
  { key: 'act_scheme', label: '激活量化', axis: 'precision', group: 'precision',
    get: (m) => m.precision?.quantization?.activation_scheme },
  // 永远有值：缺口径时报「未公布」或「不适用」，不留 `—`（判据见 kvCachePrecision）。
  // 值刻意只有这两种说法，不带「按 dtype 兜底 / 厂商只公布字节」这类后缀 ——
  // 带了的话，兜底的前代与厂商公布的后代会比出一条 from → to，
  // 而那是本仓查到了什么的差别，不是模型的差别。兜底来源看同表的「KV 字节 / token」
  { key: 'kv_dtype', label: 'KV 缓存精度', axis: 'precision', group: 'precision',
    get: (m) => kvCachePrecision(m).text },
  // 字节随精度假设变，口径写在数字旁边；口径不同的两版不进变化矩阵 —— 结构变化看上面的元素行
  { key: 'kv_bytes', label: 'KV 字节 / token', axis: 'precision', group: 'precision',
    diff: 'same-basis',
    basis: (m) => m.derived?.kv_cache?.bytes_basis,
    get: (m) => {
      const kv = m.derived?.kv_cache;
      if (!kv?.per_token_bytes) return undefined;
      const b = KV_BASIS_LABEL[kv.bytes_basis];
      return `${fmtInt(Math.round(kv.per_token_bytes))} B${b ? `（${b}）` : ''}`;
    } },
  ...subsystemFields(),
];

/**
 * 逐子系统的实测精度，每个子系统一行，格子里是「权重精度 / scale 精度」。
 *
 * 和上面的 config 声明项并列进同一组，所以它们也能进变化矩阵 ——
 * 「路由专家 FP8_E4M3 / FP32 → FP4 / FP8_E8M0」比「路由专家精度 无 → FP4」说得清楚得多。
 * 但只在两边都测过时才比（diff: 'both-measured'）：没测过的版本缺值是调研进度，不是模型变化。
 */
function subsystemFields(): TrackedField[] {
  return SUBSYSTEMS.map((s) => ({
    key: `sub_${s.key}`,
    label: `${s.label}（实测）`,
    axis: 'precision',
    group: 'precision' as const,
    diff: 'both-measured' as const,
    get: (m: Model) => {
      const q = m.precision?.quantization;
      const w = q?.per_subsystem?.[s.key];
      if (!w) return undefined;
      const sc = q?.per_subsystem_scale?.[s.key];
      const bk = q?.per_subsystem_block?.[s.key];
      // 三件事挤在一格，但它们必须一起看：反量化要同时知道权重精度、scale 精度和块。
      // 少看一个就会错 —— V4 的专家是 FP4 / FP8_E8M0 / [1,32]，
      // 照 config 的 [128,128] 去读 scale 会差 128 倍。
      return [w, sc, bk].filter(Boolean).join(' / ');
    },
  }));
}

/** 子系统行的 key 前缀，页面用它在「config 声明」与「实测」之间插分组行。 */
export const SUBSYSTEM_FIELD_PREFIX = 'sub_';


export function changesFor(m: Model): FieldChange[] {
  const prevId = m.lineage?.predecessor;
  if (!prevId) return [];
  const prev = modelById(prevId);
  if (!prev) return [];
  return TRACKED_FIELDS.filter((f) => f.diff !== false)
    .map((f) => ({ field: f, from: f.get(prev), to: f.get(m) }))
    .filter((c) => {
      // 实测字段缺值是「这一版没测」，不是「这一版取消了」。
      // 两边都有值才算得上变化，否则矩阵会把调研进度显示成模型变化。
      if (c.field.diff === 'both-measured' && (c.from === undefined || c.to === undefined)) {
        return false;
      }
      if (c.field.diff === 'same-basis') {
        const a = c.field.basis?.(prev);
        const b = c.field.basis?.(m);
        if (!a || !b || a !== b || a === 'mixed') return false;
      }
      return String(c.from ?? '') !== String(c.to ?? '');
    });
}

/** 某家族的记录，按发布日期升序 —— 家族页要顺着读演进，和列表页的倒序相反。 */
export function familyModels(key: string): Model[] {
  return allModels()
    .filter((m) => m.family === key)
    .sort((a, b) => (a.released ?? '').localeCompare(b.released ?? ''));
}

/**
 * 维度 × 版本矩阵。
 *
 * 内容全部来自各记录的 lineage.changed_from_predecessor，不另写一份表 ——
 * 手写的表会和记录各自漂移，而记录是校验覆盖到的那一份。
 * 返回只包含真的有内容的维度行，避免整行空白。
 */
export function axisMatrix(models: Model[]) {
  const cell = (m: Model, axis: string) =>
    (m.lineage?.changed_from_predecessor ?? []).filter((c: any) => c.axis === axis);
  return AXIS_ORDER.filter((axis) => models.some((m) => cell(m, axis).length > 0)).map((axis) => ({
    axis,
    label: AXIS_LABEL[axis] ?? axis,
    cells: models.map((m) => ({ model: m, entries: cell(m, axis) })),
  }));
}

/**
 * 字段级的变化矩阵：维度 × 版本，格子里是「字段：旧值 → 新值」。
 *
 * 和 axisMatrix 的区别是来源：这里的每一格都由 changesFor 从两份记录的字段值算出来，
 * 不经过任何人写的文字，所以不会和记录漂移；axisMatrix 给的是叙述与收益。
 * 只保留真的有字段变化的版本做列 —— 纯后训练版本在这张表上整列空白，没有信息。
 */
export function changeMatrix(models: Model[]) {
  const changes = new Map<string, FieldChange[]>(models.map((m) => [m.id, changesFor(m)]));
  const cols = models.filter((m) => (changes.get(m.id) ?? []).length > 0);
  const rows = AXIS_ORDER.filter((axis) =>
    cols.some((m) => changes.get(m.id)!.some((c) => c.field.axis === axis)),
  ).map((axis) => ({
    axis,
    label: AXIS_LABEL[axis] ?? axis,
    cells: cols.map((m) => ({
      model: m,
      changes: changes.get(m.id)!.filter((c) => c.field.axis === axis),
    })),
  }));
  // 「零变化」只对有前代的版本成立。家族首录没有前代可比，changesFor 同样返回空，
  // 混进来会让页面把它说成「相对前代一个字段都没动」—— 它根本没有前代
  const unchanged = models.filter(
    (m) => m.lineage?.predecessor && (changes.get(m.id) ?? []).length === 0,
  );
  return { cols, rows, unchanged };
}

export interface RosterEntry {
  family: string;
  flagship: string;
  hf_repo?: string;
  released?: string;
  model_id?: string;
  status: 'recorded' | 'pending';
  note?: string;
}

/** 追踪名册（data/tracking.yaml）：选取结果，不是模型记录。 */
export function roster(): { updated?: string; policy?: string; families: RosterEntry[] } {
  const p = path.join(DATA, 'tracking.yaml');
  if (!fs.existsSync(p)) return { families: [] };
  const doc = loadYaml(p) as any;
  return { updated: doc?.updated, policy: doc?.policy, families: doc?.families ?? [] };
}
