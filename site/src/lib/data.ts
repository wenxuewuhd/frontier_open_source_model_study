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

function readDerived(dir: string, id: string): Record<string, any> | null {
  const p = path.join(dir, 'derived', `${id}.json`);
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
  const cell = (m: Model, axis: string): string[] =>
    (m.lineage?.changed_from_predecessor ?? [])
      .filter((c: any) => c.axis === axis)
      .map((c: any) => c.text);
  return AXIS_ORDER.filter((axis) => models.some((m) => cell(m, axis).length > 0)).map((axis) => ({
    axis,
    label: AXIS_LABEL[axis] ?? axis,
    cells: models.map((m) => ({ model: m, texts: cell(m, axis) })),
  }));
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
