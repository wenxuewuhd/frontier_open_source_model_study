/**
 * 列的含义：读 data/schema/columns.json，回答「这个空格子是哪一种空」。
 *
 * Python 侧的 scripts/lib/columns.py 读同一份文件、实现同一套 op —— 一份定义，
 * 两边不会漂移。两侧的 test() 必须一致，否则同一个空格子在闸口里是「不适用」、
 * 在页面上是「未录」；`validate.py` 的第 5 道闸校验那份定义是完整的。
 *
 * **这里不放 bytes_basis 的标签表。** 那一套是另一件事（口径的强弱与措辞），
 * 它的家在 data.ts 的 KV_BASIS_LABEL / KV_BASIS_NOTE。缺值三态和 basis 语义混在
 * 一个文件里，就又会出现「改了一边忘了另一边」—— 取值表已经这么漂过一次了。
 */
import fs from 'node:fs';
import path from 'node:path';

const REPO = path.resolve(process.cwd(), '..');
const DATA = process.env.DATA_DIR ?? path.join(REPO, 'data');

const spec = JSON.parse(
  fs.readFileSync(path.join(DATA, 'schema', 'columns.json'), 'utf8'),
) as ColumnsSpec;

export type AbsenceState = 'unsurveyed' | 'not_applicable' | 'confirmed_absent';

interface AbsenceRule {
  state: AbsenceState;
  why?: string;
  label?: string;
  test?: { src?: 'rec' | 'derived'; op: string; path: string; values?: any[]; pattern?: string };
}

export interface ColumnSpec {
  unit?: string | null;
  quantity?: string;
  scope?: string;
  scope_from?: string;
  absent?: { state?: AbsenceState; why?: string; label?: string; rules?: AbsenceRule[] };
}

interface ColumnsSpec {
  absence_states: Record<string, { label: string; meaning: string; is_gap: boolean }>;
  columns: Record<string, ColumnSpec>;
}

/** 缺值的三态。渲染器拿 label 代替那个什么都不说的 `—`。 */
export interface Absent {
  absent: true;
  state: AbsenceState;
  label: string;
  why: string;
  /** true = 这是调研缺口（未录）；false = 这是信息（不适用 / 查了但没有） */
  is_gap: boolean;
}

export function isAbsent(v: unknown): v is Absent {
  return typeof v === 'object' && v !== null && (v as any).absent === true;
}

export function columnSpec(key: string): ColumnSpec | undefined {
  if (key.startsWith('sub_')) return spec.columns['$subsystem'];
  return key.startsWith('$') ? undefined : spec.columns[key];
}

function dig(obj: any, dotted: string): any {
  let cur = obj;
  for (const part of dotted.split('.')) {
    if (cur === null || typeof cur !== 'object' || !(part in cur)) return undefined;
    cur = cur[part];
  }
  return cur;
}

/**
 * 判定一条 absent 规则。op 是个固定的小词表，Python 侧实现同一套。
 * 新加 op 要同时改 scripts/lib/columns.py —— 第 5 道闸会比对两边的词表。
 */
function test(rule: AbsenceRule, rec: any, derived: any): boolean {
  const t = rule.test;
  if (!t) return true;
  const src = t.src === 'derived' ? derived ?? {} : rec;
  const val = dig(src, t.path);
  switch (t.op) {
    case 'truthy':
      return Boolean(val);
    case 'absent':
      return val === undefined || val === null;
    case 'eq':
      return (t.values ?? []).includes(val);
    case 'in':
      return Array.isArray(val) && (t.values ?? []).some((v) => val.includes(v));
    case 'not_in':
      return val !== undefined && val !== null && !(t.values ?? []).includes(val);
    case 'matches':
      return typeof val === 'string' && new RegExp(t.pattern ?? '$^').test(val);
    default:
      throw new Error(`columns.json 用了未实现的 op: ${t.op}`);
  }
}

/**
 * 这个 key 在这份记录上为空时，是三态里的哪一种。
 *
 * 说不出来就返回 unsurveyed（未录）—— 替记录把话说满（比如默认报「不适用」）
 * 会把调研缺口渲染成模型事实，那正是要修的那个错。
 *
 * label 可以逐列覆盖：态是同一个，说法可以更贴切（KV 精度那一列的
 * confirmed_absent 说「未公布」比通用的「确认无」准）。覆盖的是说法不是态。
 */
export function absence(key: string, rec: any, derived?: any): Absent {
  const a = columnSpec(key)?.absent ?? {};
  let state: AbsenceState = a.state ?? 'unsurveyed';
  let why = a.why;
  let label = a.label;
  for (const rule of a.rules ?? []) {
    if (test(rule, rec, derived)) {
      state = rule.state;
      why = rule.why ?? why;
      label = rule.label ?? label;
      break;
    }
  }
  const meta = spec.absence_states[state] ?? { label: state, meaning: '', is_gap: true };
  return {
    absent: true,
    state,
    label: label || meta.label,
    why: why || meta.meaning,
    is_gap: meta.is_gap,
  };
}

/**
 * 单位贴在数字上，不只写在表头。
 *
 * 元素/token 与字节/token 两张表并排时，列名离数字十几个字符 —— 1,600（元素）
 * 被读成了字节，和另一张表的 890 B 对不上，看起来像哪一边算错了。两个数都对：
 * 1,600 元素 × 0.556 B/元素 = 890 B，是两个口径。
 *
 * 只给**裸数字**贴单位：`47 / 8`、`MLA` 这类复合值或枚举值不贴。
 */
export function withUnit(key: string, text: string): string {
  const unit = columnSpec(key)?.unit;
  if (!unit || !/^[≤≥]?\s*[\d,.]+$/.test(text.trim())) return text;
  return `${text} ${unit}`;
}

/**
 * 三态的标签与含义，直接取自 data/schema/columns.json。
 *
 * 新范式看板也要在页面上写「未录 / 确认无」，所以这里开一个读取口 ——
 * **在那边重打一遍标签就是第二份会漂的拷贝**，basis 那套取值表已经漂过一次
 * （两份措辞不一样、而守卫按设计只对键不对值，于是一直放过）。
 */
export function absenceStates(): Record<
  AbsenceState,
  { label: string; meaning: string; is_gap: boolean }
> {
  return Object.fromEntries(
    Object.entries(spec.absence_states).filter(([k]) => !k.startsWith('$')),
  ) as Record<AbsenceState, { label: string; meaning: string; is_gap: boolean }>;
}
