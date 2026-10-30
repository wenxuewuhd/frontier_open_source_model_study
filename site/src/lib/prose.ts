/**
 * 记录散文里的行内标记 —— 把 `**粗体**` 与 `` `代码` `` 切成词元。
 *
 * 为什么需要这个：本仓的记录散文是**按 Markdown 写的**（`counting_note`、`finding`、
 * `infra_signals[].text` 一律用 `**…**` 强调结论、用反引号标张量名），而站点此前
 * 一处 Markdown 渲染都没有，散文按纯文本插进去 —— 于是全站 3,665 处字面 `**`
 * 直接显示给读者。四道闸当时全绿：validate 查语义、第 5 道闸查标签表、
 * kv_table --check 查定义段、站点构建查语法，**没有一道问「渲染出来的东西和作者
 * 想表达的是不是一回事」**（model-audit skill 规则 11：所有闸都锚在数据和代码上，
 * 没有一道锚在产物上）。
 *
 * **只认这两种，不引整个 Markdown 管线。** 理由不是图省事：
 *
 *   - 全仓统计，粗体 1,712 处、反引号 199 处、链接 0 处
 *   - **单个 `*` 必须保持字面**。全仓 42 处单星全是张量名里的通配符
 *     （`hc_attn_*`、`mlp_hyper_connection.*`），不是斜体。按 Markdown 斜体去解
 *     会把张量名吃掉 —— 而张量名正是这些散文要说清的东西
 *
 * 切词元而不是生成 HTML 字符串，所以**不需要 `set:html`**：片段都是纯文本，
 * 由 Astro 自己转义。注入在构造上就不可能发生，不依赖任何清洗。
 */

export type Token =
  | { kind: 'text'; text: string }
  | { kind: 'code'; text: string }
  /** 粗体里还可能有代码跨段，所以它装的是子词元而不是一串文本 */
  | { kind: 'bold'; parts: Token[] };

const TICK = '`';

/** 从 i 处找配对的收尾 `**`。跳过代码跨段 —— 里面的星号是字面内容。 */
function findBoldEnd(s: string, i: number): number {
  let j = i;
  while (j < s.length) {
    if (s[j] === TICK) {
      const close = s.indexOf(TICK, j + 1);
      if (close < 0) return -1;
      j = close + 1;
      continue;
    }
    if (s.startsWith('**', j)) return j;
    if (s[j] === '\n') return -1; // 强调不跨行
    j += 1;
  }
  return -1;
}

/** 把一段文本里的代码跨段切出来，其余原样。 */
function splitCode(s: string): Token[] {
  const out: Token[] = [];
  let i = 0;
  let buf = '';
  while (i < s.length) {
    if (s[i] === TICK) {
      const close = s.indexOf(TICK, i + 1);
      if (close > i + 1 && !s.slice(i + 1, close).includes('\n')) {
        if (buf) {
          out.push({ kind: 'text', text: buf });
          buf = '';
        }
        out.push({ kind: 'code', text: s.slice(i + 1, close) });
        i = close + 1;
        continue;
      }
    }
    buf += s[i];
    i += 1;
  }
  if (buf) out.push({ kind: 'text', text: buf });
  return out;
}

/**
 * 把一段散文切成词元。
 *
 * **先取粗体、再在段内解代码**，顺序不能反：反过来的话
 * `**字节那一列的口径是 \`dtype_fallback\`（按模型 dtype 兜底）：**`
 * 会被代码跨段从中间切开，两个 `**` 落进不同片段，谁也配不上对 ——
 * 这一类真的出现过，而且正是第一版漏掉的那些。
 *
 * 配不上对的 `**` 原样留着：那通常意味着作者写漏了一侧（或者在定界符里带了空格，
 * `** 这样 **` 在 Markdown 里本来就不是粗体）。**让它显形比悄悄吃掉强** ——
 * 产物侧的检查会把它报出来，由人去改记录。
 */
export function inlineTokens(src: unknown): Token[] {
  if (typeof src !== 'string' || !src) return [];
  const out: Token[] = [];
  let i = 0;
  let plain = '';
  const flush = () => {
    if (plain) {
      out.push(...splitCode(plain));
      plain = '';
    }
  };
  while (i < src.length) {
    if (src.startsWith('**', i)) {
      const end = findBoldEnd(src, i + 2);
      if (end > i + 2) {
        flush();
        out.push({ kind: 'bold', parts: splitCode(src.slice(i + 2, end)) });
        i = end + 2;
        continue;
      }
    }
    if (src[i] === TICK) {
      const close = src.indexOf(TICK, i + 1);
      if (close > i + 1 && !src.slice(i + 1, close).includes('\n')) {
        flush();
        out.push({ kind: 'code', text: src.slice(i + 1, close) });
        i = close + 1;
        continue;
      }
    }
    plain += src[i];
    i += 1;
  }
  flush();
  return out;
}
