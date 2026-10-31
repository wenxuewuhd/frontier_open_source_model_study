/**
 * 块格式名的推导：元素格式 + 块 + scale 三个因子对上签名才给名字。
 *
 * **只此一份。** 精度演进页的格子（badge + title）和 `data.ts` 里逐子系统的变化行
 * 都走这里 —— 两边各留一份副本就会出现「徽章写 MXFP4、下面那行写 FP4」这种
 * 同一个子系统两种说法的并置，而那正是这个模块存在的原因：读者为此困惑过三轮。
 *
 * 和 `DTYPE_CELL` / `dtype.css` 是同一类约定：一件事的几个面分开放，但必须一起改。
 */

/**
 * 块格式的名字由**签名**推出来，不取记录的 `dtype` 字段。
 *
 * 因为 `FP4` 不是一种存储格式，它是**元素格式**（E2M1：1 符号 + 2 指数 + 1 尾数）。
 * NVFP4 与 MXFP4 的元素都是 E2M1，分开它们的是块大小与 scale：
 *   块 16 + E4M3  → NVFP4 式（NVIDIA；严格的 NVFP4 还有第二级 FP32 全局 scale）
 *   块 32 + E8M0  → MXFP4（OCP Microscaling 标准）
 * 本仓的 dtype 字段记的是元素格式，所以 DeepSeek 全族的 4bit 都写作 `FP4`，
 * 而它们其实分属两种块格式 —— 页面只印 `FP4` 等于把这个区别丢了。
 *
 * **只在签名完全对上时给名字，对不上就不给。** 推名字是推断，不是记录里的事实：
 *   - scale 记成容器（如 `UINT8`，E8M0 的容器）时不推 —— 容器说明不了逻辑精度
 *   - 块是混合的（`[1,32]+[32,32]`）时这里不推 —— 整格没有单一名字。**退一步按主方案
 *     再推一次见 `mainFormat`**，条件是记录补了 `per_subsystem_schemes`
 * 推出来的名字一律标成推导，读者要核就看并排的块与 scale。
 */
export function blockSize(block: unknown): number | undefined {
  if (typeof block === 'number') return block;
  if (typeof block !== 'string') return undefined;
  // 混合块不给单一名字。**不要改成「取末尾那个数」**：V4.1 的 '[1,32]+[32,32]' 会取到 32、
  // V4 三份的 '[1,32]+[128,128]' 取到 128，同一个 FP4 方案因此一个得名一个不得名。
  // 要给带 + 的格子推名字，走 mainFormat 那条路（配对关系有数据支撑）。
  if (block.includes('+')) return undefined;
  const m = block.match(/(\d+)\s*\]?\s*$/);
  return m ? Number(m[1]) : undefined;
}

/** 元素格式是第三个因子，**少验一个就会把 FP8 认成 MXFP4**。 */
const IS_E2M1 = new Set(['FP4', 'MXFP4', 'NVFP4']);

/**
 * 记录自己已经给了块格式名的，不再推 —— **推导只用来补空缺，不用来复述，更不用来反驳。**
 *
 * `minimax-m3.1-flash` 的专家记作 `NVFP4`，而签名（块 16 + E4M3）在这里会推出
 * 「NVFP4 式」。两者看着像同一句话，其实差一个关键事实：严格的 NVFP4 有**第二级
 * FP32 全局 scale**，「…式」是专门留给没有它的那种。而那份记录实测到了
 * 21,888 个 weight_global_scale 与同样多的 weight_packed 一一对应，**第二级 scale
 * 在**，所以它是严格 NVFP4；DeepSeek 的全局 KV 才是「式」——技术报告原话
 * "following NVFP4 but omitting its second-level global scale"。
 * 推导值盖在记录值旁边会把这个区别抹平，所以 dtype 已是块格式名时直接让位。
 */
const NAMED_BLOCK_FORMAT = new Set(['MXFP4', 'NVFP4']);

export function blockFormat(dtype: unknown, block: unknown, scale: unknown): string | undefined {
  // 初版只验了块与 scale 两个因子，于是 `FP8_E4M3 · 块 32 · scale E8M0` 的滑窗
  // 被标成了 MXFP4 —— 那是 8 bit 元素，怎么也成不了 FP4。本仓「一个乘式逐因子验」
  // 说的就是这个：三个因子只验两个，剩下那个照样能错一个量级。
  // 元素不是 E2M1 就一律不给名字（MXFP8 之类另有定义，但本仓没有一手依据去断言）。
  const dt = typeof dtype === 'string' ? dtype.toUpperCase() : '';
  if (!IS_E2M1.has(dt)) return undefined;
  if (NAMED_BLOCK_FORMAT.has(dt)) return undefined; // 记录已经给了名字，让位
  const n = blockSize(block);
  const sc = typeof scale === 'string' ? scale.toUpperCase() : '';
  if (!n) return undefined;
  if (n === 16 && sc === 'FP8_E4M3') return 'NVFP4 式';
  if (n === 32 && sc === 'FP8_E8M0') return 'MXFP4';
  return undefined;
}

/** 一个子系统里的一套量化方案（记录的 `per_subsystem_schemes`，只有带 `+` 的格子有）。 */
export type Scheme = {
  dtype: string;
  block?: string;
  scale_dtype?: string;
  tensors: number;
  elements: number;
};

/** 按元素数降序 —— **第一条就是主方案**。记录里本来就是这个序，这里不信它。 */
export const sortSchemes = (rows: Scheme[] | undefined): Scheme[] =>
  (rows ?? []).slice().sort((a, b) => b.elements - a.elements);

/**
 * 带 `+` 的格子退一步：按**主方案**（元素数最多的那一条）的三因子推名字。
 *
 * 不这么做的代价不是「少一个名字」：`expert` 显示 MXFP4、`draft` 显示裸 FP4，
 * 可两格 4 bit 那一半逐因子相同（E2M1 + [1,32] + E8M0），读者横着看得到的是
 * 「草稿换了格式」，一个错结论。
 *
 * **主方案的 dtype 要和这一格的代表值一致才认。** 不一致说明记录自相矛盾，
 * 那时返回 undefined —— 页面不替记录在两个矛盾值里挑一个。
 */
export function mainFormat(dtype: unknown, schemes: Scheme[]): string | undefined {
  const main = schemes[0];
  if (!main || typeof dtype !== 'string') return undefined;
  if (main.dtype.toUpperCase() !== dtype.toUpperCase()) return undefined;
  return blockFormat(main.dtype, main.block, main.scale_dtype);
}

/**
 * 逐方案一行，带条数与元素占比。
 *
 * **这一行是「代表值」三个字的兑付。** 整列都是代表值（`attention` 那格同样盖掉了
 * BF16 的 norm），平时不碍事；但 `draft` 推出 `MXFP4` 之后，不写明它只说 95.5% 那一半，
 * 「草稿头全是 4 bit」就是个很自然的误读 —— 而草稿头的注意力投影其实停在 FP8。
 * 未量化的档也列出来，所以占比合得上 100%，不会让人以为漏了。
 */
export function mixSummary(sch: Scheme[]): string | undefined {
  const tot = sch.reduce((s, r) => s + r.elements, 0);
  if (sch.length < 2 || !tot) return undefined;
  // 小数档不印 `0.0%` —— 那读着像零，而它们是实测到的 20 条 BF16 与 27 条 FP32，
  // 恰恰是精度页「小而对误差敏感的块反而被提上去」那条结论的样本，不能显示成不存在
  const pct = (n: number) => {
    const p = (n / tot) * 100;
    return p > 0 && p < 0.05 ? '<0.1%' : `${p.toFixed(1)}%`;
  };
  return sch
    .map((r) => `${r.dtype}${r.block ? ` 块 ${r.block}` : ''} ${r.tensors} 条 / ${pct(r.elements)}`)
    .join('；');
}
