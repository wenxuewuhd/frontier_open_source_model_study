#!/usr/bin/env python3
"""闸口：渲染出来的东西和作者想表达的是不是一回事。

**这是本仓第一道锚在产物上的闸。** 其余几道都锚在数据或代码上：
`validate.py` 查记录的语义与契约、`kv_table.py --check` 查文档的生成区、
站点构建查语法。四道全绿的时候，全站 **3,665 处字面 `**`** 正显示给读者 ——
记录散文是按 Markdown 写的（粗体 1,712 处、反引号 199 处），而站点一处 Markdown
渲染都没有，散文被当纯文本插进页面。没有任何一道闸问过「页面上最后长什么样」。

这正是 `model-audit` skill 规则 11 的另一面：锚在内侧的闸守不住外侧。
数据对、代码对、构建过，产物照样可以是错的。

查两件事，都只看**读者看得见的正文**（剥掉标签、脚本与样式）：

  1. 成对的 `**…**`  —— 作者想要粗体，读者看到星号
  2. 成对的 `` `…` `` —— 作者想标张量名，读者看到反引号

**判据刻意不复用渲染器的 tokenizer。** 复用的话，渲染器漏解什么，这道闸就跟着
漏查什么 —— 它会继承被它检查的那个东西的盲区，等于没查。这里用一条独立的、更粗的
规则：正文里就不该出现成对的标记，不管渲染器怎么想。

**落单的 `**` 不算。** `head_dim**-0.5` 是幂运算，不是写漏一侧的粗体；
按「成对」判才分得开这两件事。

用法:
  python3 scripts/check_rendered.py              # 需要先 npm run build
  python3 scripts/check_rendered.py --dist DIR
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 成对的行内标记。限制在一行之内、长度有上限 —— 跨段落的两个星号凑一对是巧合，不是标记。
BOLD = re.compile(r"\*\*([^*\n]{1,80})\*\*")
CODE = re.compile(r"`([^`\n]{1,60})`")
DROP = re.compile(r"<(script|style|template)[^>]*>.*?</\1>", re.S)
TAG = re.compile(r"<[^>]+>")


def visible_text(html: str) -> str:
    """读者看得见的正文。属性里的内容不算 —— title= 里有标记是另一回事。"""
    return TAG.sub(" ", DROP.sub(" ", html))


def check(dist: str, problems: list, stale: list) -> int:
    pages = sorted(glob.glob(os.path.join(dist, "**", "*.html"), recursive=True))
    if not pages:
        problems.append(f"{os.path.relpath(dist, ROOT)} 下一个 html 都没有 —— "
                        "先 `cd site && npm run build`，这道闸查的是构建产物")
        return 0

    newest_src = 0.0
    for p in glob.glob(os.path.join(ROOT, "site", "src", "**", "*"), recursive=True):
        if os.path.isfile(p):
            newest_src = max(newest_src, os.path.getmtime(p))
    oldest_page = min(os.path.getmtime(p) for p in pages)
    if newest_src > oldest_page:
        # 产物比源码旧，这道闸查的就是上一版的页面 —— 绿了也不说明什么
        stale.append("site/dist 比 site/src 旧，先重新构建再查，否则查的是上一版产物")

    for p in pages:
        rel = os.path.relpath(p, ROOT)
        text = visible_text(open(p, encoding="utf-8").read())
        for pat, what, why in (
            (BOLD, "**…**", "作者想要粗体，读者看到的是星号"),
            (CODE, "`…`", "作者想标张量名或字段名，读者看到的是反引号"),
        ):
            hits = pat.findall(text)
            if hits:
                sample = "、".join(f"「{h.strip()[:24]}」" for h in hits[:3])
                problems.append(
                    f"{rel}: 正文里有 {len(hits)} 处没解开的 {what}（{sample}）—— {why}。"
                    "散文字段要走 <Prose>，不要直接 {x.text}")
    return len(pages)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", default=os.path.join(ROOT, "site", "dist"))
    args = ap.parse_args()

    problems: list = []
    stale: list = []
    n = check(args.dist, problems, stale)

    print(f"检查 {n} 个构建产物页面")
    for s in stale:
        print(f"  warn  {s}")
    for p in problems:
        print(f"  ERROR {p}")
    print(f"\n{len(problems)} 个错误, {len(stale)} 个告警")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
