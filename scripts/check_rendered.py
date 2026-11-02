#!/usr/bin/env python3
"""闸口：渲染出来的东西和作者想表达的是不是一回事。

**这是本仓第一道锚在产物上的闸。** 其余几道都锚在数据或代码上：
`validate.py` 查记录的语义与契约、`kv_table.py --check` 查文档的生成区、
站点构建查语法。四道全绿的时候，全站 **3,665 处字面 `**`** 正显示给读者 ——
记录散文是按 Markdown 写的（粗体 1,712 处、反引号 199 处），而站点一处 Markdown
渲染都没有，散文被当纯文本插进页面。没有任何一道闸问过「页面上最后长什么样」。

这正是 `model-audit` skill 规则 11 的另一面：锚在内侧的闸守不住外侧。
数据对、代码对、构建过，产物照样可以是错的。

前两件只看**读者看得见的正文**（剥掉标签、脚本与样式）：

  1. 成对的 `**…**`  —— 作者想要粗体，读者看到星号
  2. 成对的 `` `…` `` —— 作者想标张量名，读者看到反引号

第三件查的不是正文而是**链接目标在不在**：

  3. 站内死链 —— 页面发出去的 href 在产物里没有对应文件

**第 3 件加进来的理由和前两件同一个**：数据对、代码对、构建过，产物照样可以是错的。
总览页按 `tracking.yaml` 的 `model_id` 发链接，而那个字段的意思是
「这条记录将来的 id」，不是「这条记录已经存在」—— 于是每个「已选中、但还没录」
的候选都成了一条死链（首次发现 3 条：sana-video / hunyuanvideo-13b / magi-2-preview）。
schema 校验查不到它，构建也不报错：Astro 不知道 `url()` 拼出来的那个路径有没有页面。

**base 从产物反推，不在这里复制一份默认值。** 复制就会和 `astro.config.mjs`
分叉，而分叉那天这道闸查的是另一个站。反推拿不到 base 时**报错而不是跳过** ——
跳过的话这道检查会在前提消失的那天安静地变成空操作，绿得和「查过、没死链」一样。

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
# 站内链接。外链以 http / mailto 开头，纯锚点以 # 开头，都不在这道闸的范围里
HREF = re.compile(r'href="([^"]+)"')


def visible_text(html: str) -> str:
    """读者看得见的正文。属性里的内容不算 —— title= 里有标记是另一回事。"""
    return TAG.sub(" ", DROP.sub(" ", html))


def internal_hrefs(html: str) -> set:
    """站内 href，去掉锚点与查询串。"""
    out = set()
    for h in HREF.findall(html):
        if h.startswith("/"):
            out.add(h.split("#")[0].split("?")[0])
    return out


def site_base(by_page: dict) -> str | None:
    """从产物反推 base。

    Base.astro 每页都有一条 `url('/')` 的站标链接，渲染出来正是 `<base>/`，
    也就是全站最短的那个站内 href，且是其余站内 href 的公共前缀。
    这两条同时成立才认，否则返回 None 让调用方报错。
    """
    every = set().union(*by_page.values()) if by_page else set()
    if not every:
        return None
    shortest = min(every, key=len)
    if not all(h.startswith(shortest) for h in every):
        return None
    return shortest.rstrip("/")


def resolves(dist: str, base: str, href: str) -> bool:
    rel = href[len(base):].lstrip("/")
    if rel == "" or href.endswith("/"):
        return os.path.isfile(os.path.join(dist, rel, "index.html"))
    # 不带尾斜杠的目录链接也算 —— format: 'directory' 下它由服务端补斜杠
    return (os.path.isfile(os.path.join(dist, rel))
            or os.path.isfile(os.path.join(dist, rel, "index.html")))


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

    # 3. 站内死链
    by_page = {}
    for p in pages:
        by_page[p] = internal_hrefs(open(p, encoding="utf-8").read())
    base = site_base(by_page)
    if base is None:
        problems.append(
            "反推不出站点 base：产物里没有站内链接，或最短的那个不是其余的公共前缀。"
            "这道检查的前提没了，不是没死链 —— 先看 Base.astro 的站标链接还在不在")
    else:
        dead = {}
        for p, hrefs in by_page.items():
            for h in hrefs:
                if not resolves(dist, base, h):
                    dead.setdefault(h, []).append(os.path.relpath(p, ROOT))
        for h in sorted(dead):
            srcs = "、".join(sorted(set(dead[h]))[:3])
            problems.append(
                f"站内死链 {h}：产物里没有这个页面，而 {srcs} 在往那儿发链接。"
                "发链接之前要确认目标记录真的存在 —— "
                "tracking.yaml 的 model_id 是「将来的 id」，不是「已经录了」")
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
