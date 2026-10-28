"""统一的 YAML 读取，关掉两个会静默出错的默认行为。

1. ISO 日期会被解析成 datetime 对象，而 schema 里声明的是 string + format:date。
   交给 schema 报错不如一开始就按字符串读。
2. PyYAML 走 YAML 1.1，科学计数法的指数必须带显式符号：
   748.0e9 会变成字符串，748.0e+9 才是浮点数。这个坑会反复咬人，直接补掉。

两条都只在子类上改，不污染全局 SafeLoader。
"""
import re

import yaml


class Loader(yaml.SafeLoader):
    """日期保持字符串；无符号指数的科学计数法正常识别为浮点。"""


# 每个首字符一份独立副本，避免改到 SafeLoader 自己的表
Loader.yaml_implicit_resolvers = {
    k: list(v) for k, v in yaml.SafeLoader.yaml_implicit_resolvers.items()
}
for _ch, _res in Loader.yaml_implicit_resolvers.items():
    Loader.yaml_implicit_resolvers[_ch] = [
        (tag, rx) for tag, rx in _res if tag != "tag:yaml.org,2002:timestamp"
    ]

# YAML 1.2 风格的浮点：指数符号可选
_FLOAT = re.compile(
    r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?[0-9][0-9_]*(?:[eE][-+]?[0-9]+)
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
    re.X,
)
Loader.add_implicit_resolver("tag:yaml.org,2002:float", _FLOAT, list("-+0123456789."))


def load(path):
    with open(path) as f:
        return yaml.load(f, Loader=Loader)
