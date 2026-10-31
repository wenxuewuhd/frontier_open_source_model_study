"""张量名 → 组件 → 子系统的归类。

只此一份。derive.py 按它分参数量与带宽，fetch_dtypes.py 按它挑代表张量测精度 ——
两边用不同的归类会让「参数量里的 expert」和「精度表里的 expert」指的不是同一批张量。

**两层**。生成类的一份权重里常常同时装着 DiT、文本编码器、VAE、声码器、上采样器，
它们是不同的模型：MiniMax-H3 的文本编码器是 33B 的 Qwen3-VL-32B，LTX-2.3 的单文件里
DiT、VAE、audio VAE、vocoder 四样打包在一起。先按顶层前缀分到组件（component()），
再在组件内跑子系统正则（classify()）—— 不分这一层，text_encoder 的 mlp 会和 DiT 自己的
FFN 合成一桶，一个外部文本编码器被算进 DiT 的参数量。

语言模型的张量没有组件前缀，全部落进缺省桶 backbone，现有行为不变。

顺序即优先级，先匹配先得。draft 排在 expert 之前是有意的：
mtp.N.ffn.experts.N.w1 属于草稿侧，它的权重要另算一份，不能并进主干 MoE。
"""
import json
import os
import re

SUBSYSTEM = [
    # ngram_embedding / ple_embedding 是 Qwen3.8-Flash-Next 的 n-gram 查表（128 个分片
    # 加 3 条索引元数据）。engram 差一个 e 认不出它 —— 51.2B 整块落进 other，而 other 在
    # active_params 的常驻名单里，于是带宽口径激活偏高 8.7 倍（6.67B → 57.87B）。
    # 这张表每 token 只读 16 行 × 160 维 = 5,120 B，不是常驻
    ("retrieval", re.compile(r"engram|memory_table|retriev|ngram_embedding|ple_embedding", re.I)),
    # projector / patch_merge 是视觉到语言主干的投影（MiniMax-M3 的 multi_modal_projector 与
    # patch_merge_mlp、Kimi-K3 的 mm_projector、Step-5 的 vit_large_projector）。不加的话
    # patch_merge_mlp 因为带 mlp 落进 dense_ffn，其余落进 other
    # \bvit[._] 而不是 vit\.：Step-3 的两级下采样卷积叫裸的 vit_downsampler / vit_downsampler2
    # （4 个张量、331,362,304 参数），没有点号也没有 vision_model. 前缀，要求字面点号就落进 other
    ("vision", re.compile(r"vision|visual|\bvit[._]|aligner|image_|projector|patch_merge", re.I)),
    # **`audio` 必须写成 `audio_<具体词>`，不能是裸 `audio`。** 它原先是
    # `audio|speech|whisper`，于是命中每一个 `audio_*` 名字 —— 音视频联合 DiT 里，
    # LTX 的整条音频支路（ltx-2 57.14 亿、ltx-2.3 57.35 亿）连带它的注意力、FFN、
    # 调制、时间步嵌入整块落进「音频」一桶。那是**按流分桶**，而本仓的 param_scaling
    # 轴已经把「流」单列（per_sample / per_video_token / per_text_token），
    # 子系统这一层必须按**功能**分才和它一致。收紧后这一条只认真正的音频塔
    # （理解类多模态那一侧），生成侧的音频支路交给各功能桶。
    # **只挪位置无效** —— 它在任何位置都命中每个 `audio_*`，必须改模式。
    ("audio", re.compile(r"audio_tower|audio_encoder|audio_model|audio_embed"
                         r"|speech|whisper", re.I)),
    # eh_proj / enorm / hnorm / shared_head 是 DeepSeek V3 一代 MTP 模块的名字 ——
    # 那一代 MTP 借用层索引（第 61 层），名字里没有 mtp，只按 mtp 匹配会把它算进主干
    ("draft", re.compile(r"mtp|draft|dspark|eagle|eh_proj|enorm|hnorm|shared_head", re.I)),
    # \.moe\.(gate|up|down)_proj 是 Step-5 的 3 维堆叠专家 [352, …]，名字里没有 experts ——
    # 不认它就落进 dense_ffn，激活参数会把全部专家算成常驻
    ("expert", re.compile(r"\.experts?\.\d+\.|\bexperts\.|\.moe\.(gate|up|down)_proj\b", re.I)),
    # Step-5 写成 share_expert（单数、无 d）
    ("shared_expert", re.compile(r"shared?_expert", re.I)),
    ("router", re.compile(r"\bgate\b|router|e_score|expert_bias", re.I)),
    ("lm_head", re.compile(r"\blm_head\b|output_layer|^head\.weight$", re.I)),
    # word_embeddings 是 chatglm 一代的写法（glm-4-9b：transformer.embedding.word_embeddings）。
    # 不认它，输入词表 620,756,992 落进 other 被当成常驻，active_params 偏高 7.07%
    ("embed", re.compile(r"embed_tokens|\bwte\b|embed_in|^embed\.weight$|word_embeddings", re.I)),
    # 必须排在 attention / dense_ffn 之前：mHC 的系数叫 hc_attn_* / hc_ffn_*，
    # 名字里带 attn 和 ffn，但它们是残差系数不是注意力或 FFN 权重，
    # 而且往往是 FP32 —— 混进去会把那两个子系统的精度带偏
    # _res_proj 是 Kimi-K3 的 Attention Residuals 系数（self_attention_res_proj / mlp_res_proj /
    # output_attn_res_proj，均 [1, hidden]）；对应的 *_res_norm 仍归 norm
    # hyper_connection 是 Qwen3.8-Flash-Next 对同一机制的另一套命名
    # （attn_hyper_connection / mlp_hyper_connection / hyper_connection_mixer）。
    # 它必须跟着这一条留在 attention 与 dense_ffn 之前，否则 attn_* 被 attn 抢走、
    # mlp_* 被 mlp 抢走 —— 后者会让一个没有任何稠密 FFN 的模型显示出 316,538,880 的 dense_ffn
    # hyper_connection(?!.*norm)：排除 *_hyper_connection.hc_norm.weight（Flash-Next 97 个）。
    # 本条排在 norm 之前，不加这个否定前瞻就会把 hc_norm 从 norm 抢走，
    # 违反「带 norm 的就是归一化，没有例外」那条规矩
    ("residual", re.compile(
        r"\bhc_[a-z]+_(base|fn|scale)\b|^hc_|residual|_res_proj\b|hyper_connection(?!.*norm)",
        re.I)),
    # **modulation 在这张表里出现两次，这不是笔误。**
    # CogVideoX 把 AdaLN 调制实现成 `transformer_blocks.N.norm1.linear` ——
    # 名字里带 norm，所以会被下面那条 norm 规则领走。后果不是「少一个桶」：
    # 794,271,744 个参数（总量的 14.26%）被标成「归一化」，而**这一份记录的全部意义
    # 就是它是「逐层放矩阵」那组的代表**，页面上却会说它没有调制子系统。
    # 所以这一条必须排在 norm 之前，而尾部那条 modulation 不能上移 ——
    # 上移会从 attention 与 audio 抢走 LTX 的 54 与 103–175 个张量（实测过）。
    # 全仓只有 cogvideox-1.5-5b 有 normN.linear 这种写法，168 条，
    # 抢占仅此一处、非生成类受影响为 0。
    #
    # **`norm_out.linear` 是第二刀，第一版漏了。** `\bnorm\d*\.linear\b` 要求数字，
    # 配不上最后一层的 `norm_out.linear`：改完之后 cogvideox 新 norm 桶的 85.4%、
    # h3 的 98.0% 仍是调制（v2-cog 发现）。补上之后出现一个**逐位恒等式**：
    #     modulation + time_embed == generation.backbone.param_scaling.per_sample
    # 在此之前我两次把这两个数说成「量级吻合但不是同一个量」——**它本来就是同一个量，
    # 只是张量在错桶里**。
    #
    # **但它不是一条律，是一条带前提的判据。** 四份有 `per_sample` 字段的里三份成立：
    #     cogvideox-1.5-5b    797,423,616 +      1,836,032 ==      799,259,648  ✓
    #     minimax-h3       13,039,369,728 +     15,835,008 ==   13,055,204,736  ✓
    #     stepvideo-t2v-30b   228,311,040 +     39,333,888 ==      267,644,928  ✓
    #     hunyuanvideo-1.5  2,744,414,208 +     20,983,808 !=    2,723,964,928
    #
    # **hunyuan 那 41,433,088 是口径差不是漏刀，两个方向都对平过**（v2-ltx 拆出来的）：
    #     per_sample 2,723,964,928 = double_blocks.*_mod.linear 2,719,236,096
    #                              + time_in.mlp 4,722,688
    #                              + cond_type_embedding 6,144   ← 它在 other 桶里
    #     差 41,433,088 = txt_in.* 的 mod/time_embed 33,046,528
    #                   + final_layer.adaLN_modulation 8,392,704 − 6,144
    #
    # 前提是**那份记录的 `breakdown` 本身是纯功能分解**。hunyuan 不满足：它的
    # `breakdown.token_refiner` 141,105,152 是**按模块**切的，其中 23.4%
    # （33,046,528）是 AdaLN 与时间步嵌入，功能轴会把这个桶切开；
    # 另外 `final_layer.adaLN_modulation` 压根不在它的 `adaln_modulation` 里。
    # cogvideox / h3 / stepvideo / ltx 两份的 breakdown 都是纯功能分解，所以都成立。
    #
    # **所以判据要带前提写**：breakdown 是纯功能分解时，`modulation + time_embed`
    # 必须等于它的 `adaln_modulation` / `per_sample`；不等就先去看 breakdown 里
    # 有没有按模块切的桶 —— 这样它既能当校验器，又不会在下一份带 `token_refiner`
    # 这类桶的记录上误报。
    #
    # **这一条自己就是「断言规律不要断言人口普查」的实例**：我先按手上那两份把它
    # 写成了普适结论，v2-h3 与 v2-step 各自独立复算九份找出反例，v2-ltx 再把
    # 反例的根因拆平 —— 三步都不是我做的。
    ("modulation", re.compile(r"\bnorm(?:\d+|_out)\.linear\b", re.I)),
    # norm 必须排在 attention / dense_ffn 之前：attn_norm、ffn_norm、kv_norm、q_norm
    # 名字里带 attn / ffn / kv / q，按那两条先匹配会把归一化算成注意力或 FFN 权重。
    # 带 norm 的就是归一化，没有例外，所以提前判定是安全的
    ("norm", re.compile(r"norm|\bln\b|layernorm", re.I)),
    # ── 以下四条是生成类 DiT 的桶。**次序被四个约束同时钉死，每一条都踩出来过。** ──
    #
    #   1. 在 `vision` 之后。否则 `patch_embed` 抢走视觉塔的 ViT patch embedding，
    #      13 份语言模型记录受影响。
    #   2. 在 `norm` 之后。否则抢走 h3 的 `norm_out.linear` 之外的真 norm；
    #      而「名字带 norm 的调制投影」由上面那条单独的 modulation 在 norm 之前截。
    #   3. 在 `attention` / `dense_ffn` 之前。否则 `*cross_attn*scale_shift*` 被 `attn`
    #      抢走 —— ltx-2 的 60 条 / 120,557,568 就是这么丢的；`audio_ff.` 被 `ffn` 抢走。
    #   4. `time_embed` 在 `modulation` 之前。否则 `adaln` 先命中
    #      `*adaln_single.emb.timestep_embedder.*`，**时间步嵌入整桶消失** ——
    #      stepvideo 的 39,333,888 与 ltx-2.3 的 90,226,688 原先就是这么没的，
    #      而 ltx-2.3 那一格判的正是「不适用 = 这个子系统在这个模型里不存在」。
    #
    # **「合并成一条放在 norm 之前」行不通，v2-ltx 两条路都试了。** 把上面那条
    # `\bnorm(?:\d+|_out)\.linear\b` 换成全量 modulation 模式，cogvideox 的 7.94 亿
    # 整块退回 norm（撤销 ebfb3cd 的修正）；而合并后的 modulation 置于 norm 之前，
    # 就也在 time_embed 之前，于是约束 4 当场破。**modulation 出现两次不是冗余**：
    # 前一条管「名字带 norm 的调制投影」，这一条管其余，中间夹的 norm 与 time_embed
    # 各要截一刀。
    #
    # 「音频支路的调制归 modulation 还是 audio」这个按功能 / 按流的问题已定：按功能。
    # 依据是本仓 param_scaling 轴已把「流」单列，子系统再按流分就是两套口径。
    # 代价是 LTX 两份的数字要重写，由 v2-ltx 那条线做。
    #
    # 全仓 64 个非空事实目录 / 3,014,489 个张量上实跑：**有记录的非生成类受影响 0**，
    # 受影响 7 个目录（5 份生成记录 + ltx-2-pack / ltx-2.5，后两个还没有记录）。
    # **时间步嵌入器的 MLP 要在 dense_ffn 之前截下来，但只截带 `.mlp.` 的那种。**
    # HunyuanVideo 的两个时间步嵌入器 8 条 / 9,445,376 全算进了稠密 FFN，
    # 两条失效原因还不一样：`txt_in.t_embedder.mlp.*` 是 \bt_embedder\b 命中了
    # 但 time_embed 排在 dense_ffn 之后、`mlp` 先匹配先得（优先级问题）；
    # `time_in.mlp.*` 是词条里压根没有 `time_in` 这个写法（词表缺项）。
    #
    # **键在 `\.mlp\.` 上，不是把宽的那条 time_embed 整体前移。** 前移会同时跑到
    # modulation 前面，抢走 LTX 与 stepvideo 的 `adaln_single.emb.timestep_embedder.*`
    # （ltx-2.3 20 条 / 76,058,624、stepvideo 4 条 / 39,333,888），而 ltx-2.3 的
    # time_embed 这一格现在判的正是「不适用 = 这个子系统在这个模型里不存在」——
    # 那句假话会变成「桶里躺着 7,600 万个参数」。那些名字是 `...timestep_embedder.linear_1`，
    # **不带 `.mlp.`**，所以这条一条都碰不到。全仓实测：只有 hunyuanvideo-1.5 一处变动。
    #
    # 那个口径问题已经定了（见下面那段）：`adaln_single.emb.timestep_embedder` 归
    # time_embed。所以这一条现在只剩一个独有的活 —— `time_in.mlp.*`，
    # 宽的那条词表里没有 `time_in` 这个写法。
    ("time_embed", re.compile(r"(?:time_in|t_embedder|time_embedder|timestep_embedder)"
                              r"\.mlp\.", re.I)),
    ("time_embed", re.compile(r"timestep_embed|time_embedder|time_embedding|time_projection"
                              r"|\bt_embedder\b|c_embedder", re.I)),
    ("modulation", re.compile(r"adaln|ada_ln|\bmodulation\b|scale_shift"
                             r"|(?:_mod|_gate|_scale_shift|adaln|time_embed)\w*\.linear\b",
                             re.I)),
    # `\bpos_embed\b` 的词界不是风格，是判据：裸 `pos_embed` 会把 ltx-2.5 的
    # `keyframes_abs_pos_embedding` 捞进来，而那是一张绝对位置嵌入表、不是入口投影。
    # 三个追加词条各自堵的洞：stepvideo 的入口叫 `pos_embed` 而不是 `patch_embed`
    # （于是「入口/出口投影」桶只剩出口，少掉的一半在叫「其它」的桶里）、
    # stepvideo 的 `clip_projection` 与已有的 text_proj / caption_proj 同类、
    # h3 的 `context_embedder` 是它整个 other 桶（27,530,496）。
    # HunyuanVideo 的入口/出口投影在原有六个模式上一条都不命中，整批落在 other：
    # `txt_in.input_embedder` 7,342,080 + `byt5_in.fc1/2/3` 11,409,408
    # + `img_in.proj` 135,168 + `final_layer.linear` 65,568 = 18,952,224，
    # 是它 other 桶的 99.97%。剩下的 6,144 是 `cond_type_embedding`，**有意留在 other**：
    # 它不是投影，塞进 embed 会让「输入词表」在一个没有 token 词表的 DiT 上长出值。
    # `final_layer.adaLN_modulation` 仍归 modulation、`byt5_in.layernorm` 仍归 norm，
    # 因为这一条排在 modulation 与 norm 之后 —— 逐条点验过。
    # `patchify` 顺带吃掉 `unpatchify`，所以不再单列后者；
    # `\bproj_in\b` 改成 `(?:^|[._])proj_(?:in|out)\b` 是因为 `\b` 配不上
    # `audio_proj_in` —— 下划线本身就是 word char，词界在那里不成立。
    ("io_proj", re.compile(r"patch_embed|patchify|\bpos_embed\b|(?:^|[._])proj_(?:in|out)\b"
                           r"|text_proj|caption_proj|clip_proj|\bcontext_embedder\b"
                           r"|input_embedder|\bimg_in\b|\bfinal_layer\b|\bbyt5_in\b",
                           re.I)),
    # indexer / compressor 必须排在 attention 之前：它们挂在 attn. 下，名字带 attn，
    # 但精度和注意力主投影不同 —— DeepSeek V4 一代主投影是 FP8，压缩器整条与
    # indexer.wk 却是 BF16，并成一桶后表上只剩 FP8，看不出 BF16 那一截。
    # indexer 又必须排在 compressor 之前：attn.indexer.compressor.* 是 indexer
    # 自己产出 index K 的投影，归 indexer，不归产出全局 KV 的压缩器
    # index_[qk]_proj 是 MiniMax-M3 的 MSA 索引投影（挂在 self_attn 下、名字带 k_proj）；
    # 它的 index_[qk]_norm 照上面的规则归 norm，与 GLM 的 indexer.k_norm 同口径
    ("indexer", re.compile(r"indexer|\bindex_[qk]_proj\b", re.I)),
    ("compressor", re.compile(r"compressor", re.I)),
    # self_attention / query_key_value 是 chatglm 一代的写法：注意 attn 不是 attention 的子串
    # （a-t-t-n vs a-t-t-e-n-t-i-o-n），所以上面那串一个都不命中，1,426,247,680 落进 other。
    # self_attention.dense 靠 self_attention 命中就够，不要为它加 \bdense\b ——
    # 那会把 mlp.dense_4h_to_h 从 dense_ffn 抢走（本条排在 dense_ffn 之前）
    ("attention", re.compile(
        r"attn|self_attention|query_key_value|\bwq|\bwk|\bwv|\bwo\b"
    # **`o_proj` 要补左边界**：这三个字符就在 audi`o_proj`_in 里面（v2-ltx 试跑撞出来的）。
    # 现在被排在前面的 audio 挡着看不见，audio 一收紧，h3 的 4 条 I/O 投影就落进注意力。
    # 收紧代价实测为 0：全仓 o_proj 合法命中 3,130 条前面全是行首 / `.` / `_`，
    # 收紧后不再匹配的 18 种名字全是 audio_proj_in/out。坑 #2 的又一个实例 ——
    # 子串恰好是另一个词的中段。
        r"|q_proj|k_proj|v_proj|(?:^|[._])o_proj|kv_[ab]|q_[ab]", re.I)),
    # (?:^|\.)ff\. 是 diffusers 的 FeedForward（transformer_blocks.N.ff.net.0.proj）。
    # **a-t-t-n 不是 attention 的子串，f-f 也不是 ffn 的子串** —— 不认它的后果是
    # 五份生成记录的整个 FFN 落进 other：ltx-2 6.44B、h3 11.56B、cogvideox 3.17B、
    # stepvideo 14.50B、ltx-2.3 6.45B。要求前面是行首 / `.` / `_`，所以 `...diff.` 不会误中。
    # `_` 是后来补的：LTX 的 `audio_ff.` 前面是下划线，audio 收紧之后不补这个
    # 就有 16.1 亿掉进 other
    ("dense_ffn", re.compile(r"mlp|ffn|gate_proj|up_proj|down_proj|w1|w2|w3"
                             r"|(?:^|[._])ff\.", re.I)),
]

# 展示顺序与中文名存在 data/schema/subsystems.json，站点侧读同一个文件 ——
# 在两种语言里各写一份顺序，迟早会在加子系统时只改一边。
_SPEC = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "schema", "subsystems.json")))["subsystems"]

ORDER = [s["key"] for s in _SPEC]
LABEL = {s["key"]: s["label"] for s in _SPEC}


_COMPONENTS = json.load(open(os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "schema", "components.json")))["components"]

COMPONENT_ORDER = [c["key"] for c in _COMPONENTS]
COMPONENT_LABEL = {c["key"]: c["label"] for c in _COMPONENTS}

# 前缀 -> 组件。最长前缀优先：MAGI-2 的 model.diffusion_model.video_embeddings_connector.
# 既匹配 dit 的 model.diffusion_model. 也匹配 connector 的全称，按长度排就不会被 dit 抢走。
_COMPONENT_PREFIXES = sorted(
    ((pfx, c["key"]) for c in _COMPONENTS for pfx in c.get("prefixes", [])),
    key=lambda kv: -len(kv[0]),
)
DEFAULT_COMPONENT = "backbone"


def component(name: str) -> str:
    """张量名 -> 顶层组件。

    没有任何组件前缀的落 backbone —— 语言模型的全部张量都走这一支，
    所以加这一层不改变语言模型侧的任何输出。
    """
    for pfx, key in _COMPONENT_PREFIXES:
        if name.startswith(pfx):
            return key
    return DEFAULT_COMPONENT


# 组件内还值得按子系统细分的只有主干两类。VAE / 声码器 / 上采样器的内部结构
# 对负载账没有意义，整块算；文本编码器是另一个模型，它的内部拆分属于它自己那份记录。
SPLIT_INSIDE = {"backbone", "dit", "refiner"}


def classify2(name: str, mtp_from: int = None):
    """张量名 -> (组件, 子系统)。

    SPLIT_INSIDE 之外的组件不再细分，子系统直接返回组件名本身 ——
    这样 derive.py 既可以按组件汇总，也可以在主干内按子系统拆，用的是同一次遍历。
    """
    comp = component(name)
    if comp not in SPLIT_INSIDE:
        return comp, comp
    return comp, classify(name, mtp_from)


_LAYER_IDX = re.compile(r"layers?\.(\d+)\.", re.I)

# 视觉 / 音频塔自己也有 layers.N.，而塔的层数可以超过语言主干：Step-3 的塔
# （vision_model.transformer.layers.N.*）有 63 层，主干只有 61 层。不给层号捷径留旁路，
# 塔的第 61、62 层共 24 个张量会在任何名字正则之前被判成 draft，让一个没有 MTP 的模型
# 凭空长出 draft 桶，并让 mtp_layers_seen / draft_blocks_seen 出现假阳性。
# 后代（Step-5-Preview / Step-3.7-Flash）免疫只是因为塔改叫 resblocks.N.，配不上 layers?\.
_TOWER = re.compile(r"vision|visual|\bvit[._]|audio|speech|whisper", re.I)


def is_tower(name: str) -> bool:
    """这个张量属不属于视觉 / 音频塔。

    凡是按「层号」判断主干结构的地方都要先问这一句 —— 塔有自己的一套层号，
    和语言主干的层号不在同一个命名空间里。derive.py 统计 layer_ids / mtp_layer_ids
    时也用它，否则 classify() 这边绕开了、那边照样留假阳性。
    """
    return bool(_TOWER.search(name))


def classify(name: str, mtp_from: int = None) -> str:
    """张量名 -> 子系统。

    mtp_from：主干之后第一个 MTP 层的索引，传 config 的 num_hidden_layers 即可。

    DeepSeek V3 一代的 MTP 借用层索引（num_hidden_layers=61，主干是 0–60，
    MTP 是第 61 层），它的本体张量名里没有任何 mtp/draft 标记 —— 只有
    eh_proj / enorm / hnorm / shared_head 这几个胶水张量带标记。所以光靠名字匹配，
    第 61 层的 1536 个专家、注意力、norm、router、embed 副本会整块算进主干，
    参数量账上凭空多一层。层索引是唯一可靠的判据，但它只能由调用方给。

    层号捷径只对主干成立，所以带视觉 / 音频塔标记的张量要绕开它（见 _TOWER）。
    """
    if mtp_from is not None and not is_tower(name):
        m = _LAYER_IDX.search(name)
        if m and int(m.group(1)) >= mtp_from:
            return "draft"
    for label, rx in SUBSYSTEM:
        if rx.search(name):
            return label
    return "other"
