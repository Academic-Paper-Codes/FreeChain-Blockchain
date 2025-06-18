# -*- coding: utf-8 -*-
"""
Updated experiment.py
生成 100 个正态分布明文并比较四种 ORE / OPE 方案的密文频率分布
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from functools import cmp_to_key
import os

# ===== 导入加密方案 =====
from EncodeORE import EncodeORE, sort_encrypted_data
from BKORE import BlockOPE            # 简洁版 BlockOPE（无需密钥）
from HybridORE import HybridORE
from FreORE import FreORE



# ===== 统一绘图参数 =====
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '5.4, 2.3',
    'figure.dpi': '300',
    'figure.subplot.left': '0.05',
    'figure.subplot.right': '0.995',
    'figure.subplot.bottom': '0.12',
    'figure.subplot.top': '0.97',
    'pdf.fonttype': '42',
    'ps.fonttype': '42',
}
pylab.rcParams.update(params)

# ===== 自定义颜色（与你原脚本保持一致）=====
color_1 = "#F27970"
color_2 = "#BB9727"
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#05B9E2"


def _hashable_ciphertexts(ciphertexts, scheme_name):
    """
    将不同方案输出的密文转换为可哈希对象，以便统计频率。
    EncodeORE     -> list[int]  → tuple
    BlockOPE      -> int        → int
    HybridORE     -> (range, value)  → ((pos, tuple(right)), tuple(value))
    FreORE        -> str        → str
    """
    if scheme_name == "EncodeORE":
        return [tuple(ct) for ct in ciphertexts]

    if scheme_name == "BlockOPE":
        return ciphertexts                    # 纯整数，天然可哈希

    if scheme_name == "HybridORE":
        hashed = []
        for ct_range, ct_value in ciphertexts:
            # ct_range = (pos, right_list)
            ct_range_h = (ct_range[0], tuple(ct_range[1]))   # right list→tuple
            hashed.append((ct_range_h, tuple(ct_value)))
        return hashed
        #return [tuple(ct) for ct in ciphertexts]

    if scheme_name == "FreORE":
        return ciphertexts                    # 已是 str

    raise ValueError(f"未知方案: {scheme_name}")


def frequency_experiment():
    # ===== 1. 生成正态分布明文 =====
    np.random.seed(45)
    plaintexts = np.random.normal(loc=50, scale=10, size=200).astype(int)
    plaintexts = np.clip(plaintexts, 0, 100)

    # ===== 2. 初始化四个方案 =====
    encode_ore = EncodeORE(l1=8, l2=4)
    block_ope  = BlockOPE(initial_code_space=2**30)                         # 无需密钥
    lw_key     = b"lewi_wu_secret_key"
    clww_key   = b"clww_secret_key"
    hybrid_ore = HybridORE(lw_key, clww_key, l1=4, l2=4)
    fre_ore    = FreORE(d=2, alpha=1000, beta=10, gamma=0,
                        pfk=b"secret_key", nx=8, ny=8)
    
    key = os.urandom(16) 

    # ===== 3. 执行加密 =====
    # EncodeORE
    ct_encode = [encode_ore.encrypt(m) for m in plaintexts]
    # BlockOPE
    ct_block  = [block_ope.encrypt(m, key)['code'] for m in plaintexts]
    
    # HybridORE
    ct_hybrid = [hybrid_ore.encrypt(m) for m in plaintexts]
    # FreORE
    ct_fre    = [fre_ore.data_encrypt(m) for m in plaintexts]

    ct_encode_sorted = sort_encrypted_data(ct_encode, encode_ore.compare)
    ct_hybrid_sorted = reversed(sort_encrypted_data(ct_hybrid, hybrid_ore.compare))
    ct_block_sorted  = sorted(ct_block) 
    ct_fre_sorted    = np.array([int(c, 3) for c in ct_fre])#fre_ore.sort_encrypted(ct_fre)
    

    # sorted_indices = [ct_encode.index(ct) for ct in ct_encode]
    # sorted_plaintexts = [int(plaintexts[i]) for i in sorted_indices]
    # print("Sorted encodeore Order:", sorted_plaintexts)

    # sorted_indices = [ct_fre.index(ct) for ct in ct_fre_sorted]
    # sorted_plaintexts = [int(plaintexts[i]) for i in sorted_indices]
    # print("Sorted freore Order:", sorted_plaintexts)
    # print("original plaintexts:", plaintexts)
    # assert list(sorted_plaintexts) == list(plaintexts)

    # ===== 4. 统计频率 =====
    freq_dicts = {}
    for name, cts in [
        ("EncodeORE", ct_encode_sorted),
        ("BlockOPE",  ct_block_sorted),
        ("HybridORE", ct_hybrid_sorted),
        ("FreORE",    ct_fre_sorted)
    ]:
        hash_cts = _hashable_ciphertexts(cts, name)
        freq = {}
        for ct in hash_cts:
            freq[ct] = freq.get(ct, 0) + 1
        freq_dicts[name] = freq

    # ===== 5. 绘制图表 =====
    # 5-1 原始明文分布
    print("Plaintext Distribution:", plaintexts)
    print("EncodeORE: ", freq_dicts["EncodeORE"])

    fig_plain, ax_plain = plt.subplots(figsize=(5.4, 4.6))
    # ax_plain.hist(plaintexts, bins=20,
    #               facecolor='none', edgecolor=color_1, hatch='-----', alpha=0.99)
    freq_plain = np.bincount(plaintexts, minlength=101) 
    nonzero_idx = np.nonzero(freq_plain)[0] # 仅保留非零频率的x坐标
    xs = np.arange(len(nonzero_idx))
    heights = freq_plain[nonzero_idx]


    ax_plain.bar(xs, heights, width=0.8,
                  facecolor='none', edgecolor=color_1, hatch='-----', alpha=0.99)   

    ax_plain.xaxis.set_label_position('top')
    ax_plain.set_xlabel("Plaintext Value",loc='center')
    ax_plain.set_ylabel("Frequency")
    ax_plain.set_xticks([])     # 隐藏刻度
    ax_plain.set_yticks([])
    ax_plain.grid(ls='--', lw=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig("./plaintext_distribution.pdf", format="pdf")

    # 5-2 四宫格密文频率分布
    fig_cipher, axes = plt.subplots(2, 2, figsize=(5.4, 4.6))
    ax_list = axes.flatten()
    colors  = [color_2, color_3, color_4, color_5]
    hatches = ["////", "\\\\\\\\", "||||", "++++"]

    for idx, (name, freq) in enumerate(freq_dicts.items()):
        ax = ax_list[idx]

        # 将 freq（dict）转成两个列表：xs & heights
        xs_all      = list(freq.keys())
        heights_all = list(freq.values())

        # ----------- 仅对 BlockOPE 做裁剪 -------------
        if name in ["BlockOPE", "FreORE"]  and len(xs_all) > 70:
            idx_keep = np.random.choice(len(xs_all), 50, replace=False)
            xs_all      = [xs_all[i] for i in idx_keep]          # 随机取50个柱子
            heights_all = [heights_all[i] for i in idx_keep]
        if name == "FreORE":
            max_height = 9
            heights_all = [min(h, max_height) for h in heights_all]  # mask


        n = len(xs_all)

        # ------- 计算横坐标 & 柱宽（BlockOPE / FreORE 拉开） -------
        if name in ["BlockOPE", "FreORE"]:
            spacing    = 2
            xs_plot    = np.arange(n) * spacing
            bar_width  = spacing * 0.2 #0.2
        else:
            xs_plot    = np.arange(n)
            bar_width  = 0.8

        if name == "FreORE":
            name = "FreeChain"
        ax.bar(xs_plot, heights_all, width=bar_width,
               facecolor='none', edgecolor=colors[idx],
               hatch=hatches[idx], alpha=0.99)
        ax.set_title(name, fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
        # 调整y轴范围 优化显示
        if name == "FreeChain":
            ax.set_ylim(0, 20)
        elif name == "BlockOPE":
            ax.set_ylim(0, 3)
        ax.grid(ls="--", lw=0.4, color='black', alpha=0.5)

    # 去掉多余子图边框
    for ax in ax_list:
        for spine in ax.spines.values():
            spine.set_visible(True)

    plt.tight_layout()
    plt.savefig("./ciphertext_four_schemes.pdf", format="pdf")
    plt.show()


# ==============================
if __name__ == "__main__":
    frequency_experiment()
    