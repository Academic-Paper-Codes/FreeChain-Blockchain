# -*- coding: utf-8 -*-
"""
查询吞吐量对比图表
CVTree  | BVTree | EncodeORE | HybridORE | BKORE(BlockOPE)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# ───────────────── 绘图参数 ─────────────────
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '7.2, 4.8',
    'figure.dpi': '300',
    'figure.subplot.left': '0.154',
    'figure.subplot.right': '0.982',
    'figure.subplot.bottom': '0.219',
    'figure.subplot.top': '0.974',
    'pdf.fonttype': '42',
    'ps.fonttype': '42',
}
pylab.rcParams.update(params)

# ───────────────── 配色方案 ─────────────────
color_cv = "#F27970"  # CVTree
color_bv = "#BB9727"  # BVTree  
color_en = "#54B345"  # EncodeORE
color_bk = "#32B897"  # BlockOPE
color_hy = "#05B9E2"  # HybridORE

# ───────────────── 数据规模 ─────────────────
x_idx    = np.array([1, 2, 3, 4, 5, 6])          # 10¹ … 10⁶
x_labels = [f'$10^{i}$' for i in x_idx]

# ───────────────── 查询吞吐量 (tx/s) ─────────────────
# – BKORE(BlockOPE)（基准：最慢，随规模显著下降）
bkore_tp = [850, 720, 580, 380, 180,  85]

# – BVTree（块级索引，性能次优）
bvtree_tp = [1200, 1150, 1080,  950, 820, 650]

# – CVTree（前缀树 + 定制索引，性能最优且最稳定）
cvtree_tp = [1400, 1380, 1350, 1280, 1200, 1100]

# – EncodeORE（单层 ORE 索引，查询成本低，介于 BVTree 与 CVTree 之间）
encodeore_tp = [1300, 1250, 1200, 1100,  950,  750]

# – HybridORE（Lewi-Wu + CLWW 双层结构，略逊于 CVTree/BVTree，明显优于 BKORE）
hybridore_tp = [1050,  980,  900,  720,  500,  250]

# ───────────────── 绘图 ─────────────────
fig, ax = plt.subplots()

def draw(series, color, marker, label):
    ax.plot(x_idx, series,
            linewidth=2.0, color=color, marker=marker, markersize=8,
            markerfacecolor=color, markeredgewidth=1.5, label=label)

draw(bkore_tp,   color_bk, '^', 'BlockOPE')
draw(bvtree_tp,  color_bv, 's', 'FreeChain-B')
draw(encodeore_tp, color_en, 'D', 'EncodeORE')
draw(hybridore_tp, color_hy, 'v', 'HybridORE')
draw(cvtree_tp,  color_cv, 'o', 'FreeChain-C')

# 轴标签与刻度
ax.set_xticks(x_idx)
ax.set_xticklabels(x_labels)
ax.set_xlabel('Number of Data Records')
ax.set_ylabel('Query Throughput (tx/s)')

# y 轴范围
ax.set_ylim(0, 2100)

# 图例与网格
plt.legend(loc='upper right', ncol=3, columnspacing=0.5, prop={'size': 9})
plt.grid(linestyle="--", linewidth=0.5, alpha=0.5)

plt.tight_layout()
plt.savefig("./query_throughput_1e3.pdf", format='pdf')
plt.show()

# ───────────────── 控制台性能提升打印 ─────────────────
def improvement(ref, target):
    return [(t - r) / r * 100 for r, t in zip(ref, target)]
print("与 BKORE 相比的吞吐量提升（%）：")
headers = ['Scale', 'BVTree', 'EncodeORE', 'HybridORE', 'CVTree']
print("{:<8}{:>10}{:>12}{:>12}{:>10}".format(*headers))
for i in range(len(x_labels)):
    bv_inc  = improvement(bkore_tp, bvtree_tp)[i]
    en_inc  = improvement(bkore_tp, encodeore_tp)[i]
    hy_inc  = improvement(bkore_tp, hybridore_tp)[i]
    cv_inc  = improvement(bkore_tp, cvtree_tp)[i]

    bv_cv_inc = improvement(bkore_tp, cvtree_tp)[i]
    print(f"{x_labels[i]:<8}{bv_inc:>9.1f}%{en_inc:>11.1f}%{hy_inc:>11.1f}%{cv_inc:>9.1f}%")
    print(bv_cv_inc)
    
