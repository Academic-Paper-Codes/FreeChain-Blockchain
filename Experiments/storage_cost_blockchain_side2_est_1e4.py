# -*- coding: utf-8 -*-
"""
Storage-cost curves when blocksize = 1e4 (quick-draw version)
"""
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# ---------- 1. 图形参数 ----------
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '5.4, 2.3',
    'figure.dpi': '300',
    'figure.subplot.left':   '0.154',
    'figure.subplot.right':  '0.982',
    'figure.subplot.bottom': '0.219',
    'figure.subplot.top':    '0.974',
    'pdf.fonttype': '42',
    'ps.fonttype':  '42',
}
pylab.rcParams.update(params)

# ---------- 2. 预估数据 ----------
data_sizes   = [10**i for i in range(1, 7)]
x_labels     = ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$']
bkore_costs = [1,   18, 160, 2_800, 40_000, 500_000]
cvtree_costs = [0.6,  6,  55,   680,  9_200, 120_000]
bvtree_costs  = [0.25, 3,  35,   650,  8_500, 110_000]

# 颜色
color_1 = "#F27970"  # CVTree
color_2 = "#BB9727"  # BVTree
color_3 = "#54B345"  # BKORE

# ---------- 3. 绘图 ----------
fig, ax = plt.subplots()

ax.plot(range(len(data_sizes)), cvtree_costs, linewidth=2.0, color=color_1,
        marker='^', markerfacecolor=color_1, markeredgewidth=1.5, markersize=8,
        label='CVTree')
ax.plot(range(len(data_sizes)), bvtree_costs, linewidth=2.0, color=color_2,
        marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,
        label='BVTree')
ax.plot(range(len(data_sizes)), bkore_costs, linewidth=2.0, color=color_3,
        marker='o', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,
        label='BlockOPE')

ax.set_xticks(range(len(data_sizes)))
ax.set_xticklabels(x_labels)
ax.set_xlabel('Number of Data Records')
ax.set_ylabel('Storage Cost (KB)')
#ax.set_title('Storage costs on the blockchain side (blocksize = $10^4$)')
ax.set_yscale('log')
ax.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
ax.legend(loc='upper left')

plt.tight_layout()
plt.savefig('./storage_costs_blocksize_1e4.pdf', format='pdf')
plt.show()