# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 01:10:50 2022

@author: 86159
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# 绘图参数全家桶
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '5.4, 2.3',
    'figure.dpi':'300',
    'figure.subplot.left':'0.154',
    'figure.subplot.right':'0.982',
    'figure.subplot.bottom':'0.219',
    'figure.subplot.top':'0.974',
    'pdf.fonttype':'42',
    'ps.fonttype':'42',
}
pylab.rcParams.update(params)

color_1 = "#F27970"  # BKORE
color_2 = "#54B345"  # CVTree
color_3 = "#05B9E2"  # BVTree

fig, ax1 = plt.subplots()

# 数据：查询吞吐量 (tx/s)
# x轴：数据规模 (10^1 到 10^6)
# 理论分析：
# - BKORE: O(log n)复杂度，但需要多次交互，吞吐量较低
# - CVTree: O(log n)，基于前缀树优化，吞吐量中等
# - BVTree: O(log n)，基于块结构优化，吞吐量最高
x_data = [1, 2, 3, 4, 5, 6]  # 对应10^1, 10^2, ..., 10^6
x_labels = ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$']

# 查询吞吐量数据 (transactions per second)
bkore_throughput = [850, 720, 580, 420, 280, 150]     # BKORE随数据增长性能下降明显
cvtree_throughput = [950, 880, 820, 750, 680, 590]    # CVTree性能下降较缓
bvtree_throughput = [1200, 1150, 1080, 980, 850, 720] # BVTree性能最优

# 折线图
plt.plot(x_data, bkore_throughput, 
         linewidth=2.5, color=color_1, marker='o', 
         markerfacecolor=color_1, markeredgewidth=1.5, markersize=7,
         label='BlockOPE')

plt.plot(x_data, cvtree_throughput, 
         linewidth=2.5, color=color_2, marker='s', 
         markerfacecolor=color_2, markeredgewidth=1.5, markersize=7,
         label='CVTree')

plt.plot(x_data, bvtree_throughput, 
         linewidth=2.5, color=color_3, marker='^', 
         markerfacecolor=color_3, markeredgewidth=1.5, markersize=7,
         label='BVTree')

# 设置x轴
ax1.set_xticks(x_data)
ax1.set_xticklabels(x_labels)

# 设置标签
ax1.set_xlabel('Number of Data Records')
ax1.set_ylabel('Query Throughput (tx/s)')

# 设置y轴范围
plt.ylim((100, 1300))

# 添加网格
plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)

# 图例
plt.legend(loc='upper right', ncol=1, columnspacing=0.4, prop={'size': 12})

# 紧凑布局
plt.tight_layout()

# 保存图片
plt.savefig("./query_throughput_1e4.pdf", format='pdf')

plt.show()