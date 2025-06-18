# -*- coding: utf-8 -*-
"""
Storage Cost Estimation - Blocksize 10000
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# 绘图参数
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

# 颜色定义
color_1 = "#F27970"  # CVTree
color_2 = "#BB9727"  # BVTree  
color_3 = "#54B345"  # BKORE
color_4 = "#32B897"  # EncodeORE
color_5 = "#05B9E2"  # HybridORE

def estimate_cvtree_cost(n):
    """估算CVTree存储成本"""
    if n <= 1:
        return 0.1
    avg_depth = np.log(n) / np.log(3)
    node_count = n * avg_depth * 0.8
    return (node_count * 40 + n * 32) / 1024

def estimate_bvtree_cost(n, block_size):
    """估算BVTree存储成本"""
    if n <= 1:
        return 0.1
    num_blocks = int(np.ceil(n / block_size))
    cost_per_block = 32 * 3
    avg_items_per_block = min(n / num_blocks, block_size)
    if avg_items_per_block > 1:
        merkle_internal_nodes = max(0, avg_items_per_block - 1)
        cost_per_block += merkle_internal_nodes * 32
    return (num_blocks * cost_per_block) / 1024

def estimate_bkore_cost(n):
    """估算BKORE存储成本"""
    if n <= 1:
        return 0.1
    return (n * 56) / 1024

def estimate_encodeore_cost(n):
    """估算EncodeORE存储成本"""
    if n <= 1:
        return 0.1
    return (n * 20) / 1024

def estimate_hybridore_cost(n):
    """估算HybridORE存储成本"""
    if n <= 1:
        return 0.1
    return (n * 52) / 1024

# 数据规模和标签
data_sizes = [10**i for i in range(1, 7)]
x_labels = ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$']

# 计算blocksize = 10000的成本
cvtree_costs = [estimate_cvtree_cost(n) for n in data_sizes]
bvtree_costs = [estimate_bvtree_cost(n, 10000) for n in data_sizes]
bkore_costs = [estimate_bkore_cost(n) for n in data_sizes]
encodeore_costs = [estimate_encodeore_cost(n) for n in data_sizes]
hybridore_costs = [estimate_hybridore_cost(n) for n in data_sizes]

# 创建图表
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(range(len(data_sizes)), cvtree_costs, linewidth=2.0, color=color_1,
        marker='^', markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, label='CVTree')
ax.plot(range(len(data_sizes)), bvtree_costs, linewidth=2.0, color=color_2,
        marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, label='BVTree')
ax.plot(range(len(data_sizes)), bkore_costs, linewidth=2.0, color=color_3,
        marker='o', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8, label='BKORE')
ax.plot(range(len(data_sizes)), encodeore_costs, linewidth=2.0, color=color_4,
        marker='D', markerfacecolor=color_4, markeredgewidth=1.5, markersize=7, label='EncodeORE')
ax.plot(range(len(data_sizes)), hybridore_costs, linewidth=2.0, color=color_5,
        marker='*', markerfacecolor=color_5, markeredgewidth=1.5, markersize=9, label='HybridORE')

ax.set_xticks(range(len(data_sizes)))
ax.set_xticklabels(x_labels)
ax.set_xlabel('Data Size')
ax.set_ylabel('Storage Cost (KB)')
ax.set_title('Storage costs on the blockchain side (blocksize = $10^4$)')
ax.set_yscale('log')
ax.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
ax.legend(loc='upper left', ncol=2, columnspacing=0.8)

plt.tight_layout()
plt.savefig("./storage_costs_blockchainside_10000.pdf", format='pdf')
plt.show()

# 输出数据
print("Storage costs for blocksize = 10000:")
print(f"{'Data Size':<10} {'CVTree':<12} {'BVTree':<12} {'BKORE':<12} {'EncodeORE':<12} {'HybridORE':<12}")
print(f"{'':^10} {'(KB)':<12} {'(KB)':<12} {'(KB)':<12} {'(KB)':<12} {'(KB)':<12}")
print("-" * 78)
for i, size in enumerate(data_sizes):
    print(f"{x_labels[i]:<10} {cvtree_costs[i]:<12.2f} {bvtree_costs[i]:<12.2f} "
          f"{bkore_costs[i]:<12.2f} {encodeore_costs[i]:<12.2f} {hybridore_costs[i]:<12.2f}")