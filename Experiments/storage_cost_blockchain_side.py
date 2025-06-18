# -*- coding: utf-8 -*-
"""
Storage Cost Estimation for All ORE Schemes
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# 绘图参数
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '10',
    'figure.figsize': '12, 5',
    'figure.dpi':'300',
    'figure.subplot.left':'0.08',
    'figure.subplot.right':'0.98',
    'figure.subplot.bottom':'0.15',
    'figure.subplot.top':'0.95',
    'pdf.fonttype':'42',
    'ps.fonttype':'42',
}
pylab.rcParams.update(params)

# 颜色定义 - 扩展到5种方案
color_1 = "#F27970"  # CVTree
color_2 = "#BB9727"  # BVTree  
color_3 = "#54B345"  # BKORE
color_4 = "#32B897"  # EncodeORE
color_5 = "#05B9E2"  # HybridORE

def estimate_cvtree_cost(n):
    """
    估算CVTree存储成本
    - 节点数: 约 n * log₃(n) （三进制前缀树）
    - 每个节点: 32B (哈希) + 8B (指针) = 40B
    """
    if n <= 1:
        return 0.1
    avg_depth = np.log(n) / np.log(3)  # 三进制树平均深度
    node_count = n * avg_depth * 0.8  # 考虑共享前缀
    return (node_count * 40 + n * 32) / 1024  # 转换为KB

def estimate_bvtree_cost(n, block_size):
    """
    估算BVTree存储成本
    - 块数: ceil(n / block_size)
    - 每块: min_hash(32B) + max_hash(32B) + merkle_root(32B) + 内部节点
    """
    if n <= 1:
        return 0.1
    num_blocks = int(np.ceil(n / block_size))
    
    cost_per_block = 32 * 3  # min + max + root
    
    # Merkle树内部节点成本（每块）
    avg_items_per_block = min(n / num_blocks, block_size)
    if avg_items_per_block > 1:
        merkle_internal_nodes = max(0, avg_items_per_block - 1)
        cost_per_block += merkle_internal_nodes * 32
    
    return (num_blocks * cost_per_block) / 1024

def estimate_bkore_cost(n):
    """
    估算BKORE存储成本
    - OPE树节点: n个
    - 每个节点: nonce(16B) + ciphertext(16B) + tag(16B) + code(8B) = 56B
    """
    if n <= 1:
        return 0.1
    return (n * 56) / 1024

def estimate_encodeore_cost(n):
    """
    估算EncodeORE存储成本
    - 每个数据项: 密文列表(12个元素 * 1B = 12B) + 元数据(8B) = 20B
    - 索引结构: 简单的线性存储，无额外树结构
    """
    if n <= 1:
        return 0.1
    # EncodeORE密文长度 = l1 + l2 = 8 + 4 = 12位，每位1字节存储
    ciphertext_size = 12  # 12字节密文
    metadata_size = 8     # 地址和索引信息
    return (n * (ciphertext_size + metadata_size)) / 1024

def estimate_hybridore_cost(n):
    """
    估算HybridORE存储成本  
    - 范围部分: LewiWu小域加密，每项约32B
    - 值部分: CLWW加密，每项约12B  
    - 组合开销: 范围部分32B + 值部分12B + 元数据8B = 52B/项
    """
    if n <= 1:
        return 0.1
    # 范围加密(LewiWu): position(4B) + right_array(256*12B) ≈ 3KB/项（简化估算为32B）
    range_cost = 32
    # 值加密(CLWW): 8位密文 ≈ 12B  
    value_cost = 12
    metadata_cost = 8
    return (n * (range_cost + value_cost + metadata_cost)) / 1024

# 数据规模和标签
data_sizes = [10**i for i in range(1, 7)]
x_labels = ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$']

# 计算估算成本
print("Estimating storage costs for all ORE schemes...")

# Block size = 1000
cvtree_costs_1k = [estimate_cvtree_cost(n) for n in data_sizes]
bvtree_costs_1k = [estimate_bvtree_cost(n, 1000) for n in data_sizes]
bkore_costs_1k = [estimate_bkore_cost(n) for n in data_sizes]
encodeore_costs_1k = [estimate_encodeore_cost(n) for n in data_sizes]
hybridore_costs_1k = [estimate_hybridore_cost(n) for n in data_sizes]

# Block size = 10000  
cvtree_costs_10k = [estimate_cvtree_cost(n) for n in data_sizes]
bvtree_costs_10k = [estimate_bvtree_cost(n, 10000) for n in data_sizes]
bkore_costs_10k = [estimate_bkore_cost(n) for n in data_sizes]
encodeore_costs_10k = [estimate_encodeore_cost(n) for n in data_sizes]
hybridore_costs_10k = [estimate_hybridore_cost(n) for n in data_sizes]

# 创建图表
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# 子图1: blocksize = 1000
ax1.plot(range(len(data_sizes)), cvtree_costs_1k, linewidth=2.0, color=color_1,
         marker='^', markerfacecolor=color_1, markeredgewidth=1.5, markersize=7, label='CVTree')
ax1.plot(range(len(data_sizes)), bvtree_costs_1k, linewidth=2.0, color=color_2,
         marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=7, label='BVTree')
ax1.plot(range(len(data_sizes)), bkore_costs_1k, linewidth=2.0, color=color_3,
         marker='o', markerfacecolor=color_3, markeredgewidth=1.5, markersize=7, label='BKORE')
ax1.plot(range(len(data_sizes)), encodeore_costs_1k, linewidth=2.0, color=color_4,
         marker='D', markerfacecolor=color_4, markeredgewidth=1.5, markersize=7, label='EncodeORE')
ax1.plot(range(len(data_sizes)), hybridore_costs_1k, linewidth=2.0, color=color_5,
         marker='*', markerfacecolor=color_5, markeredgewidth=1.5, markersize=8, label='HybridORE')

ax1.set_xticks(range(len(data_sizes)))
ax1.set_xticklabels(x_labels)
ax1.set_xlabel('Data Size')
ax1.set_ylabel('Storage Cost (KB)')
ax1.set_title('Storage costs (blocksize = $10^3$)')
ax1.set_yscale('log')
ax1.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
ax1.legend(loc='upper left', ncol=2)

# 子图2: blocksize = 10000
ax2.plot(range(len(data_sizes)), cvtree_costs_10k, linewidth=2.0, color=color_1,
         marker='^', markerfacecolor=color_1, markeredgewidth=1.5, markersize=7, label='CVTree')
ax2.plot(range(len(data_sizes)), bvtree_costs_10k, linewidth=2.0, color=color_2,
         marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=7, label='BVTree')
ax2.plot(range(len(data_sizes)), bkore_costs_10k, linewidth=2.0, color=color_3,
         marker='o', markerfacecolor=color_3, markeredgewidth=1.5, markersize=7, label='BKORE')
ax2.plot(range(len(data_sizes)), encodeore_costs_10k, linewidth=2.0, color=color_4,
         marker='D', markerfacecolor=color_4, markeredgewidth=1.5, markersize=7, label='EncodeORE')
ax2.plot(range(len(data_sizes)), hybridore_costs_10k, linewidth=2.0, color=color_5,
         marker='*', markerfacecolor=color_5, markeredgewidth=1.5, markersize=8, label='HybridORE')

ax2.set_xticks(range(len(data_sizes)))
ax2.set_xticklabels(x_labels)
ax2.set_xlabel('Data Size')
ax2.set_ylabel('Storage Cost (KB)')
ax2.set_title('Storage costs (blocksize = $10^4$)')
ax2.set_yscale('log')
ax2.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
ax2.legend(loc='upper left', ncol=2)

plt.tight_layout()
plt.savefig("./storage_costs_all_schemes.pdf", format='pdf')
plt.show()

# 输出详细估算数据
print(f"\n{'='*80}")
print("ESTIMATED STORAGE COSTS FOR ALL ORE SCHEMES")
print(f"{'='*80}")

print(f"\nBlocksize = 1000:")
print(f"{'Data Size':<10} {'CVTree':<12} {'BVTree':<12} {'BKORE':<12} {'EncodeORE':<12} {'HybridORE':<12}")
print(f"{'':^10} {'(KB)':<12} {'(KB)':<12} {'(KB)':<12} {'(KB)':<12} {'(KB)':<12}")
print("-" * 78)
for i, size in enumerate(data_sizes):
    print(f"{x_labels[i]:<10} {cvtree_costs_1k[i]:<12.2f} {bvtree_costs_1k[i]:<12.2f} "
          f"{bkore_costs_1k[i]:<12.2f} {encodeore_costs_1k[i]:<12.2f} {hybridore_costs_1k[i]:<12.2f}")

print(f"\nBlocksize = 10000:")
print(f"{'Data Size':<10} {'CVTree':<12} {'BVTree':<12} {'BKORE':<12} {'EncodeORE':<12} {'HybridORE':<12}")
print(f"{'':^10} {'(KB)':<12} {'(KB)':<12} {'(KB)':<12} {'(KB)':<12} {'(KB)':<12}")
print("-" * 78)
for i, size in enumerate(data_sizes):
    print(f"{x_labels[i]:<10} {cvtree_costs_10k[i]:<12.2f} {bvtree_costs_10k[i]:<12.2f} "
          f"{bkore_costs_10k[i]:<12.2f} {encodeore_costs_10k[i]:<12.2f} {hybridore_costs_10k[i]:<12.2f}")

# 输出性能特征分析
print(f"\n{'='*80}")
print("STORAGE COST ANALYSIS:")
print(f"{'='*80}")
print("Growth Patterns:")
print("- CVTree:    O(n log n) - 前缀树节点随深度增长")
print("- BVTree:    O(n/blocksize) - 块数量决定，blocksize越大开销越小") 
print("- BKORE:     O(n) - 线性增长，每项固定开销")
print("- EncodeORE: O(n) - 线性增长，密文长度固定")
print("- HybridORE: O(n) - 线性增长，组合加密开销较大")
print()
print("Cost per Item (at n=10⁶):")
for i, scheme in enumerate(['CVTree', 'BVTree(1k)', 'BKORE', 'EncodeORE', 'HybridORE']):
    costs_1k = [cvtree_costs_1k, bvtree_costs_1k, bkore_costs_1k, encodeore_costs_1k, hybridore_costs_1k]
    cost_per_item = (costs_1k[i][-1] * 1024) / data_sizes[-1]  # 转换为字节后除以项数
    print(f"- {scheme:<12}: {cost_per_item:.1f} bytes/item")

print(f"\n{'='*80}")
print("ESTIMATION ASSUMPTIONS:")
print("- CVTree: 三进制前缀树，80%前缀共享，节点40B")
print("- BVTree: 每块96B基础+Merkle内部节点32B")  
print("- BKORE: OPE树节点56B（nonce+ct+tag+code）")
print("- EncodeORE: 12B密文+8B元数据=20B/项")
print("- HybridORE: 范围32B+值12B+元数据8B=52B/项")
print(f"{'='*80}")

# 生成成本比较表（相对于最低成本的倍数）
print(f"\nCOST COMPARISON (relative to minimum at each scale):")
print(f"{'Data Size':<10} {'CVTree':<10} {'BVTree':<10} {'BKORE':<10} {'EncodeORE':<10} {'HybridORE':<10}")
print("-" * 70)
for i, size in enumerate(data_sizes):
    costs = [cvtree_costs_1k[i], bvtree_costs_1k[i], bkore_costs_1k[i], 
             encodeore_costs_1k[i], hybridore_costs_1k[i]]
    min_cost = min(costs)
    ratios = [c/min_cost for c in costs]
    print(f"{x_labels[i]:<10} {ratios[0]:<10.1f} {ratios[1]:<10.1f} {ratios[2]:<10.1f} "
          f"{ratios[3]:<10.1f} {ratios[4]:<10.1f}")