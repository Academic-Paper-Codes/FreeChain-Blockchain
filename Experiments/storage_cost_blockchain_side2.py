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
import os
from FreORE import FreORE
from cvtree import CVTree
from bvtree import BVTree
from BKORE import BlockOPE
from Crypto.Random import get_random_bytes

# 绘图参数全家桶
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '5.4, 2.3',
    'figure.dpi':'300',
    'figure.subplot.left':'0.12',
    'figure.subplot.right':'0.98',
    'figure.subplot.bottom':'0.15',
    'figure.subplot.top':'0.95',
    'pdf.fonttype':'42',
    'ps.fonttype':'42',
}
pylab.rcParams.update(params)

# 颜色定义
color_1 = "#F27970"  # CVTree
color_2 = "#BB9727"  # BVTree
color_3 = "#54B345"  # BKORE
color_4 = "#32B897"  # 预留
color_5 = "#05B9E2"  # 预留

def simulate_storage_costs(data_sizes, block_size):
    """
    模拟不同数据规模下的存储成本
    """
    # 初始化加密实例
    freore_key = os.urandom(16)
    freore = FreORE(d=2, alpha=1000, beta=10, gamma=9, pfk=f"{freore_key}".encode(), nx=8, ny=8)
    
    cvtree_costs = []
    bvtree_costs = []
    bkore_costs = []
    
    for size in data_sizes:
        print(f"Processing data size: {size}")
        
        # CVTree存储成本
        cvtree = CVTree(freore)
        for i in range(size):
            cvtree.insert(i, f"file_address_{i}")
        cvtree_cost = cvtree.get_storage_size()
        cvtree_costs.append(cvtree_cost)
        
        # BVTree存储成本
        bvtree = BVTree(freore, block_size=block_size)
        for i in range(size):
            bvtree.insert(i, f"file_address_{i}")
        bvtree_cost = bvtree.get_storage_size()
        bvtree_costs.append(bvtree_cost)
        
        # BKORE存储成本
        bkore = BlockOPE()
        key = get_random_bytes(16)
        for i in range(size):
            bkore.encrypt(i, key)
        bkore_cost = bkore.get_storage_size() / 1024  # 转换为KB
        bkore_costs.append(bkore_cost)
    
    return cvtree_costs, bvtree_costs, bkore_costs

# 数据规模
data_sizes = [10**i for i in range(1, 7)]  # 1e1 到 1e6
x_labels = ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$']

# 创建两个子图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 子图(c1): blocksize = 1e3
print("Simulating for blocksize = 1000...")
cvtree_costs_1k, bvtree_costs_1k, bkore_costs_1k = simulate_storage_costs(data_sizes, 1000)

ax1.plot(range(len(data_sizes)), cvtree_costs_1k, linewidth=2.0, color=color_1, 
         marker='^', markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, label='CVTree')
ax1.plot(range(len(data_sizes)), bvtree_costs_1k, linewidth=2.0, color=color_2, 
         marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, label='BVTree')
ax1.plot(range(len(data_sizes)), bkore_costs_1k, linewidth=2.0, color=color_3, 
         marker='o', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8, label='BlockOPE')

ax1.set_xticks(range(len(data_sizes)))
ax1.set_xticklabels(x_labels)
ax1.set_xlabel('Data Size')
ax1.set_ylabel('Storage Cost (KB)')
ax1.set_title('(c) Storage costs (blocksize = $10^3$)')
ax1.set_yscale('log')
ax1.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
ax1.legend(loc='upper left')

# 子图(c2): blocksize = 1e4
print("Simulating for blocksize = 10000...")
cvtree_costs_10k, bvtree_costs_10k, bkore_costs_10k = simulate_storage_costs(data_sizes, 10000)

ax2.plot(range(len(data_sizes)), cvtree_costs_10k, linewidth=2.0, color=color_1, 
         marker='^', markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, label='CVTree')
ax2.plot(range(len(data_sizes)), bvtree_costs_10k, linewidth=2.0, color=color_2, 
         marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, label='BVTree')
ax2.plot(range(len(data_sizes)), bkore_costs_10k, linewidth=2.0, color=color_3, 
         marker='o', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8, label='BlockOPE')

ax2.set_xticks(range(len(data_sizes)))
ax2.set_xticklabels(x_labels)
ax2.set_xlabel('Data Size')
ax2.set_ylabel('Storage Cost (KB)')
ax2.set_title('(c) Storage costs (blocksize = $10^4$)')
ax2.set_yscale('log')
ax2.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
ax2.legend(loc='upper left')

plt.tight_layout()
plt.savefig("./storage_costs.pdf", format='pdf')
plt.show()

# 打印数据以供参考
print("\nStorage costs for blocksize = 1000:")
print("Data sizes:", data_sizes)
print("CVTree costs:", cvtree_costs_1k)
print("BVTree costs:", bvtree_costs_1k)
print("BKORE costs:", bkore_costs_1k)

print("\nStorage costs for blocksize = 10000:")
print("Data sizes:", data_sizes)
print("CVTree costs:", cvtree_costs_10k)
print("BVTree costs:", bvtree_costs_10k)
print("BKORE costs:", bkore_costs_10k)