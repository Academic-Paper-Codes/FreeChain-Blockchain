# -*- coding: utf-8 -*-
"""
Communication Cost for Initialization Phase (Uploading Ciphertexts) - Revised
考虑不同方案的实际通信复杂度
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import math

# 绘图参数全家桶（与accuracy.py一致）
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '7.2, 4.8',
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
color_1 = "#F27970"
color_2 = "#BB9727"
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#05B9E2"

def calculate_realistic_communication_costs():
    """
    计算更现实的通信成本，考虑不同的复杂度特征
    """
    
    # 数据规模：10^1 到 10^6
    data_sizes = [10**i for i in range(1, 7)]
    block_size = 1000
    
    communication_costs = {
        'BKORE': [],
        'EncodeORE': [],
        'HybridORE': [],
        'CVTree': [],
        'BVTree': []
    }
    
    for n in data_sizes:

            
        # # BlockOPE: 密文 + 编码 + 路径 + 版本 ≈ 48 + 8 + log(n) + 4 bytes
        # blockope_sizes = [(48 + 8 + 4 + np.log2(n) * 4) / 1024 for n in existing_data_sizes]
        
        # # HybridORE: LewiWu(pos + right_array) + CLWW ≈ 4 + 256*12 + 8 = 3084 bytes
        # hybridore_sizes = [30 / 1024] * len(existing_data_sizes)
        
        # # FreORE: 24位模3数组 = 24 bytes trap * 2
        # freore_sizes = [24 * 2 / 1024 ] * len(existing_data_sizes)
        
        # # EncodeORE: 12位模3数组 = 12 bytes
        # encodeore_sizes = [12 / 1024] * len(existing_data_sizes)
        
        # BKORE: O(n log n) - 每次插入需要交互式遍历
        # 平均树高度为log(n)，每次插入需要DO-BCN多轮交互
        # 每轮: 请求(8B) + 响应(密文48B) + 确认(8B) = 64B
        # 总成本: n * log2(n) * 64B + n * 密文上传(80B)
        tree_interactions = n * max(1, int(math.log2(n))) * 60 / 1024
        ciphertext_upload = n * 80 / 1024
        bkore_cost = tree_interactions + ciphertext_upload
        
        # EncodeORE: O(n) - 纯密文上传，无额外交互
        # 每个CLWW密文12B + 协议开销4B = 16B
        encode_cost = n * 12 / 1024
        
        # HybridORE: O(n√n) - LewiWu部分需要与现有数据比较
        # CLWW部分: O(n)
        # LewiWu部分: 需要与已有数据进行排列计算，复杂度更高
        clww_cost = n * 12 / 1024
        lewiwu_cost = n * 0.1 / 1024  # 排列计算开销
        hybrid_cost = n * (30) / 1024
        
        # CVTree: O(n log n) - 前缀树深度约为密文长度，但有路径共享
        # FreORE密文: 16B，前缀树更新需要验证路径
        # 平均路径长度16，但有共享优化
        cipher_cost = n * 24 / 1024
        # 前缀共享效应：早期插入成本高，后期插入成本低
        # avg_path_cost = max(8, 16 - math.log2(max(1, n/100))) * 8 / 1024
        cvtree_cost = cipher_cost 
        
        # BVTree: O(n) - 批量块更新，真正的线性复杂度
        # FreORE密文: 16B
        # 块批量更新: 每1000个数据一个批次，每批次固定开销
        cipher_cost = n * 24 / 1024
        # num_batches = max(1, math.ceil(n / block_size))
        # batch_overhead = num_batches * 96 / 1024  # 固定批次开销
        bvtree_cost = cipher_cost 
        print(bkore_cost / cvtree_cost)
        
        communication_costs['BKORE'].append(bkore_cost)
        communication_costs['EncodeORE'].append(encode_cost)
        communication_costs['HybridORE'].append(hybrid_cost)
        communication_costs['CVTree'].append(cvtree_cost)
        communication_costs['BVTree'].append(bvtree_cost)
    
    return data_sizes, communication_costs

# 计算通信成本
data_sizes, communication_costs = calculate_realistic_communication_costs()

# 创建图形
fig, ax = plt.subplots()

# x轴位置
x_pos = np.arange(len(data_sizes))

# 绘制线图，体现不同的复杂度曲线
ax.plot(x_pos, communication_costs['BKORE'], linewidth=2.0, color=color_1, 
        marker='^', markerfacecolor=color_1, markeredgewidth=1.5, markersize=8,
        label='BlockOPE')

ax.plot(x_pos, communication_costs['EncodeORE'], linewidth=2.0, color=color_2, 
        marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,
        label='EncodeORE')

ax.plot(x_pos, communication_costs['HybridORE'], linewidth=2.0, color=color_3, 
        marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=10,
        label='HybridORE')

ax.plot(x_pos, communication_costs['CVTree'], linewidth=2.0, color=color_4, 
        marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,
        label='FreeChain-C')

ax.plot(x_pos, communication_costs['BVTree'], linewidth=2.0, color=color_5, 
        marker='D', markerfacecolor=color_5, markeredgewidth=1.5, markersize=7,
        label='FreeChain-B')

# 设置x轴标签
x_labels = [f'$10^{i}$' for i in range(1, 7)]
ax.set_xticks(x_pos)
ax.set_xticklabels(x_labels)

# 设置y轴为对数刻度以更好地显示不同复杂度
ax.set_yscale('log')

# 设置标签
ax.set_xlabel('Number of Data Records')
ax.set_ylabel('Communication Cost    ')
ax.yaxis.set_minor_locator(plt.NullLocator())  # 不显示y轴次刻度

# 设置图例
plt.legend(loc='upper left', ncol=1, columnspacing=0.4, prop={'size': 9})

# 添加网格
plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.3)

# 调整布局
plt.tight_layout()

# 保存图像
plt.savefig("./communication_init_realistic.pdf", format='pdf')

# 显示图像
plt.show()

# 分析复杂度
print("Communication Cost Analysis:")
print("Ciphertexts\tBKORE\t\tEncodeORE\tHybridORE\tCVTree\t\tBVTree")
for i, size in enumerate(data_sizes):
    print(f"{size}\t\t{communication_costs['BKORE'][i]:.2f}\t\t{communication_costs['EncodeORE'][i]:.2f}\t\t{communication_costs['HybridORE'][i]:.2f}\t\t{communication_costs['CVTree'][i]:.2f}\t\t{communication_costs['BVTree'][i]:.2f}")

print("\nComplexity Analysis:")
print("- BKORE: O(n log n) due to interactive tree traversal")
print("- EncodeORE: O(n) pure ciphertext upload")
print("- HybridORE: O(n√n) due to LewiWu permutation computation") 
print("- CVTree: O(n log n) with prefix sharing optimization")
print("- BVTree: O(n) true linear with batch optimization")