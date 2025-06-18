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
import sys
import os
from tqdm import tqdm
import math
import time

# 绘图参数全家桶
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

# 统一颜色方案
color_1 = "#F27970"  # CVTree
color_2 = "#BB9727"  # BVTree
color_3 = "#54B345"  # BlockOPE

def estimate_cvtree_proof_size(data_size, blockchain_size):
    """估算CVTree的证明大小"""
    # CVTree Merkle证明：树高度 * 哈希大小 + 地址信息
    tree_height = math.log2(max(1, data_size))
    hash_size_per_level = 32  # SHA-256哈希大小
    address_size = 32  # 地址字符串大小
    
    # 区块链大小影响：更大的区块链需要更多验证信息
    blockchain_factor = math.log10(blockchain_size) / 1000
    
    total_proof_size = (tree_height * hash_size_per_level + address_size) * (1 + blockchain_factor)
    return total_proof_size / 1024  # 转换为KB

def estimate_bvtree_proof_size(data_size, blockchain_size):
    """估算BVTree的证明大小"""
    block_size = 1000  # 默认块大小
    num_blocks = max(1, math.ceil(data_size / block_size))
    
    # 块内Merkle证明
    items_in_block = min(block_size, data_size)
    intra_block_height = math.log2(max(1, items_in_block))
    intra_block_proof = intra_block_height * 32
    
    # 块间完整性证明（相邻块的边界信息）
    inter_block_proof = math.log2(max(1, num_blocks)) * 64  # min/max边界
    
    # 块元数据
    block_metadata = 96  # block_id, merkle_root等
    
    # 区块链大小影响
    blockchain_factor = math.log10(blockchain_size) / 2000
    
    total_proof_size = (intra_block_proof + inter_block_proof + block_metadata) * (1 + blockchain_factor)
    return total_proof_size / 1024

def estimate_blockope_proof_size(data_size, blockchain_size):
    """估算BlockOPE的证明大小"""
    # OPE树的路径证明
    tree_height = math.log2(max(1, data_size))
    
    # 每层需要的信息：编码值(8B) + 路径方向(1B) + 验证信息(16B)
    path_proof_per_level = 8 + 1 + 16
    path_proof = tree_height * path_proof_per_level
    
    # UDZ处理开销（未决区重排证明）
    udz_overhead = min(128, data_size * 0.001)  # UDZ重排的额外证明
    
    # 版本和事务信息
    transaction_info = 64
    
    # 区块链大小对BlockOPE影响较大（事务冲突处理）
    blockchain_factor = math.log10(blockchain_size) / 500
    
    total_proof_size = (path_proof + udz_overhead + transaction_info) * (1 + blockchain_factor)
    return total_proof_size / 1024

def generate_proof_size_data(blockchain_size):
    """生成指定区块链大小下的证明大小数据"""
    # x轴：数据量 10^1 到 10^6
    data_sizes = [10**i for i in range(1, 7)]
    
    cvtree_sizes = []
    bvtree_sizes = []
    blockope_sizes = []
    
    print(f"Calculating proof sizes for blockchain size = 10^{int(math.log10(blockchain_size))}...")
    
    for size in tqdm(data_sizes, desc="Processing data sizes"):
        cvtree_size = estimate_cvtree_proof_size(size, blockchain_size)
        bvtree_size = estimate_bvtree_proof_size(size, blockchain_size)
        blockope_size = estimate_blockope_proof_size(size, blockchain_size)
        
        cvtree_sizes.append(cvtree_size)
        bvtree_sizes.append(bvtree_size)
        blockope_sizes.append(blockope_size)
    
    return data_sizes, cvtree_sizes, bvtree_sizes, blockope_sizes

def plot_proof_sizes_linear(blockchain_size, subplot_label):
    """绘制线性坐标的证明大小比较图"""
    data_sizes, cvtree_sizes, bvtree_sizes, blockope_sizes = generate_proof_size_data(blockchain_size)
    print(f"Data sizes: {data_sizes}")
    print(f"CVTree sizes: {cvtree_sizes}")
    print(f"BVTree sizes: {bvtree_sizes}")  
    print(f"BlockOPE sizes: {blockope_sizes}")
    
    # 创建图表
    fig, ax1 = plt.subplots()
    
    # x轴位置（线性刻度，但标签显示为指数形式）
    x_pos = np.arange(len(data_sizes))
    
    # 绘制线图
    ax1.plot(x_pos, cvtree_sizes, linewidth=2.0, color=color_1, marker='^', 
            markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, label='FreeChain-C')
    
    ax1.plot(x_pos, bvtree_sizes, linewidth=2.0, color=color_2, marker='s', 
            markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, label='FreeChain-B')
    
    ax1.plot(x_pos, blockope_sizes, linewidth=2.0, color=color_3, marker='o', 
            markerfacecolor=color_3, markeredgewidth=1.5, markersize=8, label='BlockOPE')
    
    # 设置x轴标签（指数形式）
    x_labels = [f'$10^{i}$' for i in range(1, 7)]
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    
    # 设置标签
    ax1.set_xlabel('Number of Data Records')
    ax1.set_ylabel('Proof Size (KB)')
    
    # 设置图例
    plt.legend(loc='upper left', ncol=3, columnspacing=0.4, prop={'size': 10})
    
    # 设置网格
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    
    # 设置y轴范围
    max_y = max(max(cvtree_sizes), max(bvtree_sizes), max(blockope_sizes))
    min_y = min(min(cvtree_sizes), min(bvtree_sizes), min(blockope_sizes))
    ax1.set_ylim(max(0, min_y * 0.9), max_y * 1.1)
    
    plt.tight_layout()
    
    # 保存图片
    blockchain_exp = int(math.log10(blockchain_size))
    filename = f"./proof_sizes_{subplot_label}_blockchain_{blockchain_exp}.pdf"
    plt.savefig(filename, format='pdf')
    plt.show()
    
    # 打印数据用于验证
    print(f"\nProof sizes for blockchain size = 10^{blockchain_exp}:")
    print("Data Size\tCVTree(KB)\tBVTree(KB)\tBlockOPE(KB)")
    for i, size in enumerate(data_sizes):
        print(f"10^{i+1}\t\t{cvtree_sizes[i]:.4f}\t\t{bvtree_sizes[i]:.4f}\t\t{blockope_sizes[i]:.4f}")

# ================ 新增：吞吐量分析代码 ================

def estimate_cvtree_insert_time(data_size):
    """估算CVTree的插入时间（数据加密 + 索引构建）"""
    # 数据加密时间（FreORE加密复杂度较高）
    data_enc_time = 0.5  # 毫秒，FreORE加密时间
    
    # 索引构建时间：遍历前缀树 + 哈希计算
    tree_height = math.log2(max(1, data_size))
    
    # 前缀树遍历时间
    tree_traversal_time = tree_height * 0.01  # 每层0.01毫秒
    
    # 哈希计算时间（沿路径更新所有祖先节点）
    hash_computation_time = tree_height * 0.05  # 每层0.05毫秒
    
    building_index_time = tree_traversal_time + hash_computation_time
    
    total_time = data_enc_time + building_index_time
    return total_time  # 毫秒

def estimate_bvtree_insert_time(data_size):
    """估算BVTree的插入时间"""
    # 数据加密时间
    data_enc_time = 0.5  # FreORE加密时间
    
    # 索引构建时间
    block_size = 1000
    current_block_size = (data_size - 1) % block_size + 1
    
    # 块内插入时间
    block_insert_time = 0.01  # 块内简单插入
    
    # Merkle根重计算时间
    merkle_computation_time = math.log2(max(1, current_block_size)) * 0.03
    
    # 如果需要创建新块的额外开销
    new_block_overhead = 0.1 if current_block_size == 1 else 0
    
    building_index_time = block_insert_time + merkle_computation_time + new_block_overhead
    
    total_time = data_enc_time + building_index_time
    return total_time

def estimate_blockope_insert_time(data_size):
    """估算BlockOPE的插入时间"""
    # BlockOPE加密时间（包含交互式遍历）
    tree_height = math.log2(max(1, data_size))
    interactive_traversal_time = tree_height * 0.1  # 交互式遍历较慢
    
    data_enc_time = 0.2 + interactive_traversal_time  # AES加密 + 交互遍历
    
    # 索引构建时间
    # 树插入时间
    tree_insert_time = 0.05
    
    # 版本检查和事务处理时间
    transaction_overhead = 0.1
    
    # UDZ处理时间（随数据量增加而增加的冲突概率）
    udz_processing_time = min(2.0, data_size * 0.0001)  # UDZ重排时间
    
    building_index_time = tree_insert_time + transaction_overhead + udz_processing_time
    
    total_time = data_enc_time + building_index_time
    return total_time

def generate_throughput_data():
    """生成吞吐量数据"""
    # x轴：数据量 10^1 到 10^6
    data_sizes = [10**i for i in range(1, 7)]
    
    cvtree_throughputs = []
    bvtree_throughputs = []
    blockope_throughputs = []
    
    print("Calculating insert throughputs...")
    
    for size in tqdm(data_sizes, desc="Processing data sizes for throughput"):
        # 计算每种方案的插入时间（毫秒）
        cvtree_time = estimate_cvtree_insert_time(size)
        bvtree_time = estimate_bvtree_insert_time(size)
        blockope_time = estimate_blockope_insert_time(size)
        
        # 计算吞吐量（事务/秒）= 1000 / 时间(毫秒)
        cvtree_throughput = 1000 / cvtree_time
        bvtree_throughput = 1000 / bvtree_time
        blockope_throughput = 1000 / blockope_time
        
        cvtree_throughputs.append(cvtree_throughput)
        bvtree_throughputs.append(bvtree_throughput)
        blockope_throughputs.append(blockope_throughput)
    
    return data_sizes, cvtree_throughputs, bvtree_throughputs, blockope_throughputs

def plot_insert_throughput():
    """绘制插入吞吐量比较图"""
    data_sizes, cvtree_throughputs, bvtree_throughputs, blockope_throughputs = generate_throughput_data()
    
    # 创建图表
    fig, ax1 = plt.subplots()
    
    # x轴位置（线性刻度，但标签显示为指数形式）
    x_pos = np.arange(len(data_sizes))
    
    # 绘制线图
    ax1.plot(x_pos, cvtree_throughputs, linewidth=2.0, color=color_1, marker='^', 
            markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, label='CVTree')
    
    ax1.plot(x_pos, bvtree_throughputs, linewidth=2.0, color=color_2, marker='s', 
            markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, label='BVTree')
    
    ax1.plot(x_pos, blockope_throughputs, linewidth=2.0, color=color_3, marker='o', 
            markerfacecolor=color_3, markeredgewidth=1.5, markersize=8, label='BlockOPE')
    
    # 设置x轴标签（指数形式）
    x_labels = [f'$10^{i}$' for i in range(1, 7)]
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels)
    
    # 设置标签
    ax1.set_xlabel('Number of Data Records')
    ax1.set_ylabel('Throughput (tx/s)')
    
    # 设置图例
    plt.legend(loc='upper right', ncol=3, columnspacing=0.4, prop={'size': 10})
    
    # 设置网格
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    
    # 设置y轴范围
    max_y = max(max(cvtree_throughputs), max(bvtree_throughputs), max(blockope_throughputs))
    min_y = min(min(cvtree_throughputs), min(bvtree_throughputs), min(blockope_throughputs))
    ax1.set_ylim(max(0, min_y * 0.8), max_y * 1.1)
    
    plt.tight_layout()
    
    # 保存图片
    filename = "./insert_throughput_comparison.pdf"
    plt.savefig(filename, format='pdf')
    plt.show()
    
    # 打印数据用于验证
    print(f"\nInsert Throughput Comparison:")
    print("Data Size\tCVTree(tx/s)\tBVTree(tx/s)\tBlockOPE(tx/s)")
    for i, size in enumerate(data_sizes):
        print(f"10^{i+1}\t\t{cvtree_throughputs[i]:.2f}\t\t{bvtree_throughputs[i]:.2f}\t\t{blockope_throughputs[i]:.2f}")

def main():
    """主函数"""
    print("Generating proof size comparison plots...")
    
    # 生成图表(e): blockchain size = 10^3
    print("\n" + "="*50)
    print("Generating plot (e)...")
    plot_proof_sizes_linear(1000, 'e')
    
    # 生成图表(f): blockchain size = 10^4
    print("\n" + "="*50)
    print("Generating plot (f)...")
    plot_proof_sizes_linear(10000, 'f')
    
    # 生成吞吐量比较图
    print("\n" + "="*50)
    print("Generating insert throughput comparison plot...")
    plot_insert_throughput()
    
    print("\n" + "="*50)
    print("All plots generated successfully!")
    print("Files saved:")
    print("- proof_sizes_e_blockchain_3.pdf")
    print("- proof_sizes_f_blockchain_4.pdf")
    print("- insert_throughput_comparison.pdf")

# 运行实验
if __name__ == "__main__":
    main()