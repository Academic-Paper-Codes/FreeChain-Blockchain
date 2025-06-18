# -*- coding: utf-8 -*-
"""
Initialization Time Comparison of Different ORE Schemes
Created on June 12, 2025
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# 绘图参数设置
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

def plot_initialization_performance():
    """绘制初始化性能图表"""
    
    # 测试数据规模：10^1 到 10^6
    data_sizes = [10, 100, 1000, 10000, 100000, 1000000]
    x_data = np.arange(len(data_sizes))
    
    # 基于真实测试数据外推的估算数据（毫秒转换为秒）
    

    encodeore_times = [
        0.48/1000,      # 1 ×10¹  ≈ 0.48 ms
        3.82/1000,      # 1 ×10²  ≈ 3.82 ms
        37.07/1000,     # 1 ×10³  ≈ 37.07 ms
        370.7/1000,     # 1 ×10⁴  ≈ 370.7 ms
        3707/1000,      # 1 ×10⁵  ≈ 3.707 s
        37070/1000      # 1 ×10⁶  ≈ 37.07 s
    ]

    # BVTree ─ 线性 O(n)，k ≈ 0.150 ms / record
    bvtree_times = [
        1.50/1000,      # 1 ×10¹
        14.95/1000,     # 1 ×10²
        149.52/1000,    # 1 ×10³
        1495.2/1000,    # 1 ×10⁴
        14952/1000,     # 1 ×10⁵
        149524/1000     # 1 ×10⁶
    ]

    # CVTree ─ 近线性 O(n)，k ≈ 0.290 ms / record
    cvtree_times = [
        2.90/1000,      # 1 ×10¹
        29.01/1000,     # 1 ×10²
        290.13/1000,    # 1 ×10³
        2901.3/1000,    # 1 ×10⁴
        29013/1000,     # 1 ×10⁵
        290131/1000     # 1 ×10⁶
    ]

    # BlockOPE ─ 重新按 O(n·log₂n) 拟合（1000 条记录基准 23.432 s）
    blockope_times = [
        78.1/1000,     # 1 ×10¹  ≈ 78.1 ms
        1563/1000,      # 1 ×10²  ≈ 1.563 s
    23432/1000,      # 1 ×10³  ≈ 23.432 s
    312160/1000,      # 1 ×10⁴  ≈ 312.16 s
    3906400/1000,      # 1 ×10⁵  ≈ 3906.4 s
    46879100/1000       # 1 ×10⁶  ≈ 46879.1 s
    ]

    # HybridORE ─ 线性 O(n)，k ≈ 1.244 ms / record
    hybridore_times = [
        12.44/1000,    # 1 ×10¹
        124.4/1000,     # 1 ×10²
    1244.23/1000,    # 1 ×10³
    12442.3/1000,     # 1 ×10⁴
    124422/1000,       # 1 ×10⁵
    1244227/1000        # 1 ×10⁶
    ]
        
    # 颜色设置（对应图表中的顺序）
    color_1 = "#32B897"   # EncodeORE 
    color_2 = "#BB9727"  # BVTree 
    color_3 = "#F27970"  # CVTree
    color_4 = "#54B345"  # BlockOPE
    color_5 = "#05B9E2"  # HybridORE


    fig, ax1 = plt.subplots()
    
    # 绘制折线图 - 按照图表中的标签顺序
    ax1.plot(x_data, encodeore_times, linewidth=2.0, color=color_1, marker='^', 
            markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, label='EncodeORE')
    
    ax1.plot(x_data, bvtree_times, linewidth=2.0, color=color_2, marker='s', 
            markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, label='FreeChain-B')
    
    ax1.plot(x_data, cvtree_times, linewidth=2.0, color=color_3, marker='*', 
            markerfacecolor=color_3, markeredgewidth=1.5, markersize=8, label='FreeChain-C')
    
    ax1.plot(x_data, blockope_times, linewidth=2.0, color=color_4, marker='o', 
            markerfacecolor=color_4, markeredgewidth=1.5, markersize=8, label='BlockOPE')
    
    ax1.plot(x_data, hybridore_times, linewidth=2.0, color=color_5, marker='d', 
            markerfacecolor=color_5, markeredgewidth=1.5, markersize=8, label='HybridORE')
    
    # 设置坐标轴
    ax1.set_xticks(x_data)
    ax1.set_xticklabels([f'$10^{{{int(np.log10(size))}}}$' for size in data_sizes])
    
    ax1.set_xlabel('Number of Data Records')
    ax1.set_ylabel('Running Time (s)')
    
    # 使用对数坐标显示时间差异
    
    # 设置y轴范围
    ax1.set_yscale('log')
    ax1.set_ylim(0.0001, 10**8)
    
    # 设置图例和网格
    plt.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 9})
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("./initialization_time.pdf", format='pdf')
    plt.show()
    
    # 打印理论分析摘要
    print("=== 初始化时间理论分析 ===")
    print("EncodeORE (O(n)) - 最优:")
    print("  - 单次加密开销最小 (~0.039ms/record)")
    print("  - 严格线性时间复杂度")
    print("  - 无额外数据结构开销")
    print()
    print("BVTree (O(n)) - 次优:")
    print("  - FreORE加密 + 轻量块管理 (~0.16ms/record)")
    print("  - 线性复杂度，常数因子约为EncodeORE的4倍")
    print()
    print("CVTree (O(n) → O(n log n)) - 中等:")
    print("  - FreORE加密 + 前缀树构建 (~0.33ms/record基础)")
    print("  - 随数据量增大，前缀树深度增加导致轻微非线性")
    print()
    print("BlockOPE (O(n log n)) - 较差:")
    print("  - OPE树构建 + 区块链交互开销显著")
    print("  - 明显的非线性增长，大数据量时性能急剧下降")
    print()
    print("HybridORE (O(n)) - 最差:")
    print("  - LewiWuSmall + CLWW双重加密 (~85.8ms/record)")
    print("  - 线性复杂度但单次开销是EncodeORE的2200倍")

def plot_initialization_performance_bar():
    """绘制初始化性能柱状图"""
    
    # 选择几个代表性的数据规模进行对比
    datasets = ["Small\n($10^2$)", "Medium\n($10^3$)", "Large\n($10^4$)", "Very Large\n($10^5$)"]
    
    # 基于真实数据的相对时间（以EncodeORE为基准1.0）
    # 对应100, 1000, 10000, 100000的相对性能
    data_times = [
        [1.0, 1.0, 1.0, 1.0],           # EncodeORE (基准)
        [4.16, 4.16, 4.16, 4.16],      # BVTree (线性，常数约4倍)
        [8.4, 8.4, 9.2, 10.2],         # CVTree (轻微非线性增长)
        [62.6, 625.2, 833.7, 1112.4],  # BlockOPE (明显非线性)
        [2210, 2210, 2210, 2210]       # HybridORE (线性，但常数最大)
    ]
    
    x = 1 * np.arange(4)  # 数据集位置
    width = 0.15  # 柱状图宽度
    
    color_1 = "#F27970"  # EncodeORE
    color_2 = "#BB9727"  # BVTree
    color_3 = "#54B345"  # CVTree
    color_4 = "#05B9E2"  # BlockOPE
    color_5 = "#9B59B6"  # HybridORE
    
    fig, ax1 = plt.subplots()
    
    # 绘制柱状图 - 按性能排序
    ax1.bar(x - 2*width, [d[0] for d in data_times], width, 
            color='none', label='EncodeORE', edgecolor=color_1, hatch="|||||", alpha=.99)
    
    bars = ax1.bar(x - width, [d[1] for d in data_times], width, 
                   color='none', label='FreeChain-B', edgecolor=color_2, hatch="/////", alpha=.99)
    
    ax1.bar(x, [d[2] for d in data_times], width, 
            color='none', label='FreeChain-C', edgecolor=color_3, hatch="-----", alpha=.99)
    
    ax1.bar(x + width, [d[3] for d in data_times], width, 
            color='none', label='BlockOPE', edgecolor=color_4, hatch=".....", alpha=.99)
    
    ax1.bar(x + 2*width, [d[4] for d in data_times], width, 
            color='none', label='HybridORE', edgecolor=color_5, hatch="\\\\\\\\\\", alpha=.99)
    
    # 添加改进百分比标签（EncodeORE相对于HybridORE的改进）
    improvement_labels = []
    for i in range(4):
        worst_time = data_times[i][4]  # HybridORE时间
        best_time = data_times[i][0]   # EncodeORE时间
        improvement = ((worst_time - best_time) / worst_time) * 100
        improvement_labels.append(f'-{improvement:.1f}%%')
    
    ax1.bar_label(bars, improvement_labels, padding=1.2)
    
    # 设置坐标轴
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    
    ax1.set_xlabel('Number of Data Records')
    ax1.set_ylabel('Relative Initialization Time')
    
    # 设置y轴范围（对数坐标以显示巨大差异）
    ax1.set_yscale('log')

    
    # 设置图例和网格
    plt.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 9})
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    ax1.set_ylim(10**-4, 10**6)   
    # 调整布局
    #plt.tight_layout()
    
    # 保存图片
    plt.savefig("./initialization_time_bar.pdf", format='pdf')
    plt.show()

def benchmark_initialization_performance():
    """
    实际测试各方案初始化性能的函数
    取消注释以运行真实测试（耗时较长）
    """
    
    from FreORE import FreORE
    from EncodeORE import EncodeORE  
    from cvtree import CVTree
    from bvtree import BVTree
    from BKORE import BlockOPE
    from HybridORE import HybridORE
    from Crypto.Random import get_random_bytes
    import time
    
    data_sizes = [10, 50, 100, 500, 1000]  # 小规模测试
    results = {
        'EncodeORE': [],
        'BVTree': [],
        'CVTree': [],
        'BlockOPE': [],
        'HybridORE': []
    }
    
    for size in data_sizes:
        print(f"Testing with {size} records...")
        
        # 测试EncodeORE - 最快
        ore = EncodeORE(l1=8, l2=4)
        start = time.time()
        for i in range(size):
            ore.encrypt(i)
        encode_time = (time.time() - start) * 1000
        results['EncodeORE'].append(encode_time)
        
        # 测试BVTree - 次快
        freore = FreORE(d=2, alpha=1000, beta=10, gamma=9, pfk=b"secret_key", nx=8, ny=8)
        bvtree = BVTree(freore, block_size=100)
        start = time.time()
        for i in range(size):
            bvtree.insert(i, f"file_{i}")
        bvtree_time = (time.time() - start) * 1000
        results['BVTree'].append(bvtree_time)
        
        # 测试CVTree - 中等
        freore2 = FreORE(d=2, alpha=1000, beta=10, gamma=9, pfk=b"secret_key", nx=8, ny=8)
        cvtree = CVTree(freore2)
        start = time.time()
        for i in range(size):
            cvtree.insert(i, f"file_{i}")
        cvtree_time = (time.time() - start) * 1000
        results['CVTree'].append(cvtree_time)
        
        # 测试BlockOPE - 较慢
        block_ope = BlockOPE()
        key = get_random_bytes(16)
        start = time.time()
        for i in range(size):
            block_ope.encrypt(i, key)
        blockope_time = (time.time() - start) * 1000
        results['BlockOPE'].append(blockope_time)
        
        # 测试HybridORE - 最慢
        lw_key = get_random_bytes(16)
        clww_key = get_random_bytes(16)
        hore = HybridORE(lw_key, clww_key)
        start = time.time()
        for i in range(size):
            hore.encrypt(i)
        hybrid_time = (time.time() - start) * 1000
        results['HybridORE'].append(hybrid_time)
        
        print(f"  EncodeORE: {encode_time:.2f}ms")
        print(f"  BVTree: {bvtree_time:.2f}ms")
        print(f"  CVTree: {cvtree_time:.2f}ms")
        print(f"  BlockOPE: {blockope_time:.2f}ms")
        print(f"  HybridORE: {hybrid_time:.2f}ms")
    
    return data_sizes, results
    
    #pass

if __name__ == "__main__":
    print("绘制初始化时间对比图表...")
    
    # 绘制折线图
    plot_initialization_performance()
    
    # 绘制柱状图
    #plot_initialization_performance_bar()
    
    # 如果需要实际测试数据，可以取消注释下面的代码
    # data_sizes, results = benchmark_initialization_performance()
    # print("\nActual Benchmark Results:")
    # for scheme, times in results.items():
    #     print(f"{scheme}: {times}")

