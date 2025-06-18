# -*- coding: utf-8 -*-
"""
Theoretical Performance Analysis for ORE Schemes
Based on accuracy.py template
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# 绘图参数全家桶（从accuracy.py复制）
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '7.2, 4.8',
    'figure.dpi':'300',
    'figure.subplot.left':'0.12',
    'figure.subplot.right':'0.95',
    'figure.subplot.bottom':'0.15',
    'figure.subplot.top':'0.92',
    'pdf.fonttype':'42',
    'ps.fonttype':'42',
}
pylab.rcParams.update(params)

# 颜色定义（从accuracy.py复制）
color_1 = "#F27970"
color_2 = "#BB9727" 
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#05B9E2"

def theoretical_analysis():
    """理论分析各ORE方案的加密时间复杂度"""
    
    # 数据规模：从10^1到10^6
    exponents = np.arange(1, 7)  # [1, 2, 3, 4, 5, 6]
    data_sizes = [10**i for i in exponents]
    x_labels = [f"$10^{i}$" for i in exponents]
    
    # 理论时间估算（微秒为单位）
    
    # 1. BlockOPE: O(log n) - 树遍历时间随数据量对数增长
    # 假设基准时间为10微秒，每层遍历增加5微秒
    blockope_times = [10 + 5 * np.log2(n) for n in data_sizes]
    
    # 2. HybridORE: O(1) - 固定时间，包含LewiWu小域加密+CLWW加密
    # LewiWu加密约50微秒 + CLWW加密约30微秒
    hybridore_times = [80] * len(data_sizes)  # 常数时间
    
    # 3. FreORE: O(1) - 固定时间，主要是HMAC计算和模运算
    # 数据分割5微秒 + 编码10微秒 + HMAC循环计算40微秒
    freore_times = [55] * len(data_sizes)  # 常数时间
    
    # 4. EncodeORE: O(1) - 固定时间，类似FreORE但稍快
    # 数据分割5微秒 + 编码8微秒 + HMAC循环计算35微秒
    encodeore_times = [48] * len(data_sizes)  # 常数时间
    
    return data_sizes, x_labels, blockope_times, hybridore_times, freore_times, encodeore_times

def plot_performance():
    """绘制性能对比图"""
    
    data_sizes, x_labels, blockope_times, hybridore_times, freore_times, encodeore_times = theoretical_analysis()
    
    fig, ax = plt.subplots()
    
    x = np.arange(len(data_sizes))
    
    # 绘制线图
    ax.plot(x, blockope_times, linewidth=2.0, color=color_1, marker='^', 
           markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, 
           label='BlockOPE O(log n)')
    
    ax.plot(x, hybridore_times, linewidth=2.0, color=color_2, marker='s', 
           markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, 
           label='HybridORE O(1)')
    
    ax.plot(x, freore_times, linewidth=2.0, color=color_3, marker='*', 
           markerfacecolor=color_3, markeredgewidth=1.5, markersize=10, 
           label='FreORE O(1)')
    
    ax.plot(x, encodeore_times, linewidth=2.0, color=color_4, marker='o', 
           markerfacecolor=color_4, markeredgewidth=1.5, markersize=8, 
           label='EncodeORE O(1)')
    
    # 设置坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Single Encryption Time (μs)')
    
    # 设置Y轴范围，突出差异
    ax.set_ylim(40, max(blockope_times) * 1.1)
    
    # 网格和图例
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.legend(loc='upper left', ncol=2, columnspacing=0.8, prop={'size': 12})
    
    plt.tight_layout()
    
    # 保存图形
    plt.savefig("./ore_theoretical_performance.pdf", format='pdf')
    plt.savefig("./ore_theoretical_performance.png", format='png', dpi=300)
    
    plt.show()
    
    # 打印结果表格
    print("\n" + "="*80)
    print("THEORETICAL PERFORMANCE ANALYSIS - SINGLE ENCRYPTION TIME")
    print("="*80)
    print(f"{'Data Size':<12} {'BlockOPE':<15} {'HybridORE':<15} {'FreORE':<15} {'EncodeORE':<15}")
    print(f"{'(n)':<12} {'O(log n)':<15} {'O(1)':<15} {'O(1)':<15} {'O(1)':<15}")
    print("-"*80)
    
    for i, size in enumerate(data_sizes):
        print(f"{x_labels[i]:<12} {blockope_times[i]:<15.2f} {hybridore_times[i]:<15.2f} "
              f"{freore_times[i]:<15.2f} {encodeore_times[i]:<15.2f}")
    
    print("="*80)
    print("Time unit: microseconds (μs)")
    print("\nKey observations:")
    print("• BlockOPE: Time grows logarithmically due to tree traversal")
    print("• Other schemes: Constant time per encryption, independent of data size")
    print(f"• At n=10^6: BlockOPE is {blockope_times[-1]/freore_times[-1]:.1f}x slower than FreORE")

def plot_complexity_comparison():
    """绘制时间复杂度对比图（更大范围）"""
    
    # 扩展数据范围到10^8以更好展示复杂度差异
    exponents = np.arange(1, 9)  # [1, 2, 3, 4, 5, 6, 7, 8]
    data_sizes = [10**i for i in exponents]
    x_labels = [f"$10^{i}$" for i in exponents]
    
    # 重新计算更大范围的时间
    blockope_times = [10 + 5 * np.log2(n) for n in data_sizes]
    constant_time = [55] * len(data_sizes)  # 其他方案的平均时间
    
    fig, ax = plt.subplots()
    
    x = np.arange(len(data_sizes))
    
    # 绘制复杂度对比
    ax.plot(x, blockope_times, linewidth=3.0, color=color_1, marker='^', 
           markerfacecolor=color_1, markeredgewidth=2, markersize=10, 
           label='BlockOPE: O(log n)')
    
    ax.plot(x, constant_time, linewidth=3.0, color=color_3, marker='o', 
           markerfacecolor=color_3, markeredgewidth=2, markersize=10, 
           label='Other ORE schemes: O(1)')
    
    # 设置坐标轴
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel('Data Size (n)')
    ax.set_ylabel('Single Encryption Time (μs)')
    ax.set_title('Time Complexity Comparison: ORE Encryption Schemes')
    
    # 网格和图例
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.legend(loc='upper left', prop={'size': 14})
    
    # 添加注释
    ax.annotate(f'At n=$10^8$: {blockope_times[-1]:.1f}μs', 
                xy=(len(x_labels)-1, blockope_times[-1]), 
                xytext=(len(x_labels)-2, blockope_times[-1]+10),
                arrowprops=dict(arrowstyle='->', color=color_1, lw=1.5),
                fontsize=11, color=color_1)
    
    ax.annotate(f'Constant: {constant_time[0]}μs', 
                xy=(len(x_labels)-1, constant_time[-1]), 
                xytext=(len(x_labels)-2, constant_time[-1]-15),
                arrowprops=dict(arrowstyle='->', color=color_3, lw=1.5),
                fontsize=11, color=color_3)
    
    plt.tight_layout()
    
    # 保存图形
    plt.savefig("./ore_complexity_comparison.pdf", format='pdf')
    plt.savefig("./ore_complexity_comparison.png", format='png', dpi=300)
    
    plt.show()

if __name__ == "__main__":
    print("Generating theoretical performance analysis for ORE schemes...")
    
    # 生成主要性能对比图
    plot_performance()
    
    print("\nGenerating extended complexity comparison...")
    
    # 生成复杂度对比图
    plot_complexity_comparison()
    
    print("\nAnalysis complete! Generated files:")
    print("• ore_theoretical_performance.pdf/png - Main performance comparison")
    print("• ore_complexity_comparison.pdf/png - Extended complexity analysis")