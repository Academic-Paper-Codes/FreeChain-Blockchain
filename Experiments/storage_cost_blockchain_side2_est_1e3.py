# -*- coding: utf-8 -*-
"""
Quick-draw version: directly plot the estimated storage costs
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# ---------- 1. 采用新的绘图参数 ----------
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

# ---------- 2. 颜色定义（与storage_comparison.py保持一致） ----------
color_1 = "#F27970"  # FreeChain-C
color_2 = "#BB9727"  # FreeChain-B
color_3 = "#54B345"  # EncodeORE
color_4 = "#32B897"  # BlockOPE
color_5 = "#05B9E2"  # HybridORE

def plot_storage_comparison(data_sizes, costs, block_size):
    """绘制存储成本对比图"""
    fig, ax = plt.subplots()
    
    x = np.arange(len(data_sizes))
    width = 0.15  # 调整为5个柱子的宽度
    
    # 绘制柱状图（与storage_comparison.py相同的风格）
    ax.bar(x - 2*width, costs['FreeChain-C'], width, 
           color='none', label='FreeChain-C', edgecolor=color_1, hatch="-----", alpha=.99)
    
    ax.bar(x - width, costs['FreeChain-B'], width, 
           color='none', label='FreeChain-B', edgecolor=color_2, hatch="/////", alpha=.99)
    
    ax.bar(x, costs['EncodeORE'], width, 
           color='none', label='EncodeORE', edgecolor=color_3, hatch="\\\\\\\\\\", alpha=.99)
    
    ax.bar(x + width, costs['HybridORE'], width, 
           color='none', label='HybridORE', edgecolor=color_5, hatch="+++++", alpha=.99)
    
    ax.bar(x + 2*width, costs['BlockOPE'], width, 
           color='none', label='BlockOPE', edgecolor=color_4, hatch="|||||", alpha=.99)
    
    # 设置x轴标签
    ax.set_xticks(x)
    ax.set_xticklabels([f'$10^{{{int(np.log10(size))}}}$' for size in data_sizes])
    
    # 设置y轴为对数刻度
    ax.set_yscale('log')
    ax.set_ylabel('Storage Cost (KB)')
    ax.set_xlabel('Number of Data Records')
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    # 设置y轴范围
    ax.set_ylim(0.1, 1000000)
    
    # 网格和图例
    ax.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    ax.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 10})
    
    plt.tight_layout()
    plt.savefig(f"./storage_costs_on_blockchain_side_{block_size}.pdf", format='pdf')
    plt.show()

def main():
    # ---------- 3. 数据定义 ----------
    data_sizes = [10**i for i in range(1, 7)]  # [10, 100, 1000, 10000, 100000, 1000000]
    
    # 将数据组织成字典格式（与storage_comparison.py一致）
    costs = {
        'FreeChain-C': [0.6, 6, 55, 750, 10000, 130000],      # KB
        'FreeChain-B': [0.25, 3, 35, 650, 8500, 110000],      # KB
        'EncodeORE': [0.9, 13, 120, 2500, 27200, 279300],     # KB
        'BlockOPE': [1, 18, 160, 2800, 40000, 500000],        # KB
        'HybridORE': [0.95, 17.5, 156, 2700, 37700, 389790]   # KB
    }
    
    # ---------- 4. 绘图 ----------
    print("绘制存储成本对比图...")
    print(f"数据规模: {data_sizes}")
    
    # 打印各算法的存储成本
    for algorithm, cost_list in costs.items():
        print(f"\n{algorithm}:")
        for size, cost in zip(data_sizes, cost_list):
            print(f"  Size {size}: {cost} KB")
    
    # 绘制图表（block_size = 1e3）
    plot_storage_comparison(data_sizes, costs, '1e3')
    
    print("\n图表已保存至: ./storage_costs_on_blockchain_side_1e3.pdf")

if __name__ == "__main__":
    main()