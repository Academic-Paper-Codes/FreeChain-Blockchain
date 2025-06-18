# query_performance.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

# 绘图参数设置（与accuracy.py相同）
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

def plot_query_performance():
    """绘制查询性能图表（块大小 = 1e4）"""
    
    data_sizes = [10, 100, 1000, 10000, 100000, 1000000]
    x_data = np.arange(len(data_sizes))
    
    # 基于理论分析的模拟数据（毫秒）
    # CVTree: O(log n) - 最优性能（不受块大小影响）
    cvtree_times = [0.01, 0.02, 0.05, 0.08, 0.12, 0.18]
    
    # BVTree: O(log(n/B) + B) - 块大小1e4，性能比1e3提升约15-25%
    bvtree_times = [0.02, 0.04, 0.07, 0.12, 0.19, 0.27]
    
    # BKORE: O(log n) 但常数因子较大（不受块大小影响）
    bkore_times = [0.05, 0.12, 0.28, 0.55, 0.85, 1.20]
    
    # HybridORE: 中等性能（不受块大小影响）
    hybrid_times = [0.08, 0.20, 0.50, 1.20, 2.80, 6.50]
    
    # EncodeORE: O(n) - 线性增长，性能最差（不受块大小影响）
    encode_times = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    
    # 颜色设置（与accuracy.py相同）
    color_1 = "#F27970"  # CVTree
    color_2 = "#BB9727"  # BVTree  
    color_3 = "#54B345"  # BKORE
    color_4 = "#32B897"  # HybridORE
    color_5 = "#05B9E2"  # EncodeORE
    
    fig, ax = plt.subplots()
    
    # 绘制折线图
    ax.plot(x_data, cvtree_times, linewidth=2.0, color=color_1, marker='^', 
            markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, label='CVTree')
    
    ax.plot(x_data, bvtree_times, linewidth=2.0, color=color_2, marker='s', 
            markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, label='BVTree')
    
    ax.plot(x_data, bkore_times, linewidth=2.0, color=color_3, marker='*', 
            markerfacecolor=color_3, markeredgewidth=1.5, markersize=8, label='BlockOPE')
    
    ax.plot(x_data, hybrid_times, linewidth=2.0, color=color_4, marker='o', 
            markerfacecolor=color_4, markeredgewidth=1.5, markersize=8, label='HybridORE')
    
    ax.plot(x_data, encode_times, linewidth=2.0, color=color_5, marker='D', 
            markerfacecolor=color_5, markeredgewidth=1.5, markersize=8, label='EncodeORE')
    
    # 设置坐标轴
    ax.set_xticks(x_data)
    ax.set_xticklabels([f'$10^{{{int(np.log10(size))}}}$' for size in data_sizes])
    
    ax.set_xlabel('Number of Data Records')
    ax.set_ylabel('Query Time (ms)')
    
    # 使用对数坐标显示时间差异
    ax.set_yscale('log')
    
    # 设置图例和网格
    plt.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 10})
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("./query_performance_1e4.pdf", format='pdf')
    plt.show()

if __name__ == "__main__":
    # 绘制查询性能图表
    plot_query_performance()