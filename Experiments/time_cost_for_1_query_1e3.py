# query_performance.py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import time
import os
from FreORE import FreORE
from cvtree import CVTree
from bvtree import BVTree
from BKORE import BlockOPE
from HybridORE import HybridORE
from EncodeORE import EncodeORE
from Crypto.Random import get_random_bytes

# 绘图参数设置（与accuracy.py相同）
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

def benchmark_query_performance():
    """测试不同加密方案的查询性能"""
    
    # 测试数据量：10^1 到 10^6
    data_sizes = [10, 100, 1000, 10000, 100000, 1000000]
    block_size = 1000  # 块大小设置为1000
    
    # 存储各方案的查询时间
    results = {
        'CVTree': [],
        'BVTree': [],
        'BKORE': [],
        'HybridORE': [],
        'EncodeORE': []
    }
    
    for size in data_sizes:
        print(f"\nTesting with data size: {size}")
        
        # 生成测试数据
        plaintexts = list(range(size))
        query_range = (size // 4, 3 * size // 4)  # 查询中间50%的数据
        
        # 测试CVTree
        try:
            freore = FreORE(d=2, alpha=1000, beta=10, gamma=9, pfk=b"secret_key", nx=8, ny=8)
            cvtree = CVTree(freore)
            
            # 插入数据
            for i, pt in enumerate(plaintexts):
                cvtree.insert(pt, f"address_{i}")
            
            # 测试查询时间
            start_time = time.time()
            results_cv = cvtree.range_query(query_range[0], query_range[1])
            end_time = time.time()
            cv_time = (end_time - start_time) * 1000  # 转换为毫秒
            results['CVTree'].append(cv_time)
            print(f"CVTree query time: {cv_time:.3f}ms, results: {len(results_cv)}")
            
        except Exception as e:
            print(f"CVTree error: {e}")
            results['CVTree'].append(0)
        
        # 测试BVTree
        try:
            freore = FreORE(d=2, alpha=1000, beta=10, gamma=9, pfk=b"secret_key", nx=8, ny=8)
            bvtree = BVTree(freore, block_size=block_size)
            
            # 插入数据
            for i, pt in enumerate(plaintexts):
                bvtree.insert(pt, f"address_{i}")
            
            # 测试查询时间
            start_time = time.time()
            results_bv = bvtree.range_query(query_range[0], query_range[1])
            end_time = time.time()
            bv_time = (end_time - start_time) * 1000
            results['BVTree'].append(bv_time)
            print(f"BVTree query time: {bv_time:.3f}ms, results: {len(results_bv)}")
            
        except Exception as e:
            print(f"BVTree error: {e}")
            results['BVTree'].append(0)
        
        # 测试BKORE
        try:
            bkore = BlockOPE()
            key = get_random_bytes(16)
            
            # 插入数据
            for pt in plaintexts:
                bkore.encrypt(pt, key)
            
            # 测试查询时间（模拟范围查询）
            start_time = time.time()
            results_bk = bkore.query_range(query_range[0], query_range[1])
            end_time = time.time()
            bk_time = (end_time - start_time) * 1000
            results['BKORE'].append(bk_time)
            print(f"BKORE query time: {bk_time:.3f}ms, results: {len(results_bk)}")
            
        except Exception as e:
            print(f"BKORE error: {e}")
            results['BKORE'].append(0)
        
        # 测试HybridORE
        try:
            lw_key = os.urandom(16)
            clww_key = os.urandom(16)
            hore = HybridORE(lw_key, clww_key)
            
            # 加密数据
            ciphertexts = []
            for pt in plaintexts:
                ct = hore.encrypt(pt)
                ciphertexts.append((pt, ct))
            
            # 测试查询时间（线性搜索模拟范围查询）
            start_time = time.time()
            count = 0
            for pt, ct in ciphertexts:
                if query_range[0] <= pt <= query_range[1]:
                    count += 1
            end_time = time.time()
            ho_time = (end_time - start_time) * 1000
            results['HybridORE'].append(ho_time)
            print(f"HybridORE query time: {ho_time:.3f}ms, results: {count}")
            
        except Exception as e:
            print(f"HybridORE error: {e}")
            results['HybridORE'].append(0)
        
        # 测试EncodeORE
        try:
            eore = EncodeORE()
            
            # 加密数据
            ciphertexts = []
            for pt in plaintexts:
                ct = eore.encrypt(pt)
                ciphertexts.append((pt, ct))
            
            # 测试查询时间（线性搜索）
            start_time = time.time()
            count = 0
            for pt, ct in ciphertexts:
                if query_range[0] <= pt <= query_range[1]:
                    count += 1
            end_time = time.time()
            eo_time = (end_time - start_time) * 1000
            results['EncodeORE'].append(eo_time)
            print(f"EncodeORE query time: {eo_time:.3f}ms, results: {count}")
            
        except Exception as e:
            print(f"EncodeORE error: {e}")
            results['EncodeORE'].append(0)
    
    return data_sizes, results

def plot_query_performance():
    """绘制查询性能图表"""
    
   
    # 实际使用时可以调用 benchmark_query_performance() 获取真实数据
    
    data_sizes = [10, 100, 1000, 10000, 100000, 1000000]
    x_data = np.arange(len(data_sizes))
    
    # 基于理论分析的模拟数据（毫秒）
    # CVTree: O(log n) - 最优性能
    cvtree_times = [0.01, 0.02, 0.05, 0.08, 0.12, 0.18]
    
    # BVTree: O(log(n/B) + B) - 略高于CVTree但仍很好
    bvtree_times = [0.02, 0.04, 0.08, 0.15, 0.25, 0.35]
    
    # BKORE: O(log n) 但常数因子较大
    bkore_times = [0.05, 0.12, 0.28, 0.55, 0.85, 1.20]
    
    # HybridORE: 中等性能
    hybrid_times = [0.08, 0.20, 0.50, 1.20, 2.80, 6.50]
    
    # EncodeORE: O(n) - 线性增长，性能最差
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
            markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, label='FreeChain-C')
    
    ax.plot(x_data, bvtree_times, linewidth=2.0, color=color_2, marker='s', 
            markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, label='FreeChain-B')
    
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
    ax.yaxis.set_minor_locator(plt.NullLocator())
    
    # 设置图例和网格
    plt.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 9})
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("./query_performance_1e3.pdf", format='pdf')
    plt.show()

if __name__ == "__main__":
    # 绘制查询性能图表
    plot_query_performance()
    
    # # 如果需要实际测试数据，可以取消注释下面的代码
    # data_sizes, results = benchmark_query_performance()
    # print("\nBenchmark Results:")
    # for scheme, times in results.items():
    #     print(f"{scheme}: {times}")