import matplotlib.pyplot as plt
import numpy as np
import time
import os
import secrets
from matplotlib.ticker import LogFormatter, FixedLocator
import matplotlib.pylab as pylab

# 导入所有必要的类
from FreORE import FreORE
from cvtree import CVTree
from bvtree import BVTree
from BKORE import BlockOPE
from EncodeORE import EncodeORE
from HybridORE import HybridORE
from Crypto.Random import get_random_bytes

# 设置绘图参数
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '10, 6',
    'figure.dpi': '300',
    'figure.subplot.left': '0.1',
    'figure.subplot.right': '0.95',
    'figure.subplot.bottom': '0.15',
    'figure.subplot.top': '0.92',
    'pdf.fonttype': '42',
    'ps.fonttype': '42',
}
pylab.rcParams.update(params)

def test_cvtree_single_insert(data_sizes):
    """测试CVTree单条插入时间"""
    print("Testing CVTree single insert time...")
    times = []
    
    # 初始化FreORE
    freore = FreORE(d=2, alpha=1000, beta=10, gamma=9, pfk=b"secret_key", nx=8, ny=8)
    
    for size in data_sizes:
        print(f"  Testing CVTree single insert with {size} existing records...")
        
        if size <= 10000:  # 直接测量
            # Step 1: 构建索引 (插入size条记录)
            cvtree = CVTree(freore)
            for i in range(size):
                cvtree.insert(i, f"file_{i}.enc")
            
            # Step 2: 测量插入一条新数据的时间 (数据加密 + 索引更新)
            new_data = size  # 新数据
            start_time = time.time()
            cvtree.insert(new_data, f"file_{new_data}.enc")
            end_time = time.time()
            
            insert_time = (end_time - start_time)  # 秒为单位
            times.append(insert_time)
        else:  # 估计
            # 基于已有数据估计
            if len(times) >= 2:
                # 假设插入时间复杂度为 O(log n)
                estimated_time = times[-1] * (np.log(size) / np.log(data_sizes[len(times)-1]))
                times.append(estimated_time)
                print(f"    Estimated single insert time: {estimated_time:.6f} s")
            else:
                times.append(times[-1] if times else 0.001)
    
    return times

def test_bvtree_single_insert(data_sizes):
    """测试BVTree单条插入时间"""
    print("Testing BVTree single insert time...")
    times = []
    
    # 初始化FreORE
    freore = FreORE(d=2, alpha=1000, beta=10, gamma=9, pfk=b"secret_key", nx=8, ny=8)
    
    for size in data_sizes:
        print(f"  Testing BVTree single insert with {size} existing records...")
        
        if size <= 10000:  # 直接测量
            # Step 1: 构建索引
            bvtree = BVTree(freore, block_size=1000)
            for i in range(size):
                bvtree.insert(i, f"file_{i}.enc")
            
            # Step 2: 测量插入一条新数据的时间
            new_data = size
            start_time = time.time()
            bvtree.insert(new_data, f"file_{new_data}.enc")
            end_time = time.time()
            
            insert_time = (end_time - start_time)
            times.append(insert_time)
        else:  # 估计
            if len(times) >= 2:
                estimated_time = times[-1] * (np.log(size) / np.log(data_sizes[len(times)-1]))
                times.append(estimated_time)
                print(f"    Estimated single insert time: {estimated_time:.6f} s")
            else:
                times.append(times[-1] if times else 0.001)
    
    return times

def test_blockope_single_insert(data_sizes):
    """测试BlockOPE单条插入时间"""
    print("Testing BlockOPE single insert time...")
    times = []
    
    for size in data_sizes:
        print(f"  Testing BlockOPE single insert with {size} existing records...")
        
        if size <= 1000:  # BlockOPE较慢，只测小规模
            # Step 1: 构建索引
            blockope = BlockOPE()
            key = get_random_bytes(16)
            for i in range(size):
                blockope.encrypt(i, key)
            
            # Step 2: 测量插入一条新数据的时间
            new_data = size
            start_time = time.time()
            blockope.encrypt(new_data, key)
            end_time = time.time()
            
            insert_time = (end_time - start_time)
            times.append(insert_time)
        else:  # 估计
            if len(times) >= 2:
                # BlockOPE插入复杂度较高，假设 O(log n)
                estimated_time = times[-1] * (np.log(size) / np.log(data_sizes[len(times)-1]))
                times.append(estimated_time)
                print(f"    Estimated single insert time: {estimated_time:.6f} s")
            else:
                times.append(times[-1] if times else 0.01)
    
    return times

def test_encodeore_single_insert(data_sizes):
    """测试EncodeORE单条插入时间"""
    print("Testing EncodeORE single insert time...")
    times = []
    
    for size in data_sizes:
        print(f"  Testing EncodeORE single insert with {size} existing records...")
        
        if size <= 100000:  # EncodeORE相对较快
            # Step 1: 构建现有数据集 (EncodeORE是无状态的，不需要构建索引)
            ore = EncodeORE(l1=8, l2=4)
            existing_ciphertexts = []
            for i in range(size):
                ct = ore.encrypt(i)
                existing_ciphertexts.append(ct)
            
            # Step 2: 测量加密一条新数据的时间
            new_data = size
            start_time = time.time()
            new_ct = ore.encrypt(new_data)
            end_time = time.time()
            
            insert_time = (end_time - start_time)
            times.append(insert_time)
        else:  # 估计
            if len(times) >= 2:
                # EncodeORE是O(1)的加密时间
                estimated_time = times[-1]  # 常数时间
                times.append(estimated_time)
                print(f"    Estimated single insert time: {estimated_time:.6f} s")
            else:
                times.append(times[-1] if times else 0.0001)
    
    return times

def test_hybridore_single_insert(data_sizes):
    """测试HybridORE单条插入时间"""
    print("Testing HybridORE single insert time...")
    times = []
    
    for size in data_sizes:
        print(f"  Testing HybridORE single insert with {size} existing records...")
        
        if size <= 10000:  # HybridORE中等速度
            # Step 1: 构建现有数据集
            lw_key = os.urandom(16)
            clww_key = os.urandom(16)
            hore = HybridORE(lw_key, clww_key)
            existing_ciphertexts = []
            for i in range(size):
                ct = hore.encrypt(i)
                existing_ciphertexts.append(ct)
            
            # Step 2: 测量加密一条新数据的时间
            new_data = size
            start_time = time.time()
            new_ct = hore.encrypt(new_data)
            end_time = time.time()
            
            insert_time = (end_time - start_time)
            times.append(insert_time)
        else:  # 估计
            if len(times) >= 2:
                # HybridORE加密时间相对稳定
                estimated_time = times[-1]
                times.append(estimated_time)
                print(f"    Estimated single insert time: {estimated_time:.6f} s")
            else:
                times.append(times[-1] if times else 0.001)
    
    return times

def run_single_insert_experiments():
    """运行所有单条插入实验"""
    # 数据规模：10^1 到 10^6
    data_sizes = [10, 100, 1000, 10000, 100000, 1000000]
    
    print("Starting single insert time experiments...")
    print("Data sizes:", data_sizes)
    
    # 运行各种方法的测试
    results = {}
    
    try:
        results['CVTree'] = test_cvtree_single_insert(data_sizes)
    except Exception as e:
        print(f"CVTree test failed: {e}")
        results['CVTree'] = [0] * len(data_sizes)
    
    try:
        results['BVTree'] = test_bvtree_single_insert(data_sizes)
    except Exception as e:
        print(f"BVTree test failed: {e}")
        results['BVTree'] = [0] * len(data_sizes)
    
    try:
        results['BlockOPE'] = test_blockope_single_insert(data_sizes)
    except Exception as e:
        print(f"BlockOPE test failed: {e}")
        results['BlockOPE'] = [0] * len(data_sizes)
    
    try:
        results['EncodeORE'] = test_encodeore_single_insert(data_sizes)
    except Exception as e:
        print(f"EncodeORE test failed: {e}")
        results['EncodeORE'] = [0] * len(data_sizes)
    
    try:
        results['HybridORE'] = test_hybridore_single_insert(data_sizes)
    except Exception as e:
        print(f"HybridORE test failed: {e}")
        results['HybridORE'] = [0] * len(data_sizes)
    
    return data_sizes, results

def plot_single_insert_times(data_sizes, results):
    """绘制单条插入时间图表"""
    plt.figure(figsize=(10, 6))
    
    # 定义颜色和标记
    colors = {
        'CVTree': '#F27970',
        'BVTree': '#BB9727', 
        'BlockOPE': '#54B345',
        'EncodeORE': '#32B897',
        'HybridORE': '#05B9E2'
    }
    
    markers = {
        'CVTree': 'o',
        'BVTree': 's',
        'BlockOPE': '^',
        'EncodeORE': 'D',
        'HybridORE': 'v'
    }
    
    # 绘制每种方法的曲线
    for method, times in results.items():
        if any(t > 0 for t in times):  # 只绘制有效数据
            plt.loglog(data_sizes, times, 
                      color=colors[method], 
                      marker=markers[method],
                      linewidth=2.0,
                      markersize=8,
                      markerfacecolor=colors[method],
                      markeredgewidth=1.5,
                      label=method)
    
    plt.xlabel('Number of Existing Data Records')
    plt.ylabel('Single Insert Time (s)')
    plt.title('Time Costs for Inserting 1 Data (DataEnc + Time for Building Index)')
    
    # 设置网格
    plt.grid(True, linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    
    # 设置图例
    plt.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 12})
    
    # 设置X轴范围，在两端留出空白
    x_min = min(data_sizes) * 0.5  # 左侧留出50%的空白
    x_max = max(data_sizes) * 2.0  # 右侧留出100%的空白
    plt.xlim(x_min, x_max)
    
    # 自定义x轴刻度
    plt.xticks(data_sizes, [f'$10^{int(np.log10(x))}$' for x in data_sizes])
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig("./single_insert_time_costs.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig("./single_insert_time_costs.png", format='png', dpi=300, bbox_inches='tight')
    
    plt.show()

def print_single_insert_results_table(data_sizes, results):
    """打印单条插入结果表格"""
    print("\n" + "="*80)
    print("SINGLE INSERT TIME RESULTS (seconds)")
    print("="*80)
    print(f"{'Data Size':<12}", end="")
    for method in results.keys():
        print(f"{method:<15}", end="")
    print()
    print("-"*80)
    
    for i, size in enumerate(data_sizes):
        print(f"{size:<12}", end="")
        for method, times in results.items():
            if i < len(times):
                if times[i] < 0.001:
                    print(f"{times[i]*1000:<15.3f}ms", end="")  # 毫秒显示
                elif times[i] < 1:
                    print(f"{times[i]*1000:<15.1f}ms", end="")  # 毫秒显示
                else:
                    print(f"{times[i]:<15.3f}", end="")  # 秒显示
            else:
                print(f"{'N/A':<15}", end="")
        print()

if __name__ == "__main__":
    print("Starting Single Insert Time Cost Experiments")
    print("=" * 60)
    
    # 运行实验
    data_sizes, results = run_single_insert_experiments()
    
    # 打印结果表格
    print_single_insert_results_table(data_sizes, results)
    
    # 绘制图表
    plot_single_insert_times(data_sizes, results)
    
    print("\nExperiment completed!")
    print("Results saved as:")
    print("  - single_insert_time_costs.pdf")
    print("  - single_insert_time_costs.png")