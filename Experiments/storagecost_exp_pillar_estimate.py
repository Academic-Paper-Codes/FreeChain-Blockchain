# storage_comparison.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import hashlib
import random
from Crypto.Random import get_random_bytes
from cvtree import CVTree
from bvtree import BVTree
from BKORE import BlockOPE
from scipy import stats
from sklearn.linear_model import LinearRegression

# 使用accuracy.py中的绘图参数
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

# 颜色定义（来自accuracy.py）
color_1 = "#F27970"
color_2 = "#BB9727"
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#05B9E2"

# 简化的FreORE模拟类
class MockFreORE:
    def __init__(self):
        self.key = get_random_bytes(16)
    
    def data_encrypt(self, plaintext):
        # 简单的三进制编码模拟
        binary = bin(abs(int(plaintext)))[2:]
        ternary = ""
        for bit in binary:
            ternary += str(random.randint(0, 2))
        return ternary.ljust(32, '0')[:32]  # 固定32位长度
    
    def trap_encrypt(self, plaintext):
        return self.data_encrypt(plaintext)
    
    def data_compare(self, c1, c2):
        return -1 if c1 < c2 else (1 if c1 > c2 else 0)
    
    def trap_compare(self, trapdoor, ciphertext):
        return self.data_compare(trapdoor, ciphertext)

def estimate_large_scale_costs(measured_sizes, measured_costs, target_sizes):
    """基于已测量的数据估算大规模的存储成本"""
    estimated_costs = {}
    
    for algorithm in measured_costs.keys():
        costs = measured_costs[algorithm]
        
        if algorithm == 'Plaintext':
            # Plaintext是线性关系，直接按比例计算
            ratio = costs[-1] / measured_sizes[-1]  # KB per data item
            estimated_costs[algorithm] = [ratio * size for size in target_sizes]
        
        elif algorithm == 'BlockOPE':
            # BlockOPE存储成本近似线性（每个节点固定开销）
            # 使用线性回归
            X = np.array(measured_sizes).reshape(-1, 1)
            y = np.array(costs)
            model = LinearRegression().fit(X, y)
            estimated_costs[algorithm] = [max(0, model.predict([[size]])[0]) for size in target_sizes]
        
        elif algorithm == 'BVTree':
            # BVTree的成本主要取决于块数量，近似对数增长
            # 块数量 = ceil(size / block_size)
            # 每块固定开销 + 线性数据存储成本
            if len(costs) >= 2:
                # 估算每个数据项的平均成本（包含索引开销的分摊）
                avg_cost_per_item = costs[-1] / measured_sizes[-1]
                estimated_costs[algorithm] = [avg_cost_per_item * size for size in target_sizes]
            else:
                estimated_costs[algorithm] = costs + [costs[-1] * (size/measured_sizes[-1]) for size in target_sizes[len(costs):]]
        
        elif algorithm == 'CVTree':
            # CVTree的成本主要来自前缀树节点数量
            # 节点数量大致与 size * log(size) 成正比（考虑前缀共享）
            if len(measured_sizes) >= 3:
                # 使用对数线性模型: y = a * x * log(x) + b
                log_sizes = [size * np.log(size) for size in measured_sizes]
                X = np.array(log_sizes).reshape(-1, 1)
                y = np.array(costs)
                model = LinearRegression().fit(X, y)
                
                estimated_log_sizes = [size * np.log(size) for size in target_sizes]
                estimated_costs[algorithm] = [max(0, model.predict([[log_size]])[0]) for log_size in estimated_log_sizes]
            else:
                # 备用方法：基于最后两个点的增长率
                if len(costs) >= 2:
                    growth_factor = costs[-1] / costs[-2]
                    size_ratio = measured_sizes[-1] / measured_sizes[-2]
                    adjusted_growth = growth_factor / size_ratio
                    
                    estimated_costs[algorithm] = costs.copy()
                    last_cost = costs[-1]
                    last_size = measured_sizes[-1]
                    
                    for size in target_sizes[len(costs):]:
                        # 考虑对数增长特性
                        size_mult = size / last_size
                        log_mult = np.log(size) / np.log(last_size)
                        estimated_cost = last_cost * size_mult * log_mult
                        estimated_costs[algorithm].append(estimated_cost)
                        last_cost = estimated_cost
                        last_size = size
                else:
                    estimated_costs[algorithm] = costs
    
    return estimated_costs

def measure_storage_costs(data_sizes, block_size, estimate_threshold=50000):
    """测量不同算法在不同数据规模下的存储成本，超过阈值则估算"""
    freore = MockFreORE()
    
    costs = {
        'CVTree': [],
        'BVTree': [],
        'BlockOPE': [],
        'Plaintext': []
    }
    
    measured_sizes = []
    measured_costs = {key: [] for key in costs.keys()}
    
    for i, size in enumerate(data_sizes):
        if size <= estimate_threshold:
            print(f"Testing data size: {size} with block size: {block_size}")
            
            # 生成测试数据
            test_data = list(range(1, size + 1))
            random.shuffle(test_data)
            
            # CVTree测试
            cvtree = CVTree(freore)
            for j, data in enumerate(test_data):
                cvtree.insert(data, f"addr_{j}")
            cv_cost = cvtree.get_storage_size()
            measured_costs['CVTree'].append(cv_cost)
            
            # BVTree测试
            bvtree = BVTree(freore, block_size=block_size)
            for j, data in enumerate(test_data):
                bvtree.insert(data, f"addr_{j}")
            bv_cost = bvtree.get_storage_size()
            measured_costs['BVTree'].append(bv_cost)
            
            # BlockOPE测试
            blockope = BlockOPE()
            key = get_random_bytes(16)
            for data in test_data:
                blockope.encrypt(data, key)
            ope_cost = blockope.get_storage_size() / 1024  # 转换为KB
            measured_costs['BlockOPE'].append(ope_cost)
            
            # 原始数据存储成本
            plaintext_cost = size * 40 / 1024  # KB
            measured_costs['Plaintext'].append(plaintext_cost)
            
            measured_sizes.append(size)
            
            print(f"  CVTree: {cv_cost:.2f} KB")
            print(f"  BVTree: {bv_cost:.2f} KB") 
            print(f"  BlockOPE: {ope_cost:.2f} KB")
            print(f"  Plaintext: {plaintext_cost:.2f} KB")
        else:
            print(f"Estimating data size: {size} based on previous measurements...")
            break
    
    # 如果有需要估算的大规模数据
    if len(measured_sizes) < len(data_sizes):
        remaining_sizes = data_sizes[len(measured_sizes):]
        estimated_costs = estimate_large_scale_costs(measured_sizes, measured_costs, data_sizes)
        
        # 打印估算结果
        for i, size in enumerate(remaining_sizes):
            idx = len(measured_sizes) + i
            print(f"Estimated for size {size}:")
            print(f"  CVTree: {estimated_costs['CVTree'][idx]:.2f} KB")
            print(f"  BVTree: {estimated_costs['BVTree'][idx]:.2f} KB") 
            print(f"  BlockOPE: {estimated_costs['BlockOPE'][idx]:.2f} KB")
            print(f"  Plaintext: {estimated_costs['Plaintext'][idx]:.2f} KB")
        
        return estimated_costs
    else:
        return measured_costs

def plot_storage_comparison(data_sizes, costs, block_size, subplot_id):
    """绘制存储成本对比图"""
    fig, ax = plt.subplots()
    
    x = np.arange(len(data_sizes))
    width = 0.2
    print(f"costs : {costs}")
    
    # 绘制柱状图
    ax.bar(x - 1.5*width, costs['CVTree'], width, 
           color='none', label='CVTree', edgecolor=color_1, hatch="-----", alpha=.99)
    
    ax.bar(x - 0.5*width, costs['BVTree'], width, 
           color='none', label='BVTree', edgecolor=color_2, hatch="/////", alpha=.99)
    
    ax.bar(x + 0.5*width, costs['BlockOPE'], width, 
           color='none', label='BlockOPE', edgecolor=color_3, hatch="|||||", alpha=.99)
    
    # ax.bar(x + 1.5*width, costs['Plaintext'], width, 
    #        color='none', label='Plaintext', edgecolor=color_4, hatch=".....", alpha=.99)
    
    # 设置x轴标签
    ax.set_xticks(x)
    ax.set_xticklabels([f'$10^{{{int(np.log10(size))}}}$' for size in data_sizes])
    
    # 设置y轴为对数刻度
    ax.set_yscale('log')
    ax.set_ylabel('Storage Cost (KB)')
    ax.set_xlabel('Number of Data Records')
    
    # 网格和图例
    ax.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    ax.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 10})
    
    plt.tight_layout()
    plt.savefig(f"./storage_costs_blocksize_{block_size}.pdf", format='pdf')
    plt.show()

def main():
    # 完整的测试数据规模：10^1 到 10^6
    data_sizes = [10**i for i in range(1, 7)]  # [10, 100, 1000, 10000, 100000, 1000000]
    
    print("开始测试存储成本...")
    print("注意：超过50000的数据规模将基于前面的测量结果进行估算")
    
    # 测试两种block_size
    for block_size in [1000, 10000]:  # 10^3 和 10^4
        print(f"\n=== Testing with block size: {block_size} ===")
        costs = measure_storage_costs(data_sizes, block_size, estimate_threshold=50000)
        plot_storage_comparison(data_sizes, costs, block_size, 
                              'a' if block_size == 1000 else 'b')

if __name__ == "__main__":
    main()