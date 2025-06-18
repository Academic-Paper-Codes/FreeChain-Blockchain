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


def measure_storage_costs(data_sizes, block_size):
    """测量不同算法在不同数据规模下的存储成本"""
    freore = MockFreORE()
    
    costs = {
        'CVTree': [],
        'BVTree': [],
        'BlockOPE': [],
        'Plaintext': []  # 原始数据存储成本作为基准
    }
    
    for size in data_sizes:
        print(f"Testing data size: {size} with block size: {block_size}")
        
        # 生成测试数据
        test_data = list(range(1, size + 1))
        random.shuffle(test_data)
        
        # CVTree测试
        cvtree = CVTree(freore)
        for i, data in enumerate(test_data):
            cvtree.insert(data, f"addr_{i}")
        cv_cost = cvtree.get_storage_size()
        costs['CVTree'].append(cv_cost)
        
        # BVTree测试
        bvtree = BVTree(freore, block_size=block_size)
        for i, data in enumerate(test_data):
            bvtree.insert(data, f"addr_{i}")
        bv_cost = bvtree.get_storage_size()
        costs['BVTree'].append(bv_cost)
        
        # BlockOPE测试
        blockope = BlockOPE()
        key = get_random_bytes(16)
        for data in test_data:
            blockope.encrypt(data, key)
        ope_cost = blockope.get_storage_size() / 1024  # 转换为KB
        costs['BlockOPE'].append(ope_cost)
        
        # 原始数据存储成本（每个数据项32B地址 + 8B数据）
        plaintext_cost = size * 40 / 1024  # KB
        costs['Plaintext'].append(plaintext_cost)
        
        print(f"  CVTree: {cv_cost:.2f} KB")
        print(f"  BVTree: {bv_cost:.2f} KB") 
        print(f"  BlockOPE: {ope_cost:.2f} KB")
        print(f"  Plaintext: {plaintext_cost:.2f} KB")
    
    return costs

def plot_storage_comparison(data_sizes, costs, block_size, subplot_id):
    """绘制存储成本对比图（折线图）"""
    fig, ax = plt.subplots()
    
    x = np.arange(len(data_sizes))
    
    # 绘制折线图（参考accuracy.py中注释掉的折线图代码）
    plt.plot(x, costs['CVTree'], linewidth=2.0, color=color_1, marker='^',
             markerfacecolor=color_1, markeredgewidth=1.5, markersize=8, label='CVTree')
    
    plt.plot(x, costs['BVTree'], linewidth=2.0, color=color_2, marker='s', 
             markerfacecolor=color_2, markeredgewidth=1.5, markersize=8, label='BVTree')
    
    plt.plot(x, costs['BlockOPE'], linewidth=2.0, color=color_3, marker='*', 
             markerfacecolor=color_3, markeredgewidth=1.5, markersize=8, label='BlockOPE')
    
    plt.plot(x, costs['Plaintext'], linewidth=2.0, color=color_4, marker='o', 
             markerfacecolor=color_4, markeredgewidth=1.5, markersize=8, label='Plaintext')
    
    # 设置x轴标签
    x_ticks = np.arange(len(data_sizes))
    labels = [f'$10^{{{int(np.log10(size))}}}$' for size in data_sizes]
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)
    
    # 设置y轴为对数刻度
    ax.set_yscale('log')
    ax.set_ylabel('Storage Cost (KB)')
    ax.set_xlabel('Data Size')
    
    # 添加标题（可选）
    # ax.set_title(f'Storage Costs (Block Size = $10^{{{int(np.log10(block_size))}}}$)')
    
    # 网格和图例
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 10})
    
    plt.tight_layout()
    plt.savefig(f"./storage_costs_blocksize_{block_size}.pdf", format='pdf')
    plt.show()

def main():
    # 测试数据规模：10^1 到 10^6
    data_sizes = [10**i for i in range(1, 7)]  # [10, 100, 1000, 10000, 100000, 1000000]
    
    # 由于计算量大，我们使用较小的测试规模
    test_sizes = [10, 100, 1000, 10000, 100000]  # 实际测试用的规模
    
    print("开始测试存储成本...")
    
    # 测试两种block_size
    for block_size in [1000, 10000]:  # 10^3 和 10^4
        print(f"\n=== Testing with block size: {block_size} ===")
        costs = measure_storage_costs(test_sizes, block_size)
        plot_storage_comparison(test_sizes, costs, block_size, 
                              'a' if block_size == 1000 else 'b')

if __name__ == "__main__":
    main()