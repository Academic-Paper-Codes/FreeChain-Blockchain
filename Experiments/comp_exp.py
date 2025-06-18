from BKORE import BlockOPE
from EncodeORE import EncodeORE
from FreORE import FreORE
from HybridORE import HybridORE
from tqdm import tqdm
import time
from Crypto.Random import get_random_bytes
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# 绘图参数全家桶
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '8.4, 4.6',  # 调整为双子图的宽度
    'figure.dpi':'300',
    'figure.subplot.left':'0.08',
    'figure.subplot.right':'0.95',
    'figure.subplot.bottom':'0.15',
    'figure.subplot.top':'0.92',
    'figure.subplot.wspace':'0.1',  # 子图间距
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

def compare_time_experiment_modified():
    # 配置参数
    schemes_regular = ['FreORE', 'EncodeORE', 'HybridORE', 'BlockOPE']
    schemes_trap = ['FreORE-trap', 'EncodeORE', 'HybridORE', 'BlockOPE']
    n_compares = 1000  # 比较次数
    num_ct = 100       # 每个方案生成的密文数量
    time_regular = {}
    time_trap = {}

    # 初始化加密方案
    freore = FreORE(10, 1000, 10, 9, b"secret_key", 1, 3)
    encode_ore = EncodeORE(l1=8, l2=4)
    block_ope = BlockOPE()
    hybrid_ore = HybridORE(b"lewis_wu_secret_key", b"clww_secret_key", l1=4, l2=4)
    key = get_random_bytes(16)  # BlockOPE使用的密钥

    # 测试常规比较
    for scheme in tqdm(schemes_regular, desc="Testing Regular Comparisons"):
        # 生成密文池
        ct_pool = []
        for _ in range(num_ct):
            m = np.random.randint(0, 100)
            if scheme == 'FreORE':
                ct = freore.data_encrypt(m)
            elif scheme == 'EncodeORE':
                ct = encode_ore.encrypt(m)
            elif scheme == 'HybridORE':
                ct = hybrid_ore.encrypt(m)
            elif scheme == 'BlockOPE':
                ct = block_ope.encrypt(m, key)
            ct_pool.append(ct)

        # 执行比较测试
        total_time = 0.0
        for _ in range(n_compares):
            ct1, ct2 = random.sample(ct_pool, 2)
            start = time.perf_counter()
            if scheme == 'FreORE':
                freore._compare(ct1, ct2)
            elif scheme == 'EncodeORE':
                encode_ore.compare(ct1, ct2)
            elif scheme == 'HybridORE':
                hybrid_ore.compare(ct1, ct2)
            elif scheme == 'BlockOPE':
                block_ope.compare(ct1, ct2)
            total_time += time.perf_counter() - start
        
        time_regular[scheme] = total_time / n_compares

    # 测试FreORE陷阱比较
    tqdm.write("\nTesting Trapdoor Comparisons...")
    trap_ct_pool = [freore.trap_encrypt(np.random.randint(0, 100)) for _ in range(num_ct)]
    data_ct_pool = [freore.data_encrypt(np.random.randint(0, 100)) for _ in range(num_ct)]
    
    total_time = 0.0
    for _ in tqdm(range(n_compares), desc="FreORE-trap"):
        trap_ct = random.choice(trap_ct_pool)
        data_ct = random.choice(data_ct_pool)
        start = time.perf_counter()
        freore.trap_compare(trap_ct, data_ct)
        total_time += time.perf_counter() - start
    
    time_trap['FreORE-trap'] = total_time / n_compares
    for scheme in ['EncodeORE', 'HybridORE', 'BlockOPE']:
        time_trap[scheme] = time_regular[scheme]  # 复用常规时间

    # 可视化结果
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    # 颜色和图案配置
    colors = [color_1, color_2, color_3, color_4]
    hatches = ["-----", "/////", "|||||", "....."]
    
    trap_colors = [color_5 if s == 'FreORE-trap' else colors[i%4] 
                   for i, s in enumerate(schemes_trap)]
    trap_hatches = ["xxxxx" if s == 'FreORE-trap' else hatches[i%4] 
                    for i, s in enumerate(schemes_trap)]

    # 常规比较柱状图
    x1 = np.arange(len(schemes_regular))
    bars1 = ax1.bar(x1, 
                    [time_regular[s]*1000 for s in schemes_regular],
                    color='none',
                    edgecolor=colors,
                    hatch=[hatches[i] for i in range(len(schemes_regular))],
                    alpha=.99)
    
    ax1.set_title('Regular Comparison Time')
    ax1.set_ylabel('Time per Comparison (ms)')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(schemes_regular, rotation=15)
    
    # 添加数据标签
    for i, bar in enumerate(bars1):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

    # 陷阱比较柱状图
    x2 = np.arange(len(schemes_trap))
    bars2 = ax2.bar(x2,
                    [time_trap[s]*1000 for s in schemes_trap],
                    color='none',
                    edgecolor=trap_colors,
                    hatch=trap_hatches,
                    alpha=.99)
    
    ax2.set_title('Trapdoor Comparison Time')
    ax2.set_ylabel('Time per Comparison (ms)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(schemes_trap, rotation=15)
    
    # 添加数据标签
    for i, bar in enumerate(bars2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{height:.4f}',
                ha='center', va='bottom', fontsize=10)

    # 添加网格
    ax1.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    ax2.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)

    # 图例
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor='none', edgecolor=colors[i], 
                                   hatch=hatches[i], alpha=.99, label=schemes_regular[i]) 
                      for i in range(len(schemes_regular))]
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='none', edgecolor=color_5, 
                                       hatch="xxxxx", alpha=.99, label='FreORE-trap'))
    
    fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, 0.02), 
              ncol=5, columnspacing=0.4, prop={'size': 10})

    plt.tight_layout()
    plt.savefig("./comparison_time_analysis.pdf", format='pdf')
    plt.show()

if __name__ == "__main__":
    compare_time_experiment_modified()