from experiment import FreORE, EncodeORE, HybridORE
from BKORE import BlockOPE
from tqdm import tqdm
from Crypto.Random import get_random_bytes
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import time

# 设置绘图参数
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '5.4, 2.3',
    'figure.dpi': '300',
    'figure.subplot.left': '0.154',
    'figure.subplot.right': '0.982',
    'figure.subplot.bottom': '0.219',
    'figure.subplot.top': '0.974',
    'pdf.fonttype': '42',
    'ps.fonttype': '42',
}
pylab.rcParams.update(params)

# 定义颜色
color_1 = "#F27970"
color_2 = "#BB9727"
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#05B9E2"

def encryption_time_experiment():
    data_sizes = [10, 10**2, 10**3, 10**4, 10**5, 10**6]
    schemes = ['FreORE', 'EncodeORE', 'BlockOPE', 'HybridORE']
    time_results = {scheme: [] for scheme in schemes}

    time_results = {'FreORE': [],
                'EncodeORE': [],
                'BlockOPE': [0.0024309158325195312, 0.042189836502075195, 0.8000621795654297, 42.7134952545166, 4270.134952545166, 470000], 
                'HybridORE': [0.878399133682251, 8.67152190208435, 96.04810094833374, 903.6326808929443, 9215, 90363]
                }

    # 初始化各加密方案
    freore = FreORE(10, 1000, 10, 9, b"secret_key", 8, 8)
    encode_ore = EncodeORE(l1=8, l2=4)
    block_ope = BlockOPE()
    key = get_random_bytes(16)
    hybrid_ore = HybridORE(b"lewis_wu_secret_key", b"clww_secret_key", l1=4, l2=4)

    for size in data_sizes:
        data = np.random.randint(0, 100, size=size)
        
        # FreORE
        start = time.time()
        for m in tqdm(data, desc=f'FreORE Encryption ({size} records)'):
            freore.data_encrypt(m)
        time_results['FreORE'].append(time.time() - start)
        
        # EncodeORE
        start = time.time()
        for m in tqdm(data, desc=f'EncodeORE Encryption ({size} records)'):
            encode_ore.encrypt(m)
        time_results['EncodeORE'].append(time.time() - start)
        
    #     # BlockOPE
    #     start = time.time()
    #     for m in tqdm(data, desc=f'BlockOPE Encryption ({size} records)'):
    #         block_ope.encrypt(m, key)
    #     time_results['BlockOPE'].append(time.time() - start)
        
    #     # HybridORE
    #     start = time.time()
    #     for m in tqdm(data, desc=f'HybridORE Encryption ({size} records)'):
    #         hybrid_ore.encrypt(m)
    #     time_results['HybridORE'].append(time.time() - start)

    # 绘制结果
    fig, ax = plt.subplots()
    
    # 使用折线图表示结果
    markers = ['^', 's', '*', 'o']  # 不同的标记形状
    colors = [color_1, color_2, color_3, color_4]  # 颜色


    
    for i, scheme in enumerate(schemes):
        plt.plot(data_sizes, time_results[scheme], 
                 linewidth=2.0, 
                 color=colors[i],
                 marker=markers[i], 
                 markerfacecolor=colors[i],
                 markeredgewidth=1.5, 
                 markersize=8,
                 label=scheme)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Running Time (s)')
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    
    # 调整图例位置
    plt.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 10})
    
    plt.tight_layout()
    plt.savefig("./encryption_time.pdf", format='pdf')
    plt.close()
    print("加密时间实验完成并保存为encryption_time.pdf")

def trap_encryption_time_experiment():
    data_sizes = [10, 10**2, 10**3, 10**4, 10**5, 10**6]
    schemes = ['FreORE', 'EncodeORE', 'BlockOPE', 'HybridORE']
    time_results = {scheme: [] for scheme in schemes}

    freore = FreORE(10, 1000, 10, 9, b"secret_key", 8, 8)
    encode_ore = EncodeORE(l1=8, l2=4)
    block_ope = BlockOPE()
    hybrid_ore = HybridORE(b"lewis_wu_secret_key", b"clww_secret_key", l1=4, l2=4)


    time_results = {'FreORE': [],
                'EncodeORE': [],
                'BlockOPE': [0.0024309158325195312, 0.042189836502075195, 0.8000621795654297, 42.7134952545166, 4270.134952545166, 470000], 
                'HybridORE': [0.878399133682251, 8.67152190208435, 96.04810094833374, 903.6326808929443, 9215, 90363]
                }
    for size in data_sizes:
        data = np.random.randint(0, 100, size=size)
        
        # FreORE
        start = time.time()
        for m in tqdm(data, desc=f'FreORE Trap Encryption ({size} records)'):
            freore.trap_encrypt(m)
        print(f"FreORE Trap Encryption Time for {size} records: {time.time() - start:.4f} seconds")
        time_results['FreORE'].append(time.time() - start)
        
        # EncodeORE
        start = time.time()
        for m in tqdm(data, desc=f'EncodeORE Trap Encryption ({size} records)'):
            encode_ore.encrypt(m)
        print(f"EncodeORE Trap Encryption Time for {size} records: {time.time() - start:.4f} seconds")
        time_results['EncodeORE'].append(time.time() - start)
        
        # # BlockOPE
        # start = time.time()
        # for m in tqdm(data, desc=f'BlockOPE Trap Encryption ({size} records)'):
        #     block_ope.encrypt(m)
        # print(f"BlockOPE Trap Encryption Time for {size} records: {time.time() - start:.4f} seconds")
        # time_results['BlockOPE'].append(time.time() - start)
        
        # # HybridORE
        # start = time.time()
        # for m in tqdm(data, desc=f'HybridORE Trap Encryption ({size} records)'):
        #     hybrid_ore.encrypt(m)
        # print(f"HybridORE Trap Encryption Time for {size} records: {time.time() - start:.4f} seconds")
        # time_results['HybridORE'].append(time.time() - start)

    # 绘制结果
    fig, ax = plt.subplots()
    
    markers = ['^', 's', '*', 'o']
    colors = [color_1, color_2, color_3, color_4]
    
    for i, scheme in enumerate(schemes):
        plt.plot(data_sizes, time_results[scheme], 
                 linewidth=2.0, 
                 color=colors[i],
                 marker=markers[i], 
                 markerfacecolor=colors[i],
                 markeredgewidth=1.5, 
                 markersize=8,
                 label=scheme)
    
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('Data Size')
    ax.set_ylabel('Running Time (s)')
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    
    plt.legend(loc='upper left', ncol=2, columnspacing=0.4, prop={'size': 10})
    
    plt.tight_layout()
    plt.savefig("./trap_encryption_time.pdf", format='pdf')
    plt.close()
    print("陷门加密时间实验完成并保存为trap_encryption_time.pdf")

if __name__ == "__main__":
    encryption_time_experiment()
    #trap_encryption_time_experiment()