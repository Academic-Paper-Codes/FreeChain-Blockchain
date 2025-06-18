# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 01:10:50 2022

@author: 86159
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from functools import cmp_to_key
from tqdm import tqdm
from EncodeORE import EncodeORE, sort_encrypted_data
from FreORE import FreORE
from HybridORE import HybridORE
from BKORE import BlockOPE

# 绘图参数全家桶
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

# 统一颜色方案
color_1 = "#F27970"
color_2 = "#BB9727"
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#05B9E2"

def freore_experiment():
    # 生成1000个正态分布随机数
    np.random.seed(42)
    random_numbers = np.random.normal(loc=50, scale=15, size=1000)
    random_numbers = np.clip(random_numbers, 0, 100).astype(int)
    
    # 绘制原始数据分布
    fig, ax1 = plt.subplots()
    
    counts, bins, patches = ax1.hist(random_numbers, bins=20, 
                                   facecolor='none',
                                   edgecolor=color_1, 
                                   alpha=0.99,
                                   hatch="-----")
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./freore_original_distribution.pdf', format='pdf')
    plt.show()

    freore = FreORE(10, 1000, 10, 9, b"secret_key", 8, 8)
    # 加密所有数据
    ciphertexts = [freore.data_encrypt(m) for m in random_numbers]
    
    # 对密文排序
    sorted_ciphertexts = freore.sort_encrypted(ciphertexts)
    
    # 统计密文频率
    freq = {}
    for ct in sorted_ciphertexts:
        freq[ct] = freq.get(ct, 0) + 1
    
    # 绘制密文分布
    fig, ax1 = plt.subplots()
    
    ax1.bar(range(len(freq)), freq.values(), 
           facecolor='none',
           edgecolor=color_2,
           hatch="/////",
           alpha=0.99)
    ax1.set_xlabel('Ciphertext (Sorted Order)')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks([])
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./freore_ciphertext_distribution.pdf', format='pdf')
    plt.show()


def experiment():
    # 生成1000个服从标准差为15，均值为50的正态分布随机数
    np.random.seed(42)
    random_numbers = np.random.normal(loc=50, scale=15, size=1000).astype(int)
    random_numbers = np.clip(random_numbers, 0, 100)  # 限制在0-100范围内

    # 绘制频率-数值分布图
    fig, ax1 = plt.subplots()
    
    ax1.hist(random_numbers, bins=20, 
             facecolor='none',
             edgecolor=color_1,
             hatch="-----",
             alpha=0.99)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./encodeore_original_distribution.pdf', format='pdf')
    plt.show()

    # # 初始化EncodeORE实例
    ore = EncodeORE(l1=8, l2=4)

    # 对随机数进行加密
    ciphertexts = [ore.encrypt(m) for m in random_numbers]

    # 对密文数组排序
    sorted_ciphertexts = sort_encrypted_data(ciphertexts, ore.compare)

    # 统计密文频率
    ciphertext_frequencies = {}
    for ct in sorted_ciphertexts:
        ct_tuple = tuple(ct)  # 将列表转换为元组以便作为字典键
        if ct_tuple in ciphertext_frequencies:
            ciphertext_frequencies[ct_tuple] += 1
        else:
            ciphertext_frequencies[ct_tuple] = 1

    # 绘制密文频率-密文值分布图
    fig, ax1 = plt.subplots()
    
    ax1.bar(range(len(ciphertext_frequencies)), 
           list(ciphertext_frequencies.values()), 
           facecolor='none',
           edgecolor=color_2,
           hatch="/////",
           alpha=0.99)
    ax1.set_xlabel('Ciphertext Index')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks([])
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./encodeore_ciphertext_distribution.pdf', format='pdf')
    plt.show()


def hybrid_experiment():
    # 生成正态分布数据
    np.random.seed(42)
    random_numbers = np.random.normal(loc=1000, scale=500, size=1000)
    random_numbers = np.clip(random_numbers, 0, 2000).astype(int)
    
    # 绘制原始数据分布
    fig, ax1 = plt.subplots()
    
    counts, bins, patches = ax1.hist(random_numbers, bins=20, 
                                   facecolor='none',
                                   edgecolor=color_1,
                                   hatch="-----",
                                   alpha=0.99)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./hybridore_original_distribution.pdf', format='pdf')
    plt.show()
    
    # # 初始化HybridORE
    lw_key = b"lewis_wu_secret_key"
    clww_key = b"clww_secret_key"
    hybrid_ore = HybridORE(lw_key, clww_key, l1=4, l2=4)
    
    # 加密数据并确保密文可哈希
    ciphertexts = []
    for m in random_numbers:
        ct_range, ct_value = hybrid_ore.encrypt(m)
        # 将各部分转换为可哈希的元组
        ct_range = (ct_range[0], tuple(ct_range[1]))  # 转换right列表为元组
        ct_value = tuple(ct_value)
        ciphertexts.append((ct_range, ct_value))
    
    # 对密文排序
    sorted_ciphertexts = sorted(ciphertexts, key=cmp_to_key(hybrid_ore.compare))
    
    # 统计频率
    freq = {}
    for ct in sorted_ciphertexts:
        if ct in freq:
            freq[ct] += 1
        else:
            freq[ct] = 1
    
    # 绘制密文频率分布
    fig, ax1 = plt.subplots()
    
    ax1.bar(range(len(freq)), list(freq.values()), 
           facecolor='none',
           edgecolor=color_2,
           hatch="/////",
           alpha=0.99)
    ax1.set_xlabel('Ciphertext (Sorted Order)')
    ax1.set_ylabel('Frequency')
    ax1.set_xticks([])
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./hybridore_ciphertext_distribution.pdf', format='pdf')
    plt.show()


def experiment_blockope():
    # 生成1000个服从正态分布的明文
    np.random.seed(42)
    plaintexts = np.random.normal(loc=50, scale=15, size=100).astype(int)
    plaintexts = np.clip(plaintexts, 0, 100)
    
    # 显示明文分布
    fig, ax1 = plt.subplots()
    
    ax1.hist(plaintexts, bins=20, 
             facecolor='none',
             edgecolor=color_1,
             hatch="-----",
             alpha=0.99)
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Frequency')
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./blockope_plaintext_distribution.pdf', format='pdf')
    plt.show()
    
    # # 初始化BlockOPE
    ope = BlockOPE()
    
    # 加密所有明文
    ciphertexts = [ope.encrypt(m) for m in plaintexts]
    
    # 绘制密文值分布
    fig, ax1 = plt.subplots()
    
    ax1.hist(ciphertexts, bins=20, 
             facecolor='none',
             edgecolor=color_2,
             hatch="/////",
             alpha=0.99)
    ax1.set_xlabel('Ciphertext Value')
    ax1.set_ylabel('Frequency')
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./blockope_ciphertext_distribution.pdf', format='pdf')
    plt.show()
    
    # 绘制排序后的密文频率分布
    sorted_ciphertexts = sorted(ciphertexts)
    freq = {}
    for ct in sorted_ciphertexts:
        freq[ct] = freq.get(ct, 0) + 1
    
    fig, ax1 = plt.subplots()
    
    ax1.bar(list(freq.keys()), list(freq.values()), 
           facecolor='none',
           edgecolor=color_3,
           hatch="|||||",
           alpha=0.99)
    ax1.set_xlabel('Ciphertext (Sorted)')
    ax1.set_ylabel('Frequency')
    
    plt.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)
    plt.tight_layout()
    plt.savefig('./blockope_sorted_frequency.pdf', format='pdf')
    plt.show()

# 运行实验
if __name__ == "__main__":
    experiment()
    freore_experiment()
    #hybrid_experiment()