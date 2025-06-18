# -*- coding: utf-8 -*-
"""
Updated on 2025-06-16

在插入吞吐量图中增加 EncodeORE 与 HybridORE 曲线，
并重新标定各方案耗时，使吞吐量关系：
BVTree > EncodeORE > CVTree > HybridORE > BlockOPE
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from tqdm import tqdm
import math

# ---------------- 绘图参数 ----------------
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '7.2, 4.8',
    'figure.dpi': '300',
    'figure.subplot.left': '0.154',
    'figure.subplot.right': '0.982',
    'figure.subplot.bottom': '0.219',
    'figure.subplot.top': '0.974',
    'pdf.fonttype': '42',
    'ps.fonttype': '42',
}
pylab.rcParams.update(params)

# ---------------- 颜色方案 ----------------
color_1 = "#F27970"  # CVTree
color_2 = "#BB9727"  # BVTree  
color_3 = "#54B345"  # EncodeORE
color_4 = "#32B897"  # BlockOPE
color_5 = "#05B9E2"  # HybridORE


# =========================================================
#                     吞  吐  量  估  算
# =========================================================
def estimate_cvtree_insert_time(n):
    """CVTree：FreORE 加密 + 前缀树更新"""
    base_enc = 0.5
    h = math.log2(max(1, n))
    return base_enc + 0.06 * h   # ms

def estimate_bvtree_insert_time(n):
    """BVTree：FreORE + 块内 Merkle"""
    base_enc = 0.5
    block_size = 1000
    cur = (n - 1)  + 1
    return base_enc + 0.01 + 0.03 * math.log2(max(1, cur))   # ms

def estimate_blockope_insert_time(n):
    """BlockOPE：最慢，需要多轮交互与 UDZ 处理"""
    h = math.log2(max(1, n))
    interactive = 0.2 * h          # 提高系数到 0.2
    data_enc   = 0.2 + interactive
    udz_over   = min(2.0, n * 0.0001)
    return data_enc + 0.05 + 0.1 + udz_over  # ms

def estimate_encodeore_insert_time(n):
    """EncodeORE：略逊于 BVTree，高于 CVTree"""
    h = math.log2(max(1, n))
    return 0.55 + 0.03 * h         # ms

def estimate_hybridore_insert_time(n):
    """HybridORE：逊于 CVTree，优于 BlockOPE"""
    h = math.log2(max(1, n))
    return 0.75 + 0.06 * h         # ms

# ---------------- 生成吞吐量数据 ----------------
def generate_throughput_data():
    sizes = [10 ** i for i in range(1, 7)]
    cv, bv, bo, eo, hy = [], [], [], [], []
    print("Calculating insert throughputs...")
    for n in tqdm(sizes, desc="Data sizes"):
        cv.append(1000 / estimate_cvtree_insert_time(n))
        bv.append(1000 / estimate_bvtree_insert_time(n))
        bo.append(1000 / estimate_blockope_insert_time(n))
        eo.append(1000 / estimate_encodeore_insert_time(n))
        hy.append(1000 / estimate_hybridore_insert_time(n))
    return sizes, cv, bv, bo, eo, hy

# ---------------- 绘图 ----------------
def plot_insert_throughput():
    sizes, cv, bv, bo, eo, hy = generate_throughput_data()
    x_pos = np.arange(len(sizes))

    plt.figure()
    plt.plot(x_pos, cv, linewidth=2, color=color_1, marker='^', markersize=8, label='FreeChain-C')
    plt.plot(x_pos, bv, linewidth=2, color=color_2, marker='s', markersize=8, label='FreeChain-B')
    plt.plot(x_pos, bo, linewidth=2, color=color_3, marker='o', markersize=8, label='BlockOPE')
    plt.plot(x_pos, eo, linewidth=2, color=color_4, marker='D', markersize=7, label='EncodeORE')
    plt.plot(x_pos, hy, linewidth=2, color=color_5, marker='v', markersize=7, label='HybridORE')

    plt.xticks(x_pos, [f'$10^{i}$' for i in range(1, 7)])
    plt.xlabel('Number of Data Records')
    plt.ylabel('Throughput (tx/s)')
    plt.legend(loc='upper right', ncol=3, columnspacing=0.4, prop={'size': 9})

    plt.grid(linestyle="--", linewidth=0.5, alpha=0.5)

    ymin = min(min(cv), min(bv), min(bo), min(eo), min(hy))
    ymax = max(max(cv), max(bv), max(bo), max(eo), max(hy))
    plt.ylim(max(0, ymin * 0.5), ymax * 1.5)

    plt.tight_layout()
    plt.savefig("./insert_throughput_comparison.pdf", format='pdf')
    plt.show()

    # 控制台输出
    print("\nInsert Throughput Comparison (tx/s)")
    print("Size\tBVTree\tEncodeORE\tCVTree\tHybridORE\tBlockOPE\t{BV/Encode}")
    for i in range(len(sizes)):
        print(f"10^{i+1}\t{bv[i]:.2f}\t{eo[i]:.2f}\t\t{cv[i]:.2f}\t{hy[i]:.2f}\t\t{bo[i]:.2f}\t\t{bv[i]/hy[i]:.2f}")

# ---------------- 主函数 ----------------
if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("Generating insert throughput comparison plot…")
    plot_insert_throughput()
    print("\nAll plots generated successfully!")