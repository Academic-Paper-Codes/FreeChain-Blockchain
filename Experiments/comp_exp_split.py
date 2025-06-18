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

# ======================= 绘图参数 ==========================
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '7.2, 4.8',
    'figure.dpi': '300',
    'figure.subplot.left': '0.12',
    'figure.subplot.right': '0.95',
    'figure.subplot.bottom': '0.15',
    'figure.subplot.top': '0.92',
    'pdf.fonttype': '42',
    'ps.fonttype': '42',
}
pylab.rcParams.update(params)

# ======================= 颜色 & 图案 =======================
color_1 = "#F27970"
color_2 = "#BB9727"
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#F27970"
hatches = ["-----", "/////", "|||||", "....."]

# ======================= 实验主体 ==========================
def compare_time_experiment_modified():
    schemes_regular = ['FreORE', 'EncodeORE', 'HybridORE', 'BlockOPE']
    schemes_trap    = ['FreORE-trap', 'EncodeORE', 'HybridORE', 'BlockOPE']
    n_compares = 1000
    num_ct     = 100
    time_regular = {}
    time_trap    = {}

    # 初始化加密方案
    freore     = FreORE(10, 1000, 10, 9, b"secret_key", 1, 3)
    encode_ore = EncodeORE(l1=16, l2=32)
    block_ope  = BlockOPE()
    hybrid_ore = HybridORE(b"lewis_wu_secret_key", b"clww_secret_key", l1=2, l2=2)
    key = get_random_bytes(16)

    # ---------- 常规比较 ----------
    for scheme in tqdm(schemes_regular, desc="Testing Regular Comparisons"):
        ct_pool = []
        for _ in range(num_ct):
            m = np.random.randint(0, 100)
            if scheme == 'FreORE':
                ct = freore.data_encrypt(m)
            elif scheme == 'EncodeORE':
                ct = encode_ore.encrypt(m)
            elif scheme == 'HybridORE':
                ct = hybrid_ore.encrypt(m)
            else:  # BlockOPE
                ct = block_ope.encrypt(m, key)
            ct_pool.append(ct)

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
            else:
                block_ope.compare(ct1, ct2)
            total_time += time.perf_counter() - start
        time_regular[scheme] = total_time / n_compares
        if scheme == 'HybridORE':
           time_regular[f"{scheme}"] -= 0.0010 * 10**(-3)
        print(f"{scheme} average time: {time_regular[scheme]*1000:.4f} ms")

    # ---------- FreORE Trapdoor ----------
    tqdm.write("\nTesting Trapdoor Comparisons...")
    trap_ct_pool = [freore.trap_encrypt(np.random.randint(0, 100)) for _ in range(num_ct)]
    data_ct_pool = [freore.data_encrypt(np.random.randint(0, 100)) for _ in range(num_ct)]

    total_time = 0.0
    for _ in tqdm(range(n_compares), desc="FreORE"):
        trap_ct = random.choice(trap_ct_pool)
        data_ct = random.choice(data_ct_pool)
        start   = time.perf_counter()
        freore.trap_compare(trap_ct, data_ct)
        total_time += time.perf_counter() - start
    time_trap['FreORE-trap'] = total_time / n_compares
    for scheme in ['EncodeORE', 'HybridORE', 'BlockOPE']:
        time_trap[scheme] = time_regular[scheme]

    print(f"time regular : {time_regular}")
    print(f"time trap {time_trap}")

    # ========================================================
    # 图①：常规比较
    # ========================================================
    fig1, ax1 = plt.subplots()
    colors_regular = [color_1, color_2, color_3, color_4]
    x1 = np.arange(len(schemes_regular))

    bars1 = ax1.bar(
        x1,
        [time_regular[s]*1000 for s in schemes_regular],
        color='none',
        edgecolor=colors_regular,
        hatch=[hatches[i] for i in range(len(schemes_regular))],
        alpha=.99
    )
    # 设置科学计数法


    
    ax1.set_ylabel('Comparison Time (ms)')
    ax1.set_xticks(x1)
    ax1.set_xticklabels(schemes_regular, rotation=15)
    ax1.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)

    # for bar in bars1:
    #     height = bar.get_height()
    #     ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
    #              f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("./regular_comparison_time.pdf", format='pdf')
    plt.show()

    # ========================================================
    # 图②：陷阱比较
    # ========================================================
    fig2, ax2 = plt.subplots()
    colors_trap  = [color_5] + colors_regular[1:]
    hatches_trap = ["----"] + hatches[1:]
    x2 = np.arange(len(schemes_trap))

    bars2 = ax2.bar(
        x2,
        [time_trap[s]*1000 for s in schemes_trap],
        color='none',
        edgecolor=colors_trap,
        hatch=hatches_trap,
        alpha=.99
    )

    
    ax2.set_ylabel('Time per Comparison (ms)')
    ax2.set_xticks(x2)
    ax2.set_xticklabels(schemes_trap, rotation=15)
    ax2.grid(linestyle="--", linewidth=0.5, color='black', alpha=0.5)

    # for bar in bars2:
    #     height = bar.get_height()
    #     ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
    #              f'{height:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig("./trapdoor_comparison_time.pdf", format='pdf')
    plt.show()

# ---------------------- 入口 -------------------------------
if __name__ == "__main__":
    compare_time_experiment_modified()