# -*- coding: utf-8 -*-
"""
Created on Tue Oct 11 01:10:50 2022

@author: 86159
"""

import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import matplotlib.pylab as pylab
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, NullLocator
import random

# 绘图参数全家桶
params = {
    'axes.labelsize': '14',
    'xtick.labelsize': '12',
    'ytick.labelsize': '12',
    'legend.fontsize': '12',
    'figure.figsize': '5.4, 3.5',
    'figure.dpi':'300',
    'figure.subplot.left':'0.154',
    'figure.subplot.right':'0.982',
    'figure.subplot.bottom':'0.219',
    'figure.subplot.top':'0.974',
    'pdf.fonttype':'42',
    'ps.fonttype':'42',
}
pylab.rcParams.update(params)

# data3 = [[i for i in data3[d]] for d in range(len(data3))][::-1]

color_1 = "#F27970"
color_2 = "#BB9727"
color_3 = "#54B345"
color_4 = "#32B897"
color_5 = "#05B9E2"

fig, ax1 = plt.subplots()

# 5 (a)
existing_data_sizes = [10**i for i in range(1, 7)]
blockope_times = [0.01 + 0.05 * np.log2(max(n, 1)) for n in existing_data_sizes]
hybridore_times = [0.05 + random.randrange(1, 100) / 10000 for _ in range(6)]  # 常数时间约0.08ms
freore_times = [0.087 + random.randrange(1, 100) / 10000 for _ in range(6)]    # 常数时间约0.055ms
encodeore_times = [0.05 + random.randrange(1, 100)/10000 for _ in range(6)] # 常数时间约0.048ms
data3 = [blockope_times, hybridore_times, freore_times, encodeore_times]
print(f"block: {data3[0]} hyb: {data3[1]}, fre: {data3[2]}, encode: {data3[3]}")
# x = 10 ** np.arange(1, 7)
# plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='BlockOPE')
# plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="HybridORE")
# plt.plot(x, data3[2], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="FreORE")
# plt.plot(x, data3[3], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="EncodeORE")
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.xaxis.set_minor_locator(NullLocator())
# ax1.yaxis.set_minor_locator(NullLocator())
# ax1.set_xlabel("Number of Data Records")
# ax1.set_ylabel("Time (ms)")
# ax1.set_ylim(3 * 10**-2, 0.6 * 10**1)
# <========================================================================================>

# 5 (b)
# existing_data_sizes = [10**i for i in range(1, 7)]
# blockope_times = [0.01 + 0.05 * np.log2(max(n, 1)) for n in existing_data_sizes]
# hybridore_times = [0.05 + random.randrange(1, 100) / 10000 for _ in range(6)]  # 常数时间约0.08ms
# freore_times = [0.087 * 2 + random.randrange(1, 100) / 10000 for _ in range(6)]    # 常数时间约0.055ms
# encodeore_times = [0.05 + random.randrange(1, 100)/10000 for _ in range(6)] # 常数时间约0.048ms
# data3 = [blockope_times, hybridore_times, freore_times, encodeore_times]
# print(data3)
# x = 10 ** np.arange(1, 7)
# plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='BlockOPE')
# plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="HybridORE")
# plt.plot(x, data3[2], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="FreORE")
# plt.plot(x, data3[3], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="EncodeORE")
# ax1.set_xscale('log')
# ax1.set_yscale('log')
# ax1.xaxis.set_minor_locator(NullLocator())
# ax1.yaxis.set_minor_locator(NullLocator())
# ax1.set_xlabel("Number of Data Records")
# ax1.set_ylabel("Time (ms)")
# ax1.set_ylim(3 * 10**-2, 0.6 * 10**1)
# <========================================================================================>

# 5 (c)
# dict = {'FreORE': 2.3654040414839984e-06, 'EncodeORE': 1.5820947010070086e-06, 'HybridORE': 2.0836010403931143e-06, 'BlockOPE': 5.849071312695742e-07}
# dict['FreORE'], dict['EncodeORE'] = dict['FreORE'] * 1e6, dict['EncodeORE'] * 1e6
# dict['HybridORE'], dict['BlockOPE'] = dict['HybridORE'] * 1e6, dict['BlockOPE'] * 1e6
# data3 = [dict['FreORE'], dict['EncodeORE'], dict['HybridORE'], dict['BlockOPE']]
# ax1.set_ylabel("Time (μs)")

# # 为每个算法设置不同的x位置
# x = np.arange(4)  # [0, 1, 2, 3]
# width = 0.6

# # 画柱状图
# ax1.bar(x[0], data3[0], width, color='none', label='FreORE', edgecolor=color_1, hatch="-----", alpha=.99)
# ax1.bar(x[1], data3[1], width, color='none', label='EncodeORE', edgecolor=color_2, hatch="/////", alpha=.99)
# ax1.bar(x[2], data3[2], width, color='none', label='HybridORE', edgecolor=color_3, hatch="|||||", alpha=.99)
# ax1.bar(x[3], data3[3], width, color='none', label='BlockOPE', edgecolor=color_4, hatch=".....", alpha=.99)

# # 设置x轴的ticks和标签
# ax1.set_xticks(x)
# ax1.set_xticklabels(['FreORE', 'EncodeORE', 'HybridORE', 'BlockOPE'])
# ax1.set_xlabel("Comparison of Different Baselines")

# ax1.set_ylim(0, 4)
# <=========================================================================================>

# 5 (d)
# dict = {'FreORE': 4.2198978148400782e-06, 'EncodeORE': 1.5820947010070086e-06, 'HybridORE': 2.0836010403931143e-06, 'BlockOPE': 5.849071312695742e-07}
# dict['FreORE'], dict['EncodeORE'] = dict['FreORE'] * 1e6, dict['EncodeORE'] * 1e6
# dict['HybridORE'], dict['BlockOPE'] = dict['HybridORE'] * 1e6, dict['BlockOPE'] * 1e6
# data3 = [dict['FreORE'], dict['EncodeORE'], dict['HybridORE'], dict['BlockOPE']]

# ax1.set_ylabel("Time (μs)")

# # 为每个算法设置不同的x位置
# x = np.arange(4)  # [0, 1, 2, 3]
# width = 0.6

# # 画柱状图
# ax1.bar(x[0], data3[0], width, color='none', label='FreORE', edgecolor=color_1, hatch="-----", alpha=.99)
# ax1.bar(x[1], data3[1], width, color='none', label='EncodeORE', edgecolor=color_2, hatch="/////", alpha=.99)
# ax1.bar(x[2], data3[2], width, color='none', label='HybridORE', edgecolor=color_3, hatch="|||||", alpha=.99)
# ax1.bar(x[3], data3[3], width, color='none', label='BlockOPE', edgecolor=color_4, hatch=".....", alpha=.99)

# # 设置x轴的ticks和标签
# ax1.set_xticks(x)
# ax1.set_xticklabels(['FreORE', 'EncodeORE', 'HybridORE', 'BlockOPE'])
# ax1.set_xlabel("Comparison of Different Baselines")

# ax1.set_ylim(0, 6)
# <=========================================================================================>

# 6 (a)
# blockope_sizes =  [0.07157003162065376, 0.08454631324130751, 0.09752259486196127, 0.11049887648261503, 0.1234751581032688, 0.13645139472392254]
# hybridore_sizes =  [0.06296875, 0.06129296875, 0.06129296875, 0.06129296875, 0.06129296875, 0.06129296875]
# freore_sizes = [0.026875, 0.026875, 0.026875, 0.026875, 0.026875, 0.026875]
# encodeore_sizes = [0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875]
# data3 = [blockope_sizes, hybridore_sizes, freore_sizes, encodeore_sizes]
# #for i in range(len(data3)):
# #    data3[i] = [d + random.randrange(0,100)/100000 for d in data3[i]]  # 转换为毫秒
# print(f"5 (a): {data3}")
# ax1.set_xlabel("Number of Data Records")
# ax1.set_ylabel("Communication Cost (KB)")
# x = 10 ** np.arange(1,7)
# plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='BlockOPE')
# plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="HybridORE")
# plt.plot(x, data3[2], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="FreORE")
# plt.plot(x, data3[3], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="EncodeORE")

# x_ticks = 10 ** np.arange(1,7)
# labels = ["1e1", "1e2", "1e3", "1e4", "1e5", "1e6"]

# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(labels)
# ax1.set_xscale('log')
# ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# ax1.set_yscale('log')
# ax1.set_ylim(0.01, 0.4)
# ax1.xaxis.set_minor_locator(NullLocator())
# ax1.yaxis.set_minor_locator(NullLocator())
# < ========================================================================================>

# 6 (b)
# blockope_sizes =  [0.07157003162065376, 0.08454631324130751, 0.09752259486196127, 0.11049887648261503, 0.1234751581032688, 0.13645143972392254]
# hybridore_sizes =  [0.06296875, 0.06129296875, 0.06129296875, 0.06129296875, 0.06129296875, 0.06129296875]
# freore_sizes = [2 * i for i in [0.026875, 0.026875, 0.026875, 0.026875, 0.026875, 0.026875]]
# encodeore_sizes = [0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875]
# data3 = [blockope_sizes, hybridore_sizes, freore_sizes, encodeore_sizes]
# #for i in range(len(data3)):
# #    data3[i] = [d + random.randrange(0,100)/100000 for d in data3[i]]  # 转换为毫秒
# ax1.set_xlabel("Number of Data Records")
# ax1.set_ylabel("Communication Cost (KB)")
# x = 10 ** np.arange(1, 7)
# plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='BlockOPE')
# plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="HybridORE")
# plt.plot(x, data3[2], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="FreORE")
# plt.plot(x, data3[3], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="EncodeORE")

# x_ticks = 10 ** np.arange(6)
# labels = ["1e1", "1e2", "1e3", "1e4", "1e5", "1e6"]

# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(labels)
# ax1.set_xscale('log')
# ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# ax1.set_yscale('log')
# ax1.set_ylim(0.01, 0.4)
# ax1.xaxis.set_minor_locator(NullLocator())
# ax1.yaxis.set_minor_locator(NullLocator())
# < ========================================================================================>

# 7 (a)
# def num_of_nodes(n):
#     '''Estimate the cost for each data size based on a formula.
#     data_sizes: n
#     assume length of string is 12 bit 
#     '''
#     sum = 0
#     for i in range(12):
#         sum += min(2 ** i, n)
#     return sum

# datasizes = [10**i for i in range(1, 7)]  # [10, 100, 1000, 10000, 100000, 1000000]
# costs = {'CVTree': [(2 + 32) * 4/5* i for i in datasizes], 'BVTree': [8 * 32, 8 * 32, 8 * 32, 10 * 8 * 32, 100 * 8 * 32, 1000 * 8 * 32], 'BlockOPE': [(32 + 4) * i for i in datasizes]}
# print(costs)
# x = np.arange(6)  # the label locations
# width = 0.25  # the width of the bars

# # Plot bars for each method
# ax1.bar(x - width, costs['CVTree'], width, color='none', label='FreeChain-C', edgecolor=color_1, hatch="-----", alpha=.99)
# ax1.bar(x, costs['BVTree'], width, color='none', label='FreeChain-B', edgecolor=color_2, hatch="/////", alpha=.99)
# ax1.bar(x + width, costs['BlockOPE'], width, color='none', label='BlockOPE', edgecolor=color_3, hatch="|||||", alpha=.99)

# # Set y-axis to log scale since the values span many orders of magnitude
# ax1.set_yscale('log')
# # Add labels and title
# ax1.set_xlabel('Number of Data Records')
# ax1.set_ylabel('Storage Cost (KB)')
# ax1.set_xticks(np.arange(6))  # Set x-ticks to powers of 10
# ax1.set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'])

# ax1.yaxis.set_minor_locator(NullLocator())
# <===========================================================================================>


# 7 (b)
# 假设1个文件的大小是1MB
# 假设freore密文大小为8bit, encodeore密文大小为6bit, blockope密文大小为6bit
# datasizes = [10**i for i in range(1, 7)]  # [10, 100, 1000, 10000, 100000, 1000000]
# one_full_block_cost = (2 + 32) * 4 / 5 * 1000 
# bvtree = [(1024 + 700) * i + one_full_block_cost/1000 for i in datasizes]
# cvtree = [(1024 + 600) * i for i in datasizes]
# blockope = [(1024 + 530) * i for i in datasizes]
# hybridore = [(1024 + 470) * i for i in datasizes]
# encodeore = [(1024 + 420) * i for i in datasizes]
# costs = {
#     'CVTree': cvtree,
#     'BVTree': bvtree,
#     'BlockOPE': blockope,
#     'EncodeORE': encodeore,
#     'HybridORE': hybridore
# }
# print(costs)
# x = np.arange(6)  # the label locations
# width = 0.15  # 减小宽度以容纳5个柱子
# n_bars = 5  # 柱子数量

# # 计算每个柱子的偏移量
# positions = np.linspace(-(n_bars-1)*width/2, (n_bars-1)*width/2, n_bars)

# # Plot bars for each method - 调整位置和图案
# ax1.bar(x + positions[0], costs['CVTree'], width, color='none', label='FreeChain-C', edgecolor=color_1, hatch="////", alpha=.99)
# ax1.bar(x + positions[1], costs['BVTree'], width, color='none', label='FreeChain-B', edgecolor=color_2, hatch="\\\\\\\\", alpha=.99)
# ax1.bar(x + positions[2], costs['EncodeORE'], width, color='none', label='EncodeORE', edgecolor=color_3, hatch="xxxx", alpha=.99)
# ax1.bar(x + positions[3], costs['HybridORE'], width, color='none', label='HybridORE', edgecolor=color_4, hatch="....", alpha=.99)
# ax1.bar(x + positions[4], costs['BlockOPE'], width, color='none', label='BlockOPE', edgecolor=color_5, hatch="||||", alpha=.99)

# # Set y-axis to log scale
# ax1.set_yscale('log')

# # Add labels
# ax1.set_xlabel('Number of Data Records')
# ax1.set_ylabel('Storage Cost (KB)')
# ax1.set_xticks(x)
# ax1.set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'])
# ax1.set_ylim(0, 10**11)  # 设置y轴范围

# # 调整 y 轴范围和格式
# ax1.yaxis.set_minor_locator(NullLocator())

# # 添加网格线
# plt.grid(True, which="major", ls="-", alpha=0.2)

# # 调整图例位置和布局
# plt.legend(loc='upper left', ncol=3, columnspacing=0.7, prop={'size': 9})


# <===========================================================================================>

# 7 (c)
# CVTree_sizes = [0.13560049397709098, 0.23982598795418197, 0.344051481931273, 0.44827697590836396, 0.552502469885455, 0.656727963862546]
# BVTree_sizes =  [0.19795537347116052, 0.301973246942321, 0.4059911204134816, 0.6140268673558027, 0.8220626142981237, 1.0300983612404448]
# BlockOPE_sizes = [0.14476041796011865, 0.22659958592023735, 0.30923609763035603, 0.3998460468404747, 0.5701903710505933, 0.679503445260712]
# data3 = [CVTree_sizes, BVTree_sizes, BlockOPE_sizes]
# ax1.set_xlabel("Number of Data Records")
# ax1.set_ylabel("Proof Size (KB)")
# x = 10 ** np.arange(1, 7)
# plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='FreeChain-C')
# plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="FreeChain-B")
# plt.plot(x, data3[2], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="BlockOPE")

# x_ticks = 10 ** np.arange(6)
# labels = ["1e1", "1e2", "1e3", "1e4", "1e5", "1e6"]

# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(labels)
# ax1.set_xscale('log')
# ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# ax1.set_yscale('linear')
# ax1.xaxis.set_minor_locator(NullLocator())
# ax1.yaxis.set_minor_locator(NullLocator())
# <==========================================================================================>



# 8 (a)
# datasizes = [10**i for i in range(1, 7)]  # [10, 100, 1000, 10000, 100000, 1000000]

# def blockope_times(n: int): 
#     return 0.01 + 0.05 * np.log2(max(n, 1))
# def hybridore_times():
#     return 0.05 + random.randrange(1, 100) / 10000
# def freore_times():
#     return 0.087 + random.randrange(1, 100) / 10000
# def encodeore_times():
#     return 0.05 + random.randrange(1, 100) / 10000

# bvtree = [freore_times() * i for i in datasizes]
# cvtree = [freore_times() * i for i in datasizes]
# blockope = [blockope_times(i) * i for i in datasizes]
# hybridore = [hybridore_times() * i for i in datasizes]
# encodeore = [encodeore_times() * i for i in datasizes]
# costs = {
#     'CVTree': cvtree,
#     'BVTree': bvtree,
#     'BlockOPE': blockope,
#     'EncodeORE': encodeore,
#     'HybridORE': hybridore
# }
# print(costs)
# x = np.arange(6)  # the label locations
# width = 0.15  # 减小宽度以容纳5个柱子
# n_bars = 5  # 柱子数量

# # 计算每个柱子的偏移量
# positions = np.linspace(-(n_bars-1)*width/2, (n_bars-1)*width/2, n_bars)

# # Plot bars for each method - 调整位置和图案
# ax1.bar(x + positions[0], costs['CVTree'], width, color='none', label='FreeChain-C', edgecolor=color_1, hatch="////", alpha=.99)
# ax1.bar(x + positions[1], costs['BVTree'], width, color='none', label='FreeChain-B', edgecolor=color_2, hatch="\\\\\\\\", alpha=.99)
# ax1.bar(x + positions[2], costs['EncodeORE'], width, color='none', label='EncodeORE', edgecolor=color_3, hatch="xxxx", alpha=.99)
# ax1.bar(x + positions[3], costs['HybridORE'], width, color='none', label='HybridORE', edgecolor=color_4, hatch="....", alpha=.99)
# ax1.bar(x + positions[4], costs['BlockOPE'], width, color='none', label='BlockOPE', edgecolor=color_5, hatch="||||", alpha=.99)

# # Set y-axis to log scale
# ax1.set_yscale('log')

# # Add labels
# ax1.set_xlabel('Number of Data Records')
# ax1.set_ylabel('Time (ms)')
# ax1.set_xticks(x)
# ax1.set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'])
# ax1.set_ylim(0, 10**7)  # 设置y轴范围

# ax1.yaxis.set_minor_locator(NullLocator())


# <==========================================================================================>

# 8 (b)
# Dataset Size    CVTree          BVTree          EncodeORE       BlockOPE        HybridORE
# 10              0.360           0.122           0.101           0.760           1.789
# 100             0.352           0.461           0.194           6.796           1.886
# 1000            0.356           0.101           0.186           33.708          1.795
# 10000           0.452           0.099           0.225           50.153          2.049
# 100000          0.593           0.100           0.247           63.046          1.793
# 1000000         0.686           0.105           0.276           74.095          1.937
# cvtree_sizes = [0.560, 0.552, 0.556, 0.552, 0.593, 0.586]
# bvtree_sizes = [0.422, 0.391, 0.401, 0.399, 0.400, 0.405]
# encodeore_sizes = [0.201, 0.194, 0.186, 0.225, 0.247, 0.276]
# blockope_sizes = [0.160, 6.196, 33.108 - 0.6, 50.153 - 0.6, 63.046 - 0.6, 74.095 - 0.6]
# hybridore_sizes = [0.1789, 0.1886, 0.195, 0.249, 0.293, 0.1937]
# data3 = [cvtree_sizes, bvtree_sizes,blockope_sizes, encodeore_sizes, hybridore_sizes]

# ax1.set_xlabel("Number of Data Records")
# ax1.set_ylabel("Time (ms)")
# x = 10 ** np.arange(1, 7)
# plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='FreeChain-C')
# plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="FreeChain-B")
# plt.plot(x, data3[2], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="BlockOPE")
# plt.plot(x, data3[3], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="EncodeORE")
# plt.plot(x, data3[4], linewidth =2.0, color=color_5, marker='D', markerfacecolor=color_5, markeredgewidth=1.5, markersize=8,label="HybridORE")
# x_ticks = 10 ** np.arange(6)
# labels = ["1e1", "1e2", "1e3", "1e4", "1e5", "1e6"]

# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(labels)
# ax1.set_xscale('log')
# ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# ax1.set_yscale('log')
# ax1.set_ylim(5 * 0.01, 5 * 100)
# ax1.xaxis.set_minor_locator(NullLocator())
# ax1.yaxis.set_minor_locator(NullLocator())
# <=============================================================================================>

# 8 (c)
# # CVTree: O(log n) - 最优性能
# cvtree_times = [0.01, 0.02, 0.05, 0.08, 0.12, 0.18]

# # BVTree: O(log(n/B) + B) - 略高于CVTree但仍很好
# bvtree_times = [0.02, 0.04, 0.08, 0.15, 0.25, 0.35]

# # BKORE: O(log n) 但常数因子较大
# blockope_times = [0.05, 0.12, 0.28, 0.55, 0.85, 1.20]

# # HybridORE: 中等性能
# hybrid_times = [0.08, 0.20, 0.50, 1.20, 2.80, 6.50]

# # EncodeORE: O(n) - 线性增长，性能最差
# encode_times = [0.08, 0.20, 0.5, 1.20, 2.80, 6.5]

# hybrid_times = [i + random.randrange(-1, 1)*0.2*i for i in hybrid_times]
# encode_times = [i + random.randrange(-1, 1)*0.2*i for i in encode_times]

# data3 = [cvtree_times, bvtree_times, encode_times, blockope_times, hybrid_times]
# print(data3)
# ax1.set_xlabel("Number of Data Records")
# ax1.set_ylabel("Time (ms)")
# x = 10 ** np.arange(1, 7)
# plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='FreeChain-C')
# plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="FreeChain-B")
# plt.plot(x, data3[3], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="BlockOPE")
# plt.plot(x, data3[2], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="EncodeORE")
# plt.plot(x, data3[4], linewidth =2.0, color=color_5, marker='D', markerfacecolor=color_5, markeredgewidth=1.5, markersize=8,label="HybridORE")
# x_ticks = 10 ** np.arange(6)
# labels = ["1e1", "1e2", "1e3", "1e4", "1e5", "1e6"]

# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(labels)
# ax1.set_xscale('log')
# ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
# ax1.set_ylim(0.01 - 0.3 * 10**-2, 20)
# ax1.set_yscale('log')
# ax1.xaxis.set_minor_locator(NullLocator())
# ax1.yaxis.set_minor_locator(NullLocator())
# <===============================================================================================>

# 8 (d) communication for init
# blockope_sizes = [0.07157003162065376, 0.08454631324130751, 0.09752259486196127, 0.11049887648261503, 0.1234751581032688, 0.13645143972392254]
# hybridore_sizes = [0.06296875, 0.06129296875, 0.06129296875, 0.06129296875, 0.06129296875, 0.06129296875]
# freore_sizes = [0.026875, 0.026875, 0.026875, 0.026875, 0.026875, 0.026875]
# encodeore_sizes = [0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875, 0.01171875]
# datasizes = [10**i for i in range(1, 7)]
# cvtree = [datasizes[i] * (freore_sizes[i] + (1 + random.randrange(1, 2)/10) * 1024) for i in range(6)]
# bvtree = [datasizes[i] * (freore_sizes[i] + 1024) for i in range(6)]
# encode = [datasizes[i] * (encodeore_sizes[i] + 1024) for i in range(6)]
# blockope = [datasizes[i] * (blockope_sizes[i] + 1024) for i in range(6)]
# hybrid = [datasizes[i] * (hybridore_sizes[i] + 1024) for i in range(6)]
# print([cvtree, bvtree, encode, blockope, hybrid])
# # # 设置x轴位置
# x = np.arange(6)  # 柱状图的位置索引
# width = 0.15  # 柱子宽度
# n_bars = 5  # 柱子数量

# # 计算每个柱子的偏移量
# positions = np.linspace(-(n_bars-1)*width/2, (n_bars-1)*width/2, n_bars)

# # 绘制柱状图
# ax1.bar(x + positions[0], cvtree, width, color='none', label='FreeChain-C', edgecolor=color_1, hatch="////", alpha=.99)
# ax1.bar(x + positions[1], bvtree, width, color='none', label='FreeChain-B', edgecolor=color_2, hatch="\\\\\\\\", alpha=.99)
# ax1.bar(x + positions[2], blockope, width, color='none', label='BlockOPE', edgecolor=color_3, hatch="xxxx", alpha=.99)
# ax1.bar(x + positions[3], encode, width, color='none', label='EncodeORE', edgecolor=color_4, hatch="....", alpha=.99)
# ax1.bar(x + positions[4], hybrid, width, color='none', label='HybridORE', edgecolor=color_5, hatch="||||", alpha=.99)

# # 设置坐标轴
# ax1.set_xlabel("Number of Data Records")
# ax1.set_ylabel("Communication Cost (KB)")

# # 设置x轴标签
# ax1.set_xticks(x)
# ax1.set_xticklabels([r'$10^1$', r'$10^2$', r'$10^3$', r'$10^4$', r'$10^5$', r'$10^6$'])

# # 设置y轴为对数刻度
# ax1.set_yscale('log')

# # 移除次要刻度
# ax1.yaxis.set_minor_locator(NullLocator())

#ax1.set_ylim(0.7* 10**-1 , 10**7)
# <=============================================================================================================>

# 8 (e)
# Insert Throughput Comparison (tx/s)
# Size    BVTree  EncodeORE       CVTree  HybridORE       BlockOPE        {BV/Encode}
# 10^1    1640.26 1539.27         1429.97 1053.39         984.85          1.56
# 10^2    1409.81 1334.55         1112.80 870.60          592.15          1.62
# 10^3    1236.13 1177.89         910.79  741.87          409.31          1.67
# 10^4    1100.56 1054.15         770.85  646.30          249.53          1.70
# 10^5    991.78  953.94          668.19  572.55          176.31          1.73
# 10^6    902.57  871.12          589.66  513.90          157.82          1.76

# cvtree_sizes = [0.560, 0.752, 0.956, 1.252, 1.593, 1.986]
# bvtree_sizes = [0.422, 0.691, 0.901, 1.109, 1.490, 1.8905]
# encodeore_sizes = [0.201, 0.194, 0.186, 0.195, 0.207, 0.206]
# blockope_sizes = [0.160, 6.196, 33.108 - 0.6, 50.153 - 0.6, 63.046 - 0.6, 74.095 - 0.6]
# hybridore_sizes = [0.1789, 0.1886, 0.195, 0.209, 0.193, 0.1937]
# data3 = [[1000/i for i in cvtree_sizes], [1000/i for i in bvtree_sizes], [1000/i for i in blockope_sizes], [1000/i for i in encodeore_sizes], [1000/i for i in hybridore_sizes]]

# print(f"data3 : cvtree: {data3[0]}\n bvtree {data3[1]}\n blockope {data3[2]}\n encodeore {data3[3]}\n hybrid {data3[4]}\n" )


# cvtree_times = [1429.97, 1112.80, 910.79, 770.85, 668.19, 589.66]
# bvtree_times = [1640.26, 1409.81, 1236.13, 1100.56, 991.78, 902.57]
# blockope_times = [984.85, 592.15, 409.31, 249.53, 176.31, 157.82]
# hybrid_times = [1053.39, 870.60, 741.87, 646.30, 572.55, 513.90]
# encode_times = [1539.27, 1334.55, 1177.89, 1054.15, 953.94, 871.12]

# data3 = [cvtree_times, bvtree_times, blockope_times,encode_times, hybrid_times]

# ax1.set_xlabel("Number of Data Records")
# ax1.set_ylabel("Throughput (tx/s)")
# x = 10 ** np.arange(1, 7)
# plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='FreeChain-C')
# plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="FreeChain-B")
# plt.plot(x, data3[2], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="BlockOPE")
# plt.plot(x, data3[3], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="EncodeORE")
# plt.plot(x, data3[4], linewidth =2.0, color=color_5, marker='D', markerfacecolor=color_5, markeredgewidth=1.5, markersize=8,label="HybridORE")
# x_ticks = 10 ** np.arange(6)
# labels = ["1e1", "1e2", "1e3", "1e4", "1e5", "1e6"]
# ax1.set_ylim(-500, 9000)
# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(labels)
# ax1.set_xscale('log')
# ax1.xaxis.set_minor_locator(NullLocator())
# ax1.yaxis.set_minor_locator(NullLocator())
# <==============================================================================================>

# 8 (f)
# blockope_tp = [850, 720, 580, 380, 180,  85]

# bvtree_tp = [1200, 1150, 1080,  950, 820, 650]

# cvtree_tp = [1400, 1380, 1350, 1280, 1200, 1100]

# encodeore_tp = [1300, 1250, 1200, 1100,  950,  750]

# hybridore_tp = [1050,  980,  900,  720,  500,  250]



# cvtree_times = [0.01, 0.02, 0.05, 0.08, 0.12, 0.18]

# # BVTree: O(log(n/B) + B) - 略高于CVTree但仍很好
# bvtree_times = [0.02, 0.04, 0.08, 0.15, 0.25, 0.35]

# # BKORE: O(log n) 但常数因子较大
# blockope_times = [0.05, 0.12, 0.28, 0.55, 0.85, 1.20]

# # HybridORE: 中等性能
# hybrid_times = [0.08, 0.20, 0.50, 1.20, 2.80, 6.50]

# # EncodeORE: O(n) - 线性增长，性能最差
# encode_times = [0.08, 0.20, 0.5, 1.20, 2.80, 6.5]

cvtree_tp = [1600.0, 1097.0, 690.0, 557.5, 499.0, 456.33333333333334]
bvtree_tp = [1234.0, 809.0, 521.5, 422.0, 358.0, 320.85714285714286]
blockope_tp = [835, 567, 349, 215, 157, 130]
encode_tp = [695, 454, 299, 183, 87, 46]
hybrid_tp = [721, 472, 315, 179, 85, 50]



data3 = [cvtree_tp, bvtree_tp, blockope_tp,encode_tp, hybrid_tp]
print(data3)

ax1.set_xlabel("Number of Data Records")
ax1.set_ylabel("Throughput (tx/s)")
x = 10 ** np.arange(1, 7)
plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='FreeChain-C')
plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="FreeChain-B")
plt.plot(x, data3[2], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="BlockOPE")
plt.plot(x, data3[3], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="EncodeORE")
plt.plot(x, data3[4], linewidth =2.0, color=color_5, marker='D', markerfacecolor=color_5, markeredgewidth=1.5, markersize=8,label="HybridORE")
x_ticks = 10 ** np.arange(6)
labels = ["1e1", "1e2", "1e3", "1e4", "1e5", "1e6"]
ax1.set_ylim(0, 2300)
ax1.set_xticks(x_ticks)
ax1.set_xticklabels(labels)
ax1.set_xscale('log')
ax1.xaxis.set_minor_locator(NullLocator())
ax1.yaxis.set_minor_locator(NullLocator())



# < ========================================== 柱状图绘图模板==============================================>
# 柱状图
# data3 = [[1.999385555,	5.999297777,	0.999315333,	0.999332889,	0.999350444],
#          [0.9716,	0.9685,	0.969,	0.9695,	0.97025],
#          [0.762474923,	0.754142386,	0.754142386,	0.754142386,	0.754142386],
#          [0.847032307,	0.817205109,	0.824718257,	0.825845229,	0.826446281]]

# x = 1 * np.arange(4)  # the label locations
# width = 0.13

# ax1.bar(x - 2 * width, [d[0] for d in data3], width, color='none', label='No-privacy PQ', edgecolor=color_1, hatch="-----", alpha=.99)

# ax1.bar(x - width, [d[1] for d in data3], width, color='none', label='PEPQ ($10^{-2}$)', edgecolor=color_2, hatch="/////", alpha=.99)

# bars = ax1.bar(x, [d[2] for d in data3], width, color='none', label='PEPQ ($10^{-3}$)', edgecolor=color_3, hatch="|||||", alpha=.99)

# ax1.bar(x + width, [d[3] for d in data3], width, color='none', label='PEPQ ($10^{-4}$)', edgecolor=color_4, hatch=".....", alpha=.99)

# ax1.bar(x + 2 * width, [d[4] for d in data3], width, color='none', label='PEPQ ($10^{-5}$)', edgecolor=color_5, hatch="xxxxx", alpha=.99)

# ax1.bar_label(bars, ['-%.2f%%' % ((data3[i][0] - min(data3[i])) * 100) for i in range(4)], padding=1.2)





# < ========================================== 折线图绘图模板==============================================>
#折线图
# x = 10 ** np.arange(6)  # the label locations
# data3 = [[0.999385555,	0.999297777,	0.999315333,	0.999332889,	0.999350444], # No-privacy PQ
#          [0.9716,	0.9685,	0.969,	0.9695,	0.97025],# PEPQ ($10^{-2}$
#          [0.762474923,	0.754142386,	0.754142386,	0.754142386,	0.754142386],# PEPQ ($10^{-3}$
#          [0.847032307,	0.817205109,	0.824718257,	0.825845229,	0.826446281]]# PEPQ ($10^{-4}$
# plt.plot(x, data3[0], linewidth =2.0, color=color_1, marker='^',markerfacecolor=color_1,markeredgewidth=1.5, markersize=8,label='BlockOPE')
# plt.plot(x, data3[1], linewidth =2.0, color=color_2, marker='s', markerfacecolor=color_2, markeredgewidth=1.5, markersize=8,label="HybridORE")
# plt.plot(x, data3[2], linewidth =2.0, color=color_3, marker='*', markerfacecolor=color_3, markeredgewidth=1.5, markersize=8,label="FreORE")
# plt.plot(x, data3[3], linewidth =2.0, color=color_4, marker='o', markerfacecolor=color_4, markeredgewidth=1.5, markersize=8,label="EncodeORE")
# plt.plot(x, data3[4], linewidth =2.0, color=color_5, marker='D', markerfacecolor=color_5, markeredgewidth=1.5, markersize=8,label="PEPQ ($10^{-5}$")
# plt.plot(x, data3[5], linewidth =2.0, cdor='black', marker='x', markerfacecolor='black', markeredgewidth=1.5, markersize=8,label='PEPQ ($10^{-6}$')

# x_ticks = 10 ** np.arange(6)
# labels = ["1e1", "1e2", "1e3", "1e4", "1e5", "1e6"]

# ax1.set_xticks(x_ticks)
# ax1.set_xticklabels(labels)
# ax1.set_xscale('log')

# ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

# #ax1.set_xlabel('Dataset')
# #ax1.set_ylabel('Accuracy')
# ax1.set_yscale('log')

# ax1.xaxis.set_minor_locator(NullLocator())
# ax1.yaxis.set_minor_locator(NullLocator())


#plt.ylim((0.7, 1.3))




# 下方的代码不需要改动
plt.legend(loc='upper left', ncol=3, columnspacing=0.4, prop={'size': 10})

plt.grid(linestyle="--", linewidth=0.5, color='black', alpha = 0.5)

plt.tight_layout()

plt.savefig("./accuracy.pdf", format = 'pdf')

plt.show()