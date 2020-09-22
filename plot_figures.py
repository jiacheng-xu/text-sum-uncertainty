# -*- coding: utf-8 -*-


import json, math, seaborn
import statistics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

# del matplotlib.font_manager.weight_dict['roman']
matplotlib.font_manager._rebuild()
GLOBAL_FIGURE_WIDTH = 8
dpi = 800
# plt.rcParams["font.weight"] = "light"
plt.rcParams.update({'font.size': 14})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'DejaVu Sans'

import random


def read_json_file(fname):
    with open(fname, 'r') as fd:
        x = fd.read()
    data = json.loads(x)
    return data


datasets = ['cnndm', 'xsum']
import os

# dir_datadrive = '/content/drive/My Drive/prob_gpt_data/'
dir_datadrive = '/Users/user/Downloads/'
dir_datadrive = '/mnt/data0/jcxu/data/prob_gpt'
"""## Correlation of Copy/Gen & Prediction Entropy
Datasets: CNNDM, XSUM

Several axis here: Relative Position in Sentence; IsBigram/Trigram (identifies copy or generation), Prediction Entropy

Figure: X - Prediction *entropy*; Y - Number of example; Color/Label: is or is not Bigram
"""

# from google.colab import drive
# drive.mount('/content/drive')

# {}_entropy.json
# [t: current time step, l: total length of the sequence, ent, max_prob, tokens[t], bigran, trigram]


# Density: x: entropy, two plots: is bigram or not
FIG_SIZE_x = GLOBAL_FIGURE_WIDTH
FIG_SIZE_y = 3
ysize_fig1 = 4
ysize_figure2 = 4
ysize_figure3 = 4

debug = True
debug = False


def _read_data(dir, data_name, SEPS):
    data = read_json_file(os.path.join(dir, f"{data_name}_entropy.json"))
    data_for_panda = []
    bigram_entropies, not_bigram_entropies = [], []
    random.shuffle(data)
    if debug:
        data = data[:1000]
    for d in data:
        if d[1] <= 1:
            continue
        relative_position = int(math.floor(d[0] * SEPS / (d[1]))) / SEPS
        new_data = d[2:] + [relative_position]
        entropy = new_data[0]
        max_prob = new_data[1]
        # entropy  = max_prob
        if new_data[3]:
            bigram_entropies.append(entropy)
        else:
            not_bigram_entropies.append(entropy)
        data_for_panda.append(new_data)

    print("Finish reading data")

    keys = ['Entropy', 'Top1 Prob', 'token', 'Bigram ', 'InTrigramOfDocument', 'Relative Position']
    df = pd.DataFrame(data_for_panda, columns=keys)
    return df, bigram_entropies, not_bigram_entropies


def normalize_figure1(bigram: List, not_bigram: List, cnt=10000):
    if len(bigram) + len(not_bigram) > cnt:
        rate = len(bigram) + len(not_bigram)
        ratio = cnt / rate
        random.shuffle(bigram)
        bigram = bigram[: int(len(bigram) * ratio)]
        random.shuffle(not_bigram)
        not_bigram = not_bigram[: int(len(not_bigram) * ratio)]
    print(f"normalized len: {len(bigram)} + {len(not_bigram)}")
    return bigram, not_bigram


def draw_x_entropy_y_bigram_count(dir, SEPS=10, FIG_SIZE_x=10, FIG_SIZE_y=3, bins_num=18,
                                  cnndm_spec_name='cnndm', xsum_spec_name='xsum'):
    ExistingBigram = "Existing Bigrams"
    NovelBigram = "Novel Bigrams"
    linewidth = 1.5
    _, cnndm_bigram_entropies, cnndm_not_bigram_entropies = _read_data(dir, cnndm_spec_name, SEPS)
    cnndm_bigram_entropies, cnndm_not_bigram_entropies = normalize_figure1(cnndm_bigram_entropies,
                                                                           cnndm_not_bigram_entropies)
    axes1 = plt.subplot(211)
    # axes = fig.add_axes([0., 0.3, 0.84, 0.66])
    # sns.distplot(x=ykey, data=df, hist=False, rug=True)
    # axes = sns.distplot(bigram_entropies,rug=True)
    # axes = sns.kdeplot(bigram_entropies, shade=True,cumulative=True)
    # axes = sns.distplot(bigram_entropies,rug=True)
    data = 'CNN/DM'
    max_lim = 6
    color = sns.color_palette("coolwarm", 7)
    axes1 = sns.distplot(cnndm_bigram_entropies, bins=bins_num,
                         hist_kws={'range': [0, max_lim]},
                         hist=True,
                         kde=False,
                         color=color[0],
                         label=f"{ExistingBigram}"
                         )
    # axes1.set_title("CNN/DM")
    axes1.set_ylabel('CNN/DM')

    axes1 = sns.distplot(cnndm_not_bigram_entropies, bins=bins_num, hist_kws={'range': [0, max_lim]},
                         hist=True,
                         kde=False,
                         color=color[-1],
                         label=f"{NovelBigram}"
                         )
    plt.axvline(statistics.median(cnndm_bigram_entropies), color=color[0], linestyle='dashed', linewidth=linewidth)
    plt.axvline(statistics.median(cnndm_not_bigram_entropies), color=color[-1], linestyle='dashed', linewidth=linewidth)
    # axes.legend(prop={'size': 10})
    axes1.legend(frameon=False)
    # print(f"{statistics.median(cnndm_bigram_entropies), statistics.mean(cnndm_not_bigram_entropies),}")
    print(f"{statistics.median(cnndm_bigram_entropies), statistics.median(cnndm_not_bigram_entropies),}")

    plt.setp(axes1.get_xticklabels(), visible=False)
    # axes = sns.distplot(not_bigram_entropies,rug=True)
    # axes1.set_title('CNN\DM',loc='left')
    # axes1.set_xlabel('Entropy')

    _, xsum_bigram_entropies, xsum_not_bigram_entropies = _read_data(dir, xsum_spec_name, SEPS)
    xsum_bigram_entropies, xsum_not_bigram_entropies = normalize_figure1(xsum_bigram_entropies,
                                                                         xsum_not_bigram_entropies)
    # axes = fig.add_axes([0., 0.3, 0.84, 0.66])

    with sns.color_palette("Set2"):
        axes2 = plt.subplot(212, sharex=axes1)
        # axes2.set_title("XSum",loc='left')
        axes2.set_ylabel('XSum')
        data = 'XSum'
        axes2 = sns.distplot(xsum_bigram_entropies, bins=bins_num,
                             hist_kws={'range': [0, max_lim], }, hist=True, kde=False,
                             # label=f"{ExistingBigram}",
                             color=color[0]
                             )
        axes2 = sns.distplot(xsum_not_bigram_entropies, bins=bins_num,
                             hist_kws={'range': [0, max_lim], }, hist=True, kde=False,
                             # label=f"{NovelBigram}",
                             color=color[-1]
                             )

        plt.axvline(statistics.median(xsum_bigram_entropies), color=color[0], linestyle='dashed', linewidth=linewidth)
        plt.axvline(statistics.median(xsum_not_bigram_entropies), color=color[-1], linestyle='dashed',
                    linewidth=linewidth)
    # axes.legend(prop={'size': 10})
    # axes2.legend()
    # axes = sns.distplot(not_bigram_entropies,rug=True)
    # axes2.set_title('XSum')
    axes2.set_xlabel('Entropy')
    print(f"{statistics.median(xsum_bigram_entropies), statistics.mean(xsum_not_bigram_entropies),}")
    print(f"{statistics.median(xsum_bigram_entropies), statistics.median(xsum_not_bigram_entropies),}")


def draw_fig_1():
    fig = plt.figure(figsize=(FIG_SIZE_x, ysize_fig1))

    # plt.rcParams["font.weight"] = "light"
    # plt.rcParams.update({'font.size': 15})
    # plt.rcParams["font.family"] = "Times New Roman"
    cnndm_spec_name = 'd_cnn_dailymail-m_ymail-full0'
    xsum_spec_name = 'd_xsum-m_-xsum-full0'
    draw_x_entropy_y_bigram_count(dir_datadrive, FIG_SIZE_x=GLOBAL_FIGURE_WIDTH,
                                  cnndm_spec_name=cnndm_spec_name, xsum_spec_name=xsum_spec_name)
    fig.tight_layout()
    plt.savefig(f"x_entropy-y_bigram-{cnndm_spec_name}-{xsum_spec_name}.pdf", dpi=dpi)
    plt.show()
    plt.close()


"""Conclusion: for CNNDM, most of the actions are copy (Isbigram). Copy has strong correlation with Bigram.

## Token Position in the sentence - Entropy
"""

# Position related

import math

from matplotlib.axes._axes import Axes


def draw_x_rel_postion_y_entropy(dir, SEPS=10, FIG_SIZE_x=10, FIG_SIZE_y=5):
    cnndm_df, _, _ = _read_data(dir, 'cnndm', SEPS)
    xsum_df, _, _ = _read_data(dir, 'xsum', SEPS)
    colorblind = sns.color_palette("coolwarm", 10)[::-1]

    keys = ['Entropy', 'Top1 Prob', 'token', 'Bigram ', 'InTrigramOfDocument', 'Relative Position']

    # axes = fig.add_axes([0.15, 0.3, 0.84, 0.66])
    # sns.distplot(x=ykey, data=df, hist=False, rug=True)
    # axes = sns.kdeplot(bigram_entropies)
    # axes = sns.kdeplot(not_bigram_entropies)
    #
    axes1: Axes = plt.subplot(121)
    max_lim = 7
    sns.boxplot(x=keys[-1], y=keys[0], data=cnndm_df,
                fliersize=0,
                # palette='coolwarm',
                # color=colorblind[3],
                palette=colorblind,
                # notch=True,
                )
    # axes1.tick_params(which='major', length=5)
    axes1.set_xticks([0, 2, 4, 6, 8])
    axes1.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8])

    # for box in axes1['boxes']:
    #     # change outline color
    #     # box.set(color='#7570b3', linewidth=2)
    #     # change fill color
    #     box.set(edgecolor='white')
    axes1.set_title('CNN/DM')
    axes1.set_ylim(0, max_lim)
    # axes1.set_ylabel('')
    # axes1.legend()

    axes2 = plt.subplot(122, sharey=axes1)

    sns.boxplot(x=keys[-1], y=keys[0], data=xsum_df,
                # notch=True,
                fliersize=0,
                palette=colorblind,
                # palette='Set2',
                # color=colorblind,
                )

    axes2.set_xticks([0, 2, 4, 6, 8])
    axes2.set_xticklabels([0.0, 0.2, 0.4, 0.6, 0.8])
    axes2.set_ylabel('')
    axes2.set_title('XSum')
    axes2.set_ylim(0, max_lim)
    # axes2.legend()
    # plt.setp(axes2.get_yticks(), visible=False)
    # axes2.set_xlabel('Relative Position in Sentence')


def draw_fig_2():
    fig = plt.figure(figsize=(FIG_SIZE_x, ysize_figure2))

    # plt.rcParams["font.weight"] = "light"
    # plt.rcParams.update({'font.size': 15})
    # plt.rcParams["font.family"] = "Times New Roman"
    draw_x_rel_postion_y_entropy(dir_datadrive, FIG_SIZE_x=GLOBAL_FIGURE_WIDTH)
    fig.tight_layout()
    plt.savefig(f"x_rel_postion_y_entropy.pdf", dpi=dpi)

    plt.show()
    plt.close()


import numpy as np


def read_attention_data(data_name, dir='/Users/user/Downloads/', debug=False):
    print(f"DATANAME: {data_name}")
    fname = f"{data_name}_attention.json"
    with open(os.path.join(dir, fname), 'r') as fd:
        x = fd.read()
    data = json.loads(x)
    random.shuffle(data)
    if debug:
        data = data[:2000]
    data_for_panda = []
    flatten = lambda l: [item for sublist in l for item in sublist]
    compar_set1 = ['last_inp', 'cur_inp', 'cur_pred', 'next_pred']
    compar_set2 = ['top1_most_common', 'top1_distill_most_common', 'top3_distill_top3_common']
    compars = flatten([[f"{x}x{y}" for y in compar_set2] for x in compar_set1])
    keys = ['ent', 'emtpy_rate', 'layer']

    SEPS = 10
    tmp_empty = []
    stat_empty, stat_ent = [], []
    for d in data:
        d_for_panda = {}
        for k in keys:
            d_for_panda[k] = d[k]
        # d_for_panda['layer'] += 1
        for k in compars:
            d_for_panda[k] = d[k]
        data_for_panda.append(d_for_panda
                              )
        tmp_empty.append(d['emtpy_rate'])
        if len(tmp_empty) == 12:
            stat_empty.append(statistics.mean(tmp_empty))
            stat_ent.append(d['ent'])
            tmp_empty = []
    df = pd.DataFrame(data_for_panda)
    print(f"Empty rate: {stat_empty}")
    return df


def draw_fig3_barplots(dataframe):
    flatten = lambda l: [item for sublist in l for item in sublist]

    compar_set1 = ['last_inp', 'cur_inp', 'cur_pred', 'next_pred']
    compar_set2 = ['top1_most_common', 'top1_distill_most_common', 'top3_distill_top3_common']
    compars = flatten([[f"{x}x{y}" for y in compar_set2] for x in compar_set1])
    keys = ['ent', 'emtpy_rate', 'layer']
    # max_ylim = 0.75
    axes1 = plt.subplot(221)
    axes1: Axes = sns.barplot(x=keys[-1], y=compars[0 + 1], data=dataframe)
    axes1.set_title('Last Input')
    # axes1.set_ylim(0, max_ylim)
    axes1.set_xlabel('')
    axes1.set_ylabel('')

    # axes1.legend()

    axes2 = plt.subplot(222)
    axes2: Axes = sns.barplot(x=keys[-1], y=compars[1 * 3 + 1], data=dataframe)
    axes2.set_title('Current Input')
    axes2.set_xlabel('')
    axes2.set_ylabel('')
    # axes2.set_ylim(0, 1)
    # axes2.set_title('XSum')
    # axes2.set_ylim(0, 6)

    axes3 = plt.subplot(223)
    axes3: Axes = sns.barplot(x=keys[-1], y=compars[2 * 3 + 1], data=dataframe)
    axes3.set_title('Current Prediction')
    # axes3.set_ylim(0, max_ylim)
    axes3.set_ylabel('')

    axes4 = plt.subplot(224)
    axes4: Axes = sns.barplot(x=keys[-1], y=compars[3 * 3 + 1], data=dataframe)
    axes4.set_title('Next Prediction')
    # axes4.set_ylim(0, 0.2)
    axes4.set_ylabel('')


def read_data_fig3(dataframe):
    flatten = lambda l: [item for sublist in l for item in sublist]

    compar_set1 = ['last_inp', 'cur_inp', 'cur_pred', 'next_pred']
    compar_set2 = ['top1_most_common', 'top1_distill_most_common', 'top3_distill_top3_common']
    compars = flatten([[f"{x}x{y}" for y in compar_set2] for x in compar_set1])
    keys = ['ent', 'emtpy_rate', 'layer']
    max_ylim = 0.75

    last_inpxtop1_distill_most_common = dataframe['last_inpxtop1_distill_most_common'].tolist()
    cur_inpxtop1_distill_most_common = dataframe['cur_inpxtop1_distill_most_common'].tolist()
    cur_predxtop1_distill_most_common = dataframe['cur_predxtop1_distill_most_common'].tolist()
    next_predxtop1_distill_most_common = dataframe['next_predxtop1_distill_most_common'].tolist()
    layer = dataframe['layer'].tolist()
    from collections import Counter
    cnts = [[Counter() for _ in range(4)] for _ in range(12)]
    for l, li, ci, cp, nexp in zip(layer, last_inpxtop1_distill_most_common, cur_inpxtop1_distill_most_common,
                                   cur_predxtop1_distill_most_common, next_predxtop1_distill_most_common):
        cnts[l][0].update([li])
        cnts[l][1].update([ci])
        cnts[l][2].update([cp])
        cnts[l][3].update([nexp])
    bars = [[0 for _ in range(12)] for _ in range(4)]
    for idx, cnt_lay in enumerate(cnts):
        for jdx, cn in enumerate(cnt_lay):
            t = cn[True]
            f = cn[False]
            n = cn[None]
            # print(f"{idx}{jdx} {len(t)}  {len(f)}  {len(n)}")
            bars[jdx][idx] = t / (t + f)
    bar0 = bars[0]
    bar1 = bars[1]
    bar2 = bars[2]
    bar3 = bars[3]
    from operator import add
    bar01 = np.add(bar0, bar1).tolist()
    bar012 = np.add(bar01, bar2).tolist()
    x = list(range(12))
    return bar0, bar1, bar2, bar3, bar01, bar012, x


def draw_fig3_stackbarplots():
    colorblind = sns.color_palette("coolwarm", 4)

    catnames = ['$y_{t-2}$', '$y_{t-1}$',
                '$y_{t}$', '$y_{t+1}$']
    data = 'cnndm'
    dataframe = read_attention_data(data)
    bar0, bar1, bar2, bar3, bar01, bar012, x = read_data_fig3(dataframe)
    ax1: Axes = plt.subplot(121)
    ax1.set_ylim(0, 1.1)
    ax1.set_title("CNN/DM")
    # ax1.set_xticks(list(range(12)))
    # ax1.set_xticklabels(list(range(1,13)))

    ax1.set_xticks([0, 2, 4, 6, 8, 10])
    ax1.set_xticklabels([1, 3, 5, 7, 9, 11])

    ax1.set_ylabel("Aggregate Probability")
    ax1.set_xlabel("Self-Attention Layer")
    x = list(range(12))
    plt.bar(x, bar0, color=colorblind[0], label=catnames[0])

    plt.bar(x, bar1, bottom=bar0, color=colorblind[1], label=catnames[1]
            # ,hatch='-'
            )
    plt.bar(x, bar2, bottom=bar01, color=colorblind[2], label=catnames[2]
            # ,hatch='|'
            )
    plt.bar(x, bar3, bottom=bar012, color=colorblind[3], label=catnames[3]
            # hatch='/'
            )
    # plt.legend(ncol=2,frameon=False)
    data = 'xsum'
    dataframe = read_attention_data(data)
    bar0, bar1, bar2, bar3, bar01, bar012, x = read_data_fig3(dataframe)
    ax2: Axes = plt.subplot(122, sharey=ax1, sharex=ax1)
    plt.setp(ax2.get_yticklabels(), visible=False)

    # ax2.set_xticks(list(range(12)))
    # ax2.set_xticklabels(list(range(1,13)))
    ax2.set_xlabel("Self-Attention Layer")
    x = list(range(12))
    ax2.set_title("XSum")
    # ax2.set_ylim(0,1)
    # ax2.set_xticklabels(list(range(1, 13)))
    ax2 = plt.bar(x, bar0, color=colorblind[0], label=catnames[0])

    ax2 = plt.bar(x, bar1, bottom=bar0, color=colorblind[1], label=catnames[1]
                  # ,hatch='-'
                  )
    ax2 = plt.bar(x, bar2, bottom=bar01, color=colorblind[2], label=catnames[2]
                  # ,hatch='|'
                  )
    ax2 = plt.bar(x, bar3, bottom=bar012, color=colorblind[3], label=catnames[3]
                  # hatch='/'
                  )
    plt.legend(ncol=2, frameon=False)
    return ax2


def draw_figure3():
    """
    fig = plt.figure(figsize=(FIG_SIZE_x, ysize_figure3))

    draw_fig3_barplots(dataframe)
    fig.tight_layout()
    plt.savefig(f"{data}_attn_layer.pdf", dpi=dpi)

    plt.show()
    plt.close()
    """
    # fig = plt.figure(figsize=(FIG_SIZE_x / 0.48125, ysize_figure3))
    fig = plt.figure(figsize=(FIG_SIZE_x, ysize_figure3))
    colorblind = sns.color_palette("RdBu_r", 4)

    catnames = ['Last Input', 'Current Input',
                'Current Pred', 'Next Pred']
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colorblind[0],
                             label=catnames[0]),
                       Patch(facecolor=colorblind[1],
                             label=catnames[1]),
                       Patch(facecolor=colorblind[2],
                             label=catnames[2]),
                       Patch(facecolor=colorblind[3],
                             label=catnames[3]),
                       ]
    # plt.legend(handles=legend_elements,loc='upper center')
    ax2 = draw_fig3_stackbarplots()
    # ax.legend(legend_elements, loc='lower center', ncol=4, labelspacing=0.)
    # plt.legend(bbox_to_anchor=(0, 1.3),loc='upper center', borderaxespad=0.,ncol=4)

    fig.tight_layout()
    plt.savefig(f"stack_attn_layer.pdf", dpi=dpi)

    plt.show()
    plt.close()


if __name__ == '__main__':
    # draw_fig_1()
    draw_fig_2()
    # draw_figure3()
