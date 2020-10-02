
import argparse

import json, math, seaborn
import statistics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

# del matplotlib.font_manager.weight_dict['roman']
from plot_figures import _read_data, normalize_figure1

matplotlib.font_manager._rebuild()
GLOBAL_FIGURE_WIDTH = 8
dpi = 800
# plt.rcParams["font.weight"] = "light"
plt.rcParams.update({'font.size': 14})
# plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['font.family'] = 'DejaVu Sans'

dir_datadrive = '/mnt/data0/jcxu/data/prob_gpt'


# Density: x: entropy, two plots: is bigram or not
FIG_SIZE_x = GLOBAL_FIGURE_WIDTH
FIG_SIZE_y = 3
ysize_fig1 = 4
ysize_figure2 = 4
ysize_figure3 = 4

def plot_fig1_single(locid:int):
    pass

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


def draw_fig_1(cnndm_spec_name,xsum_spec_name):
    fig = plt.figure(figsize=(FIG_SIZE_x, ysize_fig1))

    # plt.rcParams["font.weight"] = "light"
    # plt.rcParams.update({'font.size': 15})
    # plt.rcParams["font.family"] = "Times New Roman"
    # cnndm_spec_name = 'd_cnn_dailymail-m_ymail-full1'
    # xsum_spec_name = 'd_xsum-m_-xsum-full1'
    draw_x_entropy_y_bigram_count(dir_datadrive, FIG_SIZE_x=GLOBAL_FIGURE_WIDTH,
                                  cnndm_spec_name=cnndm_spec_name, xsum_spec_name=xsum_spec_name)
    fig.tight_layout()
    plt.savefig(f"x_entropy-y_bigram-{cnndm_spec_name}-{xsum_spec_name}.pdf", dpi=dpi)
    plt.show()
    plt.close()
