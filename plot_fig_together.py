import argparse

import json, math, seaborn
import statistics
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List

# del matplotlib.font_manager.weight_dict['roman']
from plot_figures import _read_data, _read_data_position_fig2, normalize_figure1

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


def plot_fig1_single(this_fig, spec, dir, spec_name, SEPS, max_lim=5, bins_num=15, x_ticklabel_vis=True,
                     y_ticklabel_vis=True,
                     y_namelabel_vis=True, showLegend=False, data_name="", ylim=500, title=""):
    axes1 = this_fig.add_subplot(spec)
    ExistingBigram = "Existing Bigrams"
    NovelBigram = "Novel Bigrams"
    linewidth = 1.5
    _, cnndm_bigram_entropies, cnndm_not_bigram_entropies = _read_data(dir, spec_name, SEPS)
    cnndm_bigram_entropies, cnndm_not_bigram_entropies = normalize_figure1(cnndm_bigram_entropies,
                                                                           cnndm_not_bigram_entropies)
    # axes1 = plt.subplot(locid)
    # axes = fig.add_axes([0., 0.3, 0.84, 0.66])
    # sns.distplot(x=ykey, data=df, hist=False, rug=True)
    # axes = sns.distplot(bigram_entropies,rug=True)
    # axes = sns.kdeplot(bigram_entropies, shade=True,cumulative=True)
    # axes = sns.distplot(bigram_entropies,rug=True)
    color = sns.color_palette("coolwarm", 7)
    if showLegend:
        axes1 = sns.distplot(cnndm_bigram_entropies, bins=bins_num,
                             hist_kws={'range': [0, max_lim]},
                             hist=True,
                             kde=False,
                             color=color[0],
                             label=f"{ExistingBigram}"
                             )
    else:
        axes1 = sns.distplot(cnndm_bigram_entropies, bins=bins_num,
                             hist_kws={'range': [0, max_lim]},
                             hist=True,
                             kde=False,
                             color=color[0],
                             # label=f"{ExistingBigram}"
                             )
    if title != "":
        axes1.set_title(title)
    if y_namelabel_vis:
        axes1.set_ylabel(data_name)
    plt.ylim(0, ylim)
    if showLegend:
        axes1 = sns.distplot(cnndm_not_bigram_entropies, bins=bins_num, hist_kws={'range': [0, max_lim]},
                             hist=True,
                             kde=False,
                             color=color[-1],
                             label=f"{NovelBigram}"
                             )
    else:
        axes1 = sns.distplot(cnndm_not_bigram_entropies, bins=bins_num, hist_kws={'range': [0, max_lim]},
                             hist=True,
                             kde=False,
                             color=color[-1],
                             # label=f"{NovelBigram}"
                             )
    plt.axvline(statistics.median(cnndm_bigram_entropies), color=color[0], linestyle='dashed', linewidth=linewidth)
    plt.axvline(statistics.median(cnndm_not_bigram_entropies), color=color[-1], linestyle='dashed', linewidth=linewidth)
    # axes.legend(prop={'size': 10})
    axes1.legend(frameon=False)
    # print(f"{statistics.median(cnndm_bigram_entropies), statistics.mean(cnndm_not_bigram_entropies),}")
    print(f"{statistics.median(cnndm_bigram_entropies), statistics.median(cnndm_not_bigram_entropies)}")
    if not x_ticklabel_vis:
        plt.setp(axes1.get_xticklabels(), visible=False)
    if not y_ticklabel_vis:
        plt.setp(axes1.get_yticklabels(), visible=False)
    return axes1


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


import matplotlib.gridspec as gridspec


def draw_fig_1(cnndm_peg, xsum_peg, cnndm_bart, xsum_bart):
    fig = plt.figure(figsize=(FIG_SIZE_x, ysize_fig1))
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)

    # plt.rcParams["font.weight"] = "light"
    # plt.rcParams.update({'font.size': 15})
    # plt.rcParams["font.family"] = "Times New Roman"
    # cnndm_spec_name = 'd_cnn_dailymail-m_ymail-full1'
    # xsum_spec_name = 'd_xsum-m_-xsum-full1'
    plot_fig1_single(fig, spec2[0, 0], dir=dir_datadrive, spec_name=cnndm_peg, SEPS=10, data_name='CNN/DM',
                     x_ticklabel_vis=False, ylim=500, title="PEGASUS")
    plot_fig1_single(fig, spec2[0, 1], dir=dir_datadrive, spec_name=cnndm_bart, SEPS=10, y_ticklabel_vis=False,
                     x_ticklabel_vis=False, ylim=500, showLegend=True,
                     title="BART")
    plot_fig1_single(fig, spec2[1, 0], dir=dir_datadrive, spec_name=xsum_peg, SEPS=10, data_name='XSUM',
                     ylim=250)
    plot_fig1_single(fig, spec2[1, 1], dir=dir_datadrive, spec_name=xsum_bart, SEPS=10, y_namelabel_vis=False,
                     y_ticklabel_vis=False, ylim=250)

    fig.tight_layout()
    plt.savefig(f"x_entropy-y_bigram-grid.pdf", dpi=dpi)
    plt.show()
    plt.close()


from matplotlib.axes._axes import Axes


def plot_single_fig2(this_fig, spec, dir, spec_name, SEPS,
                     x_ticklabel_vis=True,
                     y_ticklabel_vis=True,
                     data_name="", model_name="", xlabel="",
                     ylim=7
                     ):
    axes1 = this_fig.add_subplot(spec)
    cnndm_df = _read_data_position_fig2(dir, spec_name, SEPS)
    colorblind = sns.color_palette("coolwarm", 10)[::-1]

    keys = ['Relative Position', 'Entropy']

    # axes = fig.add_axes([0.15, 0.3, 0.84, 0.66])
    # sns.distplot(x=ykey, data=df, hist=False, rug=True)
    # axes = sns.kdeplot(bigram_entropies)
    # axes = sns.kdeplot(not_bigram_entropies)
    #
    # axes1: Axes = plt.subplot(121)
    sns.boxplot(x=keys[0], y=keys[1], data=cnndm_df,
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
    if not x_ticklabel_vis:
        plt.setp(axes1.get_xticklabels(), visible=False)
    if not y_ticklabel_vis:
        plt.setp(axes1.get_yticklabels(), visible=False)
    if model_name != "":
        axes1.set_title(model_name)
    if xlabel != "":
        axes1.set_xlabel(xlabel)
    else:
        axes1.set_xlabel("")
    if data_name != "":
        axes1.set_ylabel(data_name)
    else:
        axes1.set_ylabel("")

    axes1.set_ylim(0, ylim)


def draw_fig_2(cnndm_peg, xsum_peg, cnndm_bart, xsum_bart):
    fig = plt.figure(figsize=(FIG_SIZE_x, ysize_figure2 + 2))
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    # plt.rcParams["font.weight"] = "light"
    # plt.rcParams.update({'font.size': 15})
    # plt.rcParams["font.family"] = "Times New Roman"
    """
    ax = fig.add_subplot(111)  # The big subplot
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.tick_params(labelcolor='w', top=False, bottom=False, left=False, right=False)
    # ax.set_ylabel("Entropy")
    ax.set_xlabel('Relative Position')
"""
    fig.text(0.5, 0.0, 'Relative Position', ha='center')
    fig.text(0.0, 0.5, 'Entropy', va='center', rotation='vertical')

    # cnndm_spec_name = 'd_cnn_dailymail-m_ymail-full1'
    # xsum_spec_name = 'd_xsum-m_-xsum-full1'
    plot_single_fig2(fig, spec2[0, 0], dir=dir_datadrive, spec_name=cnndm_peg, SEPS=10,
                     x_ticklabel_vis=False,
                     y_ticklabel_vis=True,
                     data_name="CNN/DM", model_name="PEGASUS", ylim=8
                     )
    plot_single_fig2(fig, spec2[0, 1], dir=dir_datadrive, spec_name=cnndm_bart, SEPS=10,
                     x_ticklabel_vis=False,
                     y_ticklabel_vis=False,
                     model_name="BART", ylim=8
                     )
    plot_single_fig2(fig, spec2[1, 0], dir=dir_datadrive, spec_name=xsum_peg, SEPS=10,
                     x_ticklabel_vis=True,
                     y_ticklabel_vis=True,
                     data_name="XSUM",
                     )
    plot_single_fig2(fig, spec2[1, 1], dir=dir_datadrive, spec_name=xsum_bart, SEPS=10,
                     x_ticklabel_vis=True,
                     y_ticklabel_vis=False,
                     )
    fig.tight_layout()
    plt.savefig(f"position_grid.pdf", dpi=dpi, bbox_inches='tight')

    plt.show()
    plt.close()


if __name__ == '__main__':
    cnndm_peg = "d_cnn_dailymail-m_googlepegasuscnn_dailymail-full10.95"
    xsum_peg = "d_xsum-m_googlepegasusxsum-full10.95"
    cnndm_bart = "d_cnn_dailymail-m_facebookbartlargecnn-full10.95"
    xsum_bart = 'd_xsum-m_facebookbartlargexsum-full10.95'
    # draw_fig_1(cnndm_peg, xsum_peg, cnndm_bart, xsum_bart)

    draw_fig_2(cnndm_peg, xsum_peg, cnndm_bart, xsum_bart)
