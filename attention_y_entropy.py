import itertools
import os, random
import statistics

import matplotlib
import matplotlib.pyplot as plt
from analyze_entropy import comp_entropy
from analyze_prob_attn import compute_idf, get_ban_positions
# from data_collection import CUR_DIR, PROB_META_DIR, spec_name, MODEL_NAME, DATA_NAME
from util import convert_enc_attn, parse_arg

from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)

font_size = 14
matplotlib.font_manager._rebuild()
GLOBAL_FIGURE_WIDTH = 8
dpi = 800
# plt.rcParams["font.weight"] = "light"
plt.rcParams.update({'font.size': 14})
plt.rcParams['font.family'] = 'DejaVu Sans'

dir_datadrive = '/mnt/data0/jcxu/data/prob_gpt'

# Density: x: entropy, two plots: is bigram or not
FIG_SIZE_x = GLOBAL_FIGURE_WIDTH


def get_ys(t, logits, BOS_token=0):
    # for one time step, get the tokens last_inp, cur_inp, cur_pred, and next_pred

    cur_pred = logits[t]
    try:
        next_pred = logits[t + 1]
    except IndexError:
        next_pred = None

    if t - 2 >= 0:
        last_inp = logits[t - 2]
    elif t - 2 == -1:
        last_inp = BOS_token
    else:
        last_inp = None

    if t - 1 >= 0:
        cur_inp = logits[t - 1]
    elif t - 1 == -1:
        cur_inp = BOS_token
    else:
        cur_inp = None
    return last_inp, cur_inp, cur_pred, next_pred


from collections import Counter


def truncate_attention_cell(attn_distb, input_doc, idf_ban_pos, tgt_prob_mass=0.9) -> Counter:
    # for each attention distribution, remove the idf ban tokens (positions), get the accumulated prob up to prob_mass.
    # return
    sorts = np.argsort(attn_distb, axis=-1, kind=None, order=None)[::-1]
    cum_prob_mass = 0
    cnt = Counter()
    for topk in sorts:
        prob_mass = attn_distb[topk]

        cum_prob_mass += prob_mass
        if topk not in idf_ban_pos:
            cnt[input_doc[topk]] = prob_mass
        if cum_prob_mass > tgt_prob_mass or prob_mass < 0.01:
            break
    return cnt


def _y_entropy_step(attn_lle, input_doc, idf_ban_pos):
    num_layer, num_head, src_len = attn_lle.shape
    all_attns = Counter()
    for l in range(num_layer):
        for h in range(num_head):
            cell_attn = truncate_attention_cell(attn_lle[l][h], input_doc, idf_ban_pos=idf_ban_pos)
            all_attns = all_attns + cell_attn
    return all_attns


def retrieve_tok_val(cnter, token):
    try:
        v = cnter[token]
    except:
        v = 0
    return v


import matplotlib

import matplotlib.pyplot as plt


def plot_hist(val_ent_pairs, title):
    weights = [p[0] for p in val_ent_pairs]
    xs = [p[1] for p in val_ent_pairs]

    plt.hist(xs, bins=20, weights=weights, density=True)

    plt.xlabel('Pred Ent')
    plt.ylabel('Cum Attn')

    plt.title(title)
    plt.grid(True)
    plt.show()


import seaborn as sns


def get_statistics(matrix):
    result = [0 for _ in range(len(matrix))]
    for idx, row in enumerate(matrix):
        try:
            m = statistics.mean(row)
        except:
            m = 0
            print("NO DATA!")
        result[idx] = m
    return result


def proceed_data(segs, val_ent_pairs, step_size=0.5):
    cat_bins = [[[] for _ in range(segs)] for _ in range(5)]
    for p in val_ent_pairs:
        last_inp, cur_inp, cur_pred, next_pred, pred_ent, atte_ent = p
        # attn_val, ent, attn_e = p[0], p[1], p[2]
        cat = int(pred_ent // step_size)
        try:
            cat_bins[0][cat].append(last_inp)
            cat_bins[1][cat].append(cur_inp)
            cat_bins[2][cat].append(cur_pred)
            cat_bins[3][cat].append(next_pred)
            cat_bins[4][cat].append(atte_ent)
        except:
            pass
    last_inp_mean = get_statistics(cat_bins[0])
    cur_inp_mean = get_statistics(cat_bins[1])
    cur_pred_mean = get_statistics(cat_bins[2])
    next_pred_mean = get_statistics(cat_bins[3])
    atte_ent_mean = get_statistics(cat_bins[4])
    return last_inp_mean, cur_inp_mean, cur_pred_mean, next_pred_mean, atte_ent_mean


def read_stack_data(last_inp_mean, cur_inp_mean, cur_pred_mean, next_pred_mean, seps=10):
    bar0 = last_inp_mean
    bar1 = cur_inp_mean
    bar2 = cur_pred_mean
    bar3 = next_pred_mean
    from operator import add
    bar01 = np.add(bar0, bar1).tolist()
    bar012 = np.add(bar01, bar2).tolist()
    # x = list(range(10))
    return bar0, bar1, bar2, bar3, bar01, bar012


def plot_single_line(this_fig, spec_config, input_data, step_size=0.5, ent_max=5,
                     show_x_ticks=False, show_y_ticks=True, data_name="", model_name="", ymin=2, ymax=5):
    segs = np.arange(0, ent_max, step_size).tolist()
    # colorblind = sns.color_palette("coolwarm", 10)[::-1]
    last_inp_mean, cur_inp_mean, cur_pred_mean, next_pred_mean, atte_ent_mean = input_data

    axes = this_fig.add_subplot(spec_config)
    sns.lineplot(x=list(np.arange(0, 5, step_size)), y=atte_ent_mean, markers=True, dashes=False)

    # axes = sns.boxplot(x=x, y=y, palette=colorblind, showfliers=False)

    axes.xaxis.set_major_locator(MultipleLocator(1))
    axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # For the minor ticks, use no labels; default NullFormatter.
    axes.xaxis.set_minor_locator(MultipleLocator(0.5))
    # axes.get_ylim()
    # axes.set_ylim(ymin, ymax)

    if show_x_ticks:
        # x_vals = [m * step_size for m in xticks]
        # axes.set_xticklabels(x_vals, rotation='vertical')center
        pass
    else:
        plt.setp(axes.get_xticklabels(), visible=False)

    # if not show_y_ticks:
    #     plt.setp(axes.get_yticklabels(), visible=False)
    if data_name != "":
        axes.set_ylabel(data_name)
    else:
        axes.set_ylabel("")
    if model_name != "":
        axes.set_title(model_name)
    return axes


def plot_single_box(this_fig, spec_config, input_data, step_size=0.5, ent_max=5,
                    show_x_ticks=False, show_y_ticks=True, ylim=0.8, data_name="", model_name="", show_legend=False):
    segs = np.arange(0, ent_max, step_size).tolist()
    # colorblind = sns.color_palette("coolwarm", 10)[::-1]
    last_inp_mean, cur_inp_mean, cur_pred_mean, next_pred_mean, atte_ent_mean = input_data
    bar0, bar1, bar2, bar3, bar01, bar012 = read_stack_data(last_inp_mean, cur_inp_mean, cur_pred_mean, next_pred_mean)
    colorblind = sns.color_palette("coolwarm", 4)
    # colorblind = sns.color_palette("Set2")

    # colorblind = sns.color_palette()

    catnames = ['$y_{t-2}$', '$y_{t-1}$',
                '$y_{t}$', '$y_{t+1}$']
    linewidth = 1.5

    axes = this_fig.add_subplot(spec_config)
    x = list(np.arange(0, 5, 0.5))
    axes.bar(x, bar0, color=colorblind[0],
             # edgecolor=colorblind[0],linewidth=linewidth,
             label=catnames[0], width=step_size,

             # hatch='/'
             )

    axes.bar(x, bar1, bottom=bar0,
             # edgecolor='white', linewidth=1,
             label=catnames[1], width=step_size,
             # hatch='-',
             facecolor=colorblind[1],
             # histtype='step', facecolor='g',
             # alpha=0.75
             # ,hatch='-'
             )
    axes.bar(x, bar2, bottom=bar01,
             # edgecolor=colorblind[3], linewidth=0,
             label=catnames[2], width=step_size, facecolor=colorblind[3],
             # histtype='step',
             # hatch='|'
             # ,hatch='|'
             )
    axes.bar(x, bar3, bottom=bar012, color=colorblind[2], label=catnames[3], width=step_size,
             # edgecolor=colorblind[2],             linewidth=linewidth,
             # hatch='\\'
             )
    # axes = sns.boxplot(x=x, y=y, palette=colorblind, showfliers=False)

    axes.xaxis.set_major_locator(MultipleLocator(1))
    axes.xaxis.set_major_formatter(FormatStrFormatter('%d'))
    if show_legend:
        axes.legend(ncol=2, frameon=False)

    # For the minor ticks, use no labels; default NullFormatter.
    axes.xaxis.set_minor_locator(MultipleLocator(0.5))

    axes.set_ylim(0, ylim)

    if show_x_ticks:
        # x_vals = [m * step_size for m in xticks]
        # axes.set_xticklabels(x_vals, rotation='vertical')center
        pass
    else:
        plt.setp(axes.get_xticklabels(), visible=False)

    if not show_y_ticks:
        plt.setp(axes.get_yticklabels(), visible=False)
    if data_name != "":
        axes.set_ylabel(data_name)
    else:
        axes.set_ylabel("")
    if model_name != "":
        axes.set_title(model_name)
    return axes


def plot_box(val_ent_pairs, title=None, step_size=.25):
    # max_pred_ent = max([p[1] for p in val_ent_pairs])
    # segs = np.linspace(0, max_pred_ent + 0.1, num=20).tolist()
    segs = np.arange(0, 8, step_size).tolist()
    colorblind = sns.color_palette("coolwarm", 10)[::-1]

    bins = [[] for _ in range(len(segs))]
    x, y = [], []
    for p in val_ent_pairs:
        v, ent = p[0], p[1]
        cat = int(ent // step_size)
        try:
            bins[cat].append(v)
            x.append(cat)
            y.append(v)
        except:
            pass
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1 = sns.violinplot(x=x, y=y, cut=0, palette=colorblind, inner='quartile')

    # ax1.set_xticks( np.arange(0, 8, step_size).tolist())
    # ax1.set_xticklabels(np.arange(0, 8, step_size).tolist())

    return ax1


def plot_single_scatter(val_ent_pairs, title):
    y_attn_frac = [m[0] for m in val_ent_pairs]
    x_pred_ent = [m[1] for m in val_ent_pairs]
    ax = sns.jointplot(x=x_pred_ent, y=y_attn_frac, kind="hex", color="#4CB391")

    # ax = sns.scatterplot(x=x_pred_ent,y=y_attn_frac)
    #
    # sns.histplot(x=x_pred_ent, y=y_attn_frac, bins=50, pthresh=.1, cmap="mako")
    # sns.kdeplot(x=x_pred_ent, y=y_attn_frac, levels=5,linewidths=1)
    # ax.set_title(title)

    plt.show()
    return ax


def analyze_attention_y_entropy(max_time_step, attn_tlle, pred_distribution, input_doc, ban_positions, logits, nuc,
                                top_p):
    # T = attn_tlle.shape[0]
    # data_pairs = [[], [], [], []]
    data = []
    for t in range(max_time_step):
        try:
            t_pred_ent = comp_entropy(pred_distribution[t], nuc, top_p)

            last_inp, cur_inp, cur_pred, next_pred = get_ys(t, logits)
            all_attns_counter = _y_entropy_step(attn_tlle[t], input_doc, ban_positions)
            total_attn_val = sum(all_attns_counter.values())
            all_attention = list(all_attns_counter.values())
            np_attn = np.asarray(all_attention) / total_attn_val
            attn_ent = comp_entropy(np_attn)
            last_inp_val = retrieve_tok_val(all_attns_counter, last_inp)
            cur_inp_val = retrieve_tok_val(all_attns_counter, cur_inp)
            cur_pred_val = retrieve_tok_val(all_attns_counter, cur_pred)
            next_pred_val = retrieve_tok_val(all_attns_counter, next_pred)
            # data_pairs[0].append((last_inp_val / total_attn_val, t_pred_ent))
            # data_pairs[1].append((cur_inp_val / total_attn_val, t_pred_ent))
            # data_pairs[2].append((cur_pred_val / total_attn_val, t_pred_ent))

            data.append((last_inp_val / total_attn_val, cur_inp_val / total_attn_val,
                         cur_pred_val / total_attn_val, next_pred_val / total_attn_val,
                         t_pred_ent, attn_ent))
        except:
            pass
        # data_pairs[3].append((next_pred_val / total_attn_val, t_pred_ent))
    return data


import pickle
import numpy as np
from scipy.stats import entropy
import matplotlib.gridspec as gridspec
import multiprocessing


def detect_useless_ids(indices):
    last = -100
    good_indices = []
    for x in indices:
        if x - 5 > last:
            last = x
            good_indices.append(x)
        else:
            break
    return good_indices


def process_data_single(args, f, eos_token_ids):
    print("running")
    BOS_TOKEN = 0

    with open(os.path.join(args.cur_dir, f), 'rb') as fd:
        data = pickle.load(fd)

    attentions, pred_distb, logits, input_doc = data['attentions'], data['pred_distributions'], data['logits'], \
                                                data['input_doc']

    timesteps = len(attentions)
    attentions_tlle = convert_enc_attn(attentions, merge_layer_head=False)  # T,L,L,E
    attention_tle = convert_enc_attn(attentions, merge_layer_head=True)  # T,L,E

    document_len = input_doc.shape[0]
    input_doc = input_doc.astype(np.int).tolist()
    logits = logits.tolist()
    indices = [i for i, x in enumerate(logits) if x in eos_token_ids]
    good_indices = detect_useless_ids(indices)
    if good_indices:
        max_t = good_indices[-1]
    else:
        max_t = attentions_tlle.shape[0]
        # dec_inp_logits = [BOS_TOKEN] + logits[:-1]
    pred_distb = np.exp(pred_distb)  # time step, vocab size
    # pred_ent = entropy(pred_distb, axis=-1)
    idf_flag = compute_idf(attention_tle[:max_t])  # E
    ban_positions = get_ban_positions(idf_flag)
    # ban_positions = []
    data_pairs = analyze_attention_y_entropy(max_t, attentions_tlle, pred_distb, input_doc, ban_positions, logits,
                                             args.nucleus, args.nuc_prob)
    return data_pairs


from itertools import product


def plot_stack_vocab(cnndm_peg, xsum_peg, cnndm_bart, xsum_bart):
    fig = plt.figure(figsize=(FIG_SIZE_x, FIG_SIZE_x - 4))
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    plot_single_box(this_fig=fig, spec_config=spec2[0, 0], input_data=cnndm_peg, show_x_ticks=False,
                    show_y_ticks=True, data_name="CNN/DM", model_name="PEGASUS", ylim=0.7)
    plot_single_box(this_fig=fig, spec_config=spec2[0, 1], input_data=cnndm_bart, show_x_ticks=False,
                    show_y_ticks=False, model_name="BART", ylim=0.7)
    plot_single_box(this_fig=fig, spec_config=spec2[1, 0], input_data=xsum_peg, show_x_ticks=True, show_y_ticks=True,
                    data_name='XSum', ylim=0.4)
    plot_single_box(this_fig=fig, spec_config=spec2[1, 1], input_data=xsum_bart, show_x_ticks=True,
                    show_y_ticks=False, ylim=0.4, show_legend=True)

    fig.text(0.5, 0.01, 'Prediction Entropy', ha='center', fontsize=font_size)
    fig.text(0.0, 0.5, 'Vocab Projected Attention', va='center', rotation='vertical', fontsize=font_size)
    fig.tight_layout()
    plt.savefig(f"x_pred_ent_y_attn_frac.pdf", dpi=dpi, bbox_inches='tight')
    plt.show()
    plt.close()


def run_one_fig(spec, args, num_samples=300):
    print(f"--{spec}--")
    CUR_DIR = os.path.join(args.prob_meta_dir, spec)
    args.cur_dir = CUR_DIR
    files = os.listdir(CUR_DIR)
    random.shuffle(files)
    files = files[:num_samples]

    BOS_TOKEN = 0
    print(args.spec_name)
    if 'pegasus' in args.model_name:
        from transformers import PegasusTokenizer

        bpe_tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
        EOS_TOK_IDs = [106, bpe_tokenizer.eos_token_id, 2]  # <n>
    elif 'gpt' in args.model_name:
        from transformers import GPT2Tokenizer

        bpe_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        EOS_TOK_IDs = [bpe_tokenizer.eos_token_id]
    elif 'bart' in args.model_name:
        from transformers import BartTokenizer

        bpe_tokenizer = BartTokenizer.from_pretrained(args.model_name)
        EOS_TOK_IDs = [bpe_tokenizer.eos_token_id]
    else:
        raise NotImplementedError
    # process_data_single(args, files[0], eos_token_ids=EOS_TOK_IDs)
    len_samples = len(files)
    cpu_cnt = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpu_cnt) as pool:
        results = pool.starmap(process_data_single, zip([args] * len_samples, files, [EOS_TOK_IDs] * len_samples))
    output = list(itertools.chain.from_iterable(results))
    print(f"Samples: {len(output)}")
    output = proceed_data(10, output)
    return output


def plot_ant_entropy(cnndm_peg, xsum_peg, cnndm_bart, xsum_bart):
    fig = plt.figure(figsize=(FIG_SIZE_x, FIG_SIZE_x - 5))
    step_size = 0.5
    d = {'PEG$_{C}$': cnndm_peg[-1],
         'PEG-X': xsum_peg[-1],
         'BART-C': cnndm_bart[-1],
         'BART-X': xsum_bart[-1],
         }
    # df = pd.DataFrame(data=d)
    ax = fig.add_subplot(1, 1, 1)
    # line1 = sns.lineplot(x=list(np.arange(0, 5, step_size)), y=cnndm_peg[-1], label='PEG$_{C}$', markers='x')
    plt.plot(list(np.arange(0, 5, step_size)), cnndm_peg[-1], label='PEG$_{C}$', marker='+',
             # color='k'
             )
    plt.plot(list(np.arange(0, 5, step_size)), xsum_peg[-1], label='PEG$_{X}$', marker='x',
             # color='k'
             )
    plt.plot(list(np.arange(0, 5, step_size)), cnndm_bart[-1], label='BART$_{C}$', ls='--', marker='+',
             # color='k'
             )
    plt.plot(list(np.arange(0, 5, step_size)), xsum_bart[-1], label='BART$_{X}$', ls='--', marker='x',
             # color='k'
             )

    plt.legend(loc='best', ncol=2, frameon=False)

    # spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    # plot_single_line(this_fig=fig, spec_config=spec2[0, 0], input_data=cnndm_peg, show_x_ticks=False,
    #                  show_y_ticks=True, data_name="CNN/DM", model_name="PEGASUS", ymin=2, ymax=4
    #                  )
    # plot_single_line(this_fig=fig, spec_config=spec2[0, 1], input_data=cnndm_bart, show_x_ticks=False,
    #                  show_y_ticks=False, model_name="BART", ymin=2, ymax=4)
    # plot_single_line(this_fig=fig, spec_config=spec2[1, 0], input_data=xsum_peg, show_x_ticks=True, show_y_ticks=True,
    #                  data_name='XSUM', ymin=2.5, ymax=4)
    # plot_single_line(this_fig=fig, spec_config=spec2[1, 1], input_data=xsum_bart, show_x_ticks=True,
    #                  show_y_ticks=False, ymin=2.5, ymax=4)
    ax.set_ylabel('Attention Entropy')
    ax.set_xlabel('Prediction Entropy')

    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # For the minor ticks, use no labels; default NullFormatter.
    # plt.xaxis.set_minor_locator(MultipleLocator(0.5))
    fig.tight_layout()
    plt.savefig(f"atten_entropy.pdf", dpi=dpi, bbox_inches='tight')

    # fig.text(0.5, 0.01, 'Prediction Entropy', ha='center')
    # fig.text(0.0, 0.5, '', va='center', rotation='vertical')

    plt.show()
    plt.close()


import pandas as pd

if __name__ == '__main__':
    args = parse_arg()
    print("Looking at attention")
    if 'pegasus' in args.model_name:
        from transformers import PegasusTokenizer

        bpe_tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
        EOS_TOK_IDs = [106, bpe_tokenizer.eos_token_id]  # <n>
        BOS_TOK_ID = 0
    else:
        raise NotImplementedError

    cnndm_peg = "d_cnn_dailymail-m_googlepegasuscnn_dailymail-full1"
    xsum_peg = "d_xsum-m_googlepegasusxsum-full1"
    cnndm_bart = "d_cnn_dailymail-m_facebookbartlargecnn-full1"
    xsum_bart = 'd_xsum-m_facebookbartlargexsum-full1'
    xsum_bart_out = run_one_fig(xsum_bart, args)
    cnndm_peg_out = run_one_fig(cnndm_peg, args)
    xsum_peg_out = run_one_fig(xsum_peg, args)
    cnndm_bart_out = run_one_fig(cnndm_bart, args)

    # df = pd.DataFrame(data=xsum_bart_out)
    # plot_stack_vocab(cnndm_peg_out, xsum_peg_out, cnndm_bart_out, xsum_bart_out)

    plot_stack_vocab(cnndm_peg_out, xsum_peg_out, cnndm_bart_out, xsum_bart_out)
    plot_ant_entropy(cnndm_peg_out, xsum_peg_out, cnndm_bart_out, xsum_bart_out)
    # plot_box(all_data_pairs[0], 'last_inp')
    # plot_box(all_data_pairs[1], 'cur_inp')
    # plot_box(all_data_pairs[2], 'cur_pred')
    # plot_box(all_data_pairs[3], 'next_pred')
