import itertools
import os, random
import matplotlib
import matplotlib.pyplot as plt
from analyze_entropy import comp_entropy
from analyze_prob_attn import compute_idf, get_ban_positions
# from data_collection import CUR_DIR, PROB_META_DIR, spec_name, MODEL_NAME, DATA_NAME
from util import convert_enc_attn, parse_arg

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


def plot_single_box(this_fig, spec_config, val_ent_pairs, step_size=0.5, ent_max=5,
                    show_x_ticks=False, show_y_ticks=True, ylim=0.8, data_name="", model_name=""):
    segs = np.arange(0, ent_max, step_size).tolist()
    colorblind = sns.color_palette("coolwarm", 10)[::-1]

    bins = [[] for _ in range(len(segs))]
    x, y = [], []
    for p in val_ent_pairs:
        attn_val, ent = p[0], p[1]
        cat = int(ent // step_size)
        try:
            bins[cat].append(attn_val)
            x.append(cat)
            y.append(attn_val)
        except:
            pass
    axes = this_fig.add_subplot(spec_config)

    axes = sns.boxplot(x=x, y=y, palette=colorblind, showfliers=False)
    axes.set_ylim(0, ylim)
    xticks = np.arange(0, len(segs), 2).tolist()
    axes.set_xticks(xticks)

    if show_x_ticks:
        x_vals = [m * step_size for m in xticks]
        axes.set_xticklabels(x_vals, rotation='vertical')
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
    max_pred_ent = max([p[1] for p in val_ent_pairs])
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


def analyze_attention_y_entropy(attn_tlle, pred_distribution, input_doc, ban_positions, logits, nuc, top_p):
    T = attn_tlle.shape[0]
    # data_pairs = [[], [], [], []]
    data = []
    for t in range(T):
        t_pred_ent = comp_entropy(pred_distribution[t], nuc, top_p)

        last_inp, cur_inp, cur_pred, next_pred = get_ys(t, logits)
        all_attns_counter = _y_entropy_step(attn_tlle[t], input_doc, ban_positions)
        total_attn_val = sum(all_attns_counter.values())
        # last_inp_val = retrieve_tok_val(all_attns_counter, last_inp)
        # cur_inp_val = retrieve_tok_val(all_attns_counter, cur_inp)
        cur_pred_val = retrieve_tok_val(all_attns_counter, cur_pred)
        # next_pred_val = retrieve_tok_val(all_attns_counter, next_pred)
        # data_pairs[0].append((last_inp_val / total_attn_val, t_pred_ent))
        # data_pairs[1].append((cur_inp_val / total_attn_val, t_pred_ent))
        # data_pairs[2].append((cur_pred_val / total_attn_val, t_pred_ent))
        try:
            data.append((cur_pred_val / total_attn_val, t_pred_ent))
        except:
            pass
        # data_pairs[3].append((next_pred_val / total_attn_val, t_pred_ent))
    return data


import pickle
import numpy as np
from scipy.stats import entropy
import matplotlib.gridspec as gridspec
import multiprocessing


def process_data_single(args, f,eos_token_ids):
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

    dec_inp_logits = [BOS_TOKEN] + logits[:-1]
    pred_distb = np.exp(pred_distb)  # time step, vocab size
    # pred_ent = entropy(pred_distb, axis=-1)
    idf_flag = compute_idf(attention_tle)  # E
    ban_positions = get_ban_positions(idf_flag)
    # ban_positions = []
    data_pairs = analyze_attention_y_entropy(attentions_tlle, pred_distb, input_doc, ban_positions, logits,
                                             args.nucleus, args.nuc_prob)
    return data_pairs


from itertools import product


def plot(cnndm_peg, xsum_peg, cnndm_bart, xsum_bart):
    fig = plt.figure(figsize=(FIG_SIZE_x, FIG_SIZE_x - 4))
    spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig)
    plot_single_box(this_fig=fig, spec_config=spec2[0, 0], val_ent_pairs=cnndm_peg, show_x_ticks=False,
                    show_y_ticks=True, data_name="CNN/DM", model_name="PEGASUS")
    plot_single_box(this_fig=fig, spec_config=spec2[0, 1], val_ent_pairs=cnndm_bart, show_x_ticks=False,
                    show_y_ticks=False, model_name="BART")
    plot_single_box(this_fig=fig, spec_config=spec2[1, 0], val_ent_pairs=xsum_peg, show_x_ticks=True, show_y_ticks=True,
                    data_name='XSUM',ylim=0.5)
    plot_single_box(this_fig=fig, spec_config=spec2[1, 1], val_ent_pairs=xsum_bart, show_x_ticks=True,
                    show_y_ticks=False,ylim=0.5)
    fig.tight_layout()
    plt.savefig(f"x_pred_ent_y_attn_frac.pdf", dpi=dpi, bbox_inches='tight')

    fig.text(0.5, 0.0, 'Prediction Entropy', ha='center')
    fig.text(0.0, 0.5, 'Aggregate Attention Probability', va='center', rotation='vertical')

    plt.show()
    plt.close()


def run_one_fig(spec, args, num_samples=30):
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
        EOS_TOK_IDs = [106, bpe_tokenizer.eos_token_id]  # <n>
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

    len_samples = len(files)
    cpu_cnt = multiprocessing.cpu_count()
    with multiprocessing.Pool(processes=cpu_cnt) as pool:
        results = pool.starmap(process_data_single, zip([args] * len_samples, files ,[EOS_TOK_IDs]*len_samples))
    output = list(itertools.chain.from_iterable(results))
    print(f"Samples: {len(output)}")
    return output


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

    # plot_single_scatter(all_data_pairs[2], 'cur_pred')
    plot(cnndm_peg_out, xsum_peg_out, cnndm_bart_out, xsum_bart_out)
    # plot_box(all_data_pairs[0], 'last_inp')
    # plot_box(all_data_pairs[1], 'cur_inp')
    # plot_box(all_data_pairs[2], 'cur_pred')
    # plot_box(all_data_pairs[3], 'next_pred')
