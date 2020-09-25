import os, random

from analyze_prob_attn import compute_idf, get_ban_positions
from data_collection import CUR_DIR, PROB_META_DIR, spec_name, MODEL_NAME, DATA_NAME
from util import convert_enc_attn


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


def plot_box(val_ent_pairs, title=None,step_size=.25):
    max_pred_ent = max([p[1] for p in val_ent_pairs])
    # segs = np.linspace(0, max_pred_ent + 0.1, num=20).tolist()
    segs = np.arange(0,8,step_size).tolist()

    bins = [[] for _ in range(len(segs))]
    for p in val_ent_pairs:
        v, ent = p[0], p[1]
        cat = int(ent // 0.25)
        try:
            bins[cat].append(v)
        except:
            pass
    fig1, ax1 = plt.subplots()
    ax1.set_title(title)
    ax1.boxplot(bins)

    plt.show()

def analyze_attention_y_entropy(attn_tlle, pred_ent, input_doc, ban_positions, logits):
    T = attn_tlle.shape[0]
    data_pairs = [[], [], [], []]
    for t in range(T):
        t_pred_ent = pred_ent[t]
        last_inp, cur_inp, cur_pred, next_pred = get_ys(t, logits)
        all_attns_counter = _y_entropy_step(attn_tlle[t], input_doc, ban_positions)
        total_attn_val = sum(all_attns_counter.values())
        last_inp_val = retrieve_tok_val(all_attns_counter, last_inp)
        cur_inp_val = retrieve_tok_val(all_attns_counter, cur_inp)
        cur_pred_val = retrieve_tok_val(all_attns_counter, cur_pred)
        next_pred_val = retrieve_tok_val(all_attns_counter, next_pred)
        data_pairs[0].append((last_inp_val / total_attn_val, t_pred_ent))
        data_pairs[1].append((cur_inp_val / total_attn_val, t_pred_ent))
        data_pairs[2].append((cur_pred_val / total_attn_val, t_pred_ent))
        data_pairs[3].append((next_pred_val / total_attn_val, t_pred_ent))
    return data_pairs


import pickle
import numpy as np

from scipy.stats import entropy

if __name__ == '__main__':
    print("Looking at attention")
    FILE_NUM = 20
    if 'pegasus' in MODEL_NAME:
        from transformers import PegasusTokenizer

        bpe_tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
        EOS_TOK_IDs = [106, bpe_tokenizer.eos_token_id]  # <n>
        BOS_TOK_ID = 0
    else:
        raise NotImplementedError
    files = os.listdir(CUR_DIR)
    random.shuffle(files)
    files = files[:FILE_NUM]

    BOS_TOKEN = 0
    all_data_pairs = [[], [], [], []]
    for f in files:
        with open(os.path.join(CUR_DIR, f), 'rb') as fd:
            data = pickle.load(fd)

        attentions, pred_distb, logits, input_doc = data['attentions'], data['pred_distributions'], data['logits'], \
                                                    data['input_doc']

        timesteps = len(attentions)
        attentions_tlle = convert_enc_attn(attentions, merge_layer_head=False)  # T,L,L,E

        attention_tle = convert_enc_attn(attentions, merge_layer_head=True)  # T,L,E

        document_len = input_doc.shape[0]
        input_doc = input_doc.astype(np.int).tolist()
        logits = logits.tolist()
        dec_inp_logits = [BOS_TOKEN] + logits[:-1]
        pred_distb = np.exp(pred_distb)  # time step, vocab size
        pred_ent = entropy(pred_distb, axis=-1)

        idf_flag = compute_idf(attention_tle)  # E
        ban_positions = get_ban_positions(idf_flag)
        ban_positions = []
        data_pairs = analyze_attention_y_entropy(attentions_tlle, pred_ent, input_doc, ban_positions, logits)
        all_data_pairs[0] += data_pairs[0]
        all_data_pairs[1] += data_pairs[1]
        all_data_pairs[2] += data_pairs[2]
        all_data_pairs[3] += data_pairs[3]

    plot_box(all_data_pairs[0], 'last_inp')
    plot_box(all_data_pairs[1], 'cur_inp')
    plot_box(all_data_pairs[2], 'cur_pred')
    plot_box(all_data_pairs[3], 'next_pred')
