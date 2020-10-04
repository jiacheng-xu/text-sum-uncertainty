import statistics

import os, random, pickle
import numpy as np

# we are going to show for each timestep, for each layer, what's the majority attention.
# marjority attention excluding roadmark tokens
# majority
from typing import List
import scipy

from collections import Counter

from util import convert_enc_attn, logger

index_of_bpe = 1


def attn_layer(inp, combined_inputs, road_mark_positions, top_k=10, min_prob=0.1):
    # inp: 12, 1, length
    nheads = inp.shape[0]
    inp = inp.squeeze()
    rt_row = {}
    stats = {
        'top1': Counter(),
        'top1_distill': Counter(),
        'top3_distill': Counter(),
        'prob': {},
        'empty_slots': 0,
        'total_slots': 0
    }
    # maintain some stat: top1 vote: counter, accum probs, top1 vote exclude roadmark, numOfEmptySlots when exclude roadmark
    for idx in range(nheads):
        rt_cell = []
        nary = inp[idx]
        indicies = nary.argsort()[-top_k:][::-1]
        for rank, jdx in enumerate(indicies):
            prob_val = nary[jdx]
            if prob_val < min_prob:
                break
            bpe = combined_inputs[jdx]
            is_road_mark = True if (jdx in road_mark_positions) or (bpe == 50256) else False
            rt_cell.append(
                [prob_val, bpe, bpe_tokenizer.decode(bpe), is_road_mark, jdx, rank, -1]
            )
            if bpe in stats['prob']:
                stats['prob'][bpe] += float(prob_val)
            else:
                stats['prob'][bpe] = float(prob_val)
        cur_rank = 0
        top1_non_trivial = None
        top3_non_trivial = []
        for kdx, cell in enumerate(rt_cell):
            if cell[3]:
                continue
            rt_cell[kdx][-1] = cur_rank
            cur_rank += 1
            if top1_non_trivial == None:
                top1_non_trivial = cell[index_of_bpe]
            if len(top3_non_trivial) < 3:
                top3_non_trivial.append(cell[index_of_bpe])
        if rt_cell:
            stats['top1'].update([rt_cell[0][index_of_bpe]])
        if top3_non_trivial != []:
            stats['top3_distill'].update(top3_non_trivial)
        if top1_non_trivial != None:
            stats['top1_distill'].update([top1_non_trivial])
        else:
            stats['empty_slots'] += 1
        stats['total_slots'] += 1

        rt_row[idx] = rt_cell
    # rt_row
    prob = stats['prob']
    list_of_prob = list(prob.items())
    norm_prob = [v for (k, v) in list_of_prob]
    s = sum(norm_prob)
    norm_prob = [v / s for v in norm_prob]

    # norm_prob_wo_rm = [ v for (k,v) in list_of_prob if ]
    # s = sum(norm_prob)
    # norm_prob = [ v / s for v in norm_prob]

    return stats


compar_set1 = ['last_inp', 'cur_inp', 'cur_pred', 'next_pred']
compar_set2 = ['top1_most_common', 'top1_distill_most_common']


def analyze_attention_step(attn, cur_t, inp_len, last_inp, cur_inp, cur_pred, next_pred, ent, input_doc,
                           dec_inputs) -> List:
    rts = []
    road_mark_positions = [0] + [idx + inp_len for idx, x in enumerate(dec_inputs) if x == 50256]
    combined_inputs = input_doc + dec_inputs
    attn_ana = [attn_layer(x, combined_inputs, road_mark_positions) for idx, x in enumerate(attn)]
    for idx, layer_ana in enumerate(attn_ana):
        layer_ana['ent'] = float(ent)
        layer_ana['emtpy_rate'] = layer_ana['empty_slots'] / layer_ana['total_slots']
        top1_most_common, top1_distill_most_common = layer_ana['top1'].most_common()[:1], \
                                                     layer_ana['top1_distill'].most_common()[:1]
        top3_distill_top3_common = layer_ana['top3_distill'].most_common()[:3]
        if len(top1_most_common) == 0:
            top1_most_common = None
        else:
            top1_most_common = top1_most_common[0][0]
        if len(top1_distill_most_common) == 0:
            top1_distill_most_common = None
        else:
            top1_distill_most_common = top1_distill_most_common[0][0]
        if len(top3_distill_top3_common) == 0:
            top3_distill_top3_common = []
        else:
            top3_distill_top3_common = [k for (k, v) in top3_distill_top3_common]
        layer_ana['layer'] = idx
        layer_ana['last_inp'] = last_inp
        layer_ana['cur_inp'] = cur_inp
        layer_ana['cur_pred'] = cur_pred
        layer_ana['next_pred'] = next_pred
        layer_ana['top1_most_common'] = top1_most_common
        layer_ana['top1_distill_most_common'] = top1_distill_most_common
        layer_ana['top3_distill_top3_common'] = top3_distill_top3_common
        for keys in compar_set1:
            for leys in compar_set2:
                if layer_ana[keys] and layer_ana[leys]:
                    layer_ana[f"{keys}x{leys}"] = layer_ana[keys] == layer_ana[leys]
                else:
                    layer_ana[f"{keys}x{leys}"] = None
        for keys in compar_set1:
            if layer_ana[keys] and layer_ana['top3_distill_top3_common']:
                layer_ana[f"{keys}xtop3_distill_top3_common"] = layer_ana[keys] in layer_ana['top3_distill_top3_common']
            else:
                layer_ana[f"{keys}xtop3_distill_top3_common"] = None
        rts.append(layer_ana)
    # page empty rate

    return rts


from scipy.stats import entropy


def compute_tf(tle_mat, num_layer=-1):
    T, L, E = tle_mat.shape
    if num_layer == -1:
        sum_over_layer = np.sum(tle_mat, axis=1)  # T, E
        return sum_over_layer
    else:
        return tle_mat[:, num_layer, :]

def get_ban_positions(idf_flag):
    result = np.where(idf_flag == 0)[0].tolist()
    return result


def compute_idf(tle_mat, sparsity=0.95, epsilon=1e-5, num_of_lay=-1):
    # sparsity: <sparsity of cells are counted as 1
    T, L, E = tle_mat.shape
    # return IDF(w_i, T)
    # result = np.zeros((E))
    if num_of_lay == -1:
        sum_over_layer = np.sum(tle_mat, axis=1)  # T, E
        base = np.ones((E)) * T
    else:
        sum_over_layer = tle_mat[:, num_of_lay, :]
        base = np.ones((E))

    prob_threshold = np.quantile(sum_over_layer.flatten(), q=sparsity)
    cnt_flag = np.greater(sum_over_layer, prob_threshold).astype(int)  # T, E
    cnt = np.sum(cnt_flag, axis=0)  # E
    # non_active_positions = np.equal(cnt,0)
    ratio = (epsilon + base) / (epsilon + cnt)
    idf = np.log(ratio)

    # low_thres = np.quantile(idf, 0.05)
    # logger.info(f"Cut threshold: {low_thres}")
    # idf_flag = np.greater(idf, low_thres).astype(int)

    idf_flag = np.greater(idf, np.log(4)).astype(int)
    # high_thres = np.quantile(idf, 0.7)
    # print(f"{high_thres} -- {low_thres}")
    # samples = np.random.choice(idf, 200)
    # for s in samples:
    #     print(s)

    return idf_flag

from scipy.special import softmax

np.set_printoptions(precision=5)


def compute_entropy_for_scores(inp, axis=-1):
    inp = inp + np.min(inp)
    sum_of_all = np.sum(inp, axis=axis)
    dist_inp = inp / sum_of_all

    # dist_inp = softmax(inp)
    ent = entropy(dist_inp)
    return ent


def visualize_tfidf(input_doc, idf, bpe_tokenizer, max_val=12):
    # print("IDF visualization: NUMBER higher = common words")
    idf_list = idf.tolist()
    outputs = []
    L = range(len(input_doc))
    for (idx, inp, val) in zip(L, input_doc, idf_list):
        dec_tok = bpe_tokenizer.decode(inp)
        if inp == bpe_tokenizer.pad_token_id:
            continue
        if val > 0 and val < max_val:
            outputs.append(
                (idx, dec_tok, val / max_val)
            )
        elif val > 0:
            outputs.append(
                (idx, f"_{dec_tok}_", 0)
            )
        else:
            outputs.append(
                (idx, dec_tok, 0)
            )
    return outputs


import matplotlib.pyplot as plt


def draw_plot(data):
    fig, axs = plt.subplots(4, 4, figsize=(10, 10))
    for idx, d in enumerate(data):
        x, y = int(idx / 4), idx % 4
        axs[x, y].scatter(data[idx][0], data[idx][1])
    plt.show()


import matplotlib
import matplotlib.pyplot as plt


def colorize(words, color_array, index_array=None):
    # words is a list of words
    # color_array is an array of numbers between 0 and 1 of length equal to words
    cmap = matplotlib.cm.get_cmap('YlGn')
    style_underscript = 'color:gray;font-size:10px'
    if index_array is None:
        index_array = range(len(words))
    template = '<span class="barcode"; style="color: black; background-color: {}">{}</span>'
    colored_string = ''
    for idx, word, color in zip(index_array, words, color_array):
        color = matplotlib.colors.rgb2hex(cmap(color)[:3])
        print_word = f" {word} <sub style='{style_underscript}'> {idx} </sub> "
        colored_string += template.format(color, '&nbsp' + print_word + '&nbsp')
    return colored_string


def visualize_distribution(input_doc, distb):
    words = 'The quick brown fox jumps over the lazy dog'.split()
    color_array = np.random.rand(len(words))
    s = colorize(words, color_array)

    print(s)


def attention_entrance(attentions: List[List[np.ndarray]], pred_distribution, logits: np.ndarray,
                       input_doc: np.ndarray, BOS_TOKEN, layer_num):
    # print("Example ..")
    timesteps = len(attentions)
    document_len = input_doc.shape[0]
    input_doc = input_doc.astype(np.int).tolist()
    logits = np.argmax(pred_distribution, axis=-1).tolist()
    dec_inp_logits = [BOS_TOKEN] + logits[:-1]
    pred_distribution = np.exp(pred_distribution)  # time step, vocab size
    pred_ent = entropy(pred_distribution, axis=-1)
    all_res = []
    if layer_num == -1:
        A = convert_enc_attn(attentions, True)  # A is the TLE matrix
    else:
        attentions = np.stack(
            np.stack([np.squeeze(head, axis=1) for head in attentions[layer_num]]))
        T, num_head, Enc_len = attentions.shape
        A = np.reshape(attentions, (T, 1 * num_head, Enc_len))

    T, L, E = A.shape
    idf = compute_idf(A)  # E
    """
    
    print("------IDF--------   ")
    visualize_tfidf(input_doc, idf)
    print("------TF IDF--------")
    """
    tf = compute_tf(A)  # T, E
    expand_idf = np.expand_dims(idf, axis=0)

    tfidf = tf * expand_idf
    # tfidf = tf

    for t in range(T):
        if logits[t] == bpe_tokenizer.eos_token_id:
            break
        # print(f"{t} - {bpe_tokenizer.decode(logits[t])}")
        cur_attn_ent = compute_entropy_for_scores(tfidf[t])
        cur_pred_ent = pred_ent[t]
        all_res.append((cur_attn_ent, cur_pred_ent))
        # print(f"{cur_attn_ent}\t{cur_pred_ent}")
        # visualize_tfidf(input_doc, tfidf[t])
    return all_res
    # for t in range(timesteps):
    #     attention = attentions[t]
    #     ent = entropy(pred_distribution[t])
    #
    #     cur_inp = dec_inp_logits[t]
    #     cur_pred = logits[t]
    #     try:
    #         next_pred = logits[t + 1]
    #     except IndexError:
    #         next_pred = None
    #     if t - 1 >= 0:
    #         last_inp = dec_inp_logits[t - 1]
    #     else:
    #         last_inp = None
    #
    #     rt_rs = analyze_attention_step(attention, t, document_len, last_inp, cur_inp, cur_pred, next_pred, ent,
    #                                    input_doc,
    #                                    dec_inp_logits)
    #     all_res += rt_rs
    # return all_res


import json


def run_trial(lay_num, files):
    results = []
    for f in files:
        with open(os.path.join(CUR_DIR, f), 'rb') as fd:
            data = pickle.load(fd)
        result = attention_entrance(data['attentions'], data['pred_distributions'], data['logits'], data['input_doc'],
                                    BOS_TOKEN=bos_token_id, layer_num=lay_num)
        results += result
    result_in_arry = np.asarray(results)
    return result_in_arry.T


if __name__ == '__main__':
    print("Looking at  attention")
    if 'pegasus' in MODEL_NAME:
        from transformers import PegasusTokenizer

        bpe_tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
        EOS_TOK_IDs = [106, bpe_tokenizer.eos_token_id]  # <n>
        bos_token_id = 0
    else:
        raise NotImplementedError
    # visualize_distribution(None,None)
    files = os.listdir(CUR_DIR)
    random.shuffle(files)
    files = files[:20]

    if True:
        all_outputs = []
        for layer_num in range(16):
            print(f"Layer :{layer_num}")
            output_array = run_trial(layer_num, files)
            all_outputs.append(output_array)

        draw_plot(all_outputs)

    exit()
    results = []
    layer_num = 0

    for f in files:
        with open(os.path.join(CUR_DIR, f), 'rb') as fd:
            data = pickle.load(fd)
        result = attention_entrance(data['attentions'], data['pred_distributions'], data['logits'], data['input_doc'],
                                    BOS_TOKEN=bos_token_id, layer_num=layer_num)
        results += result
    result_in_arry = np.asarray(results)
    draw_plot(result_in_arry.T, layer_num)
    # print("Start writing analysis result to disk...")
    # print(len(results))
    # with open(os.path.join(PROB_META_DIR, f"{spec_name}_attention.json"), 'w') as fd:
    #     json.dump(results, fd)
    #     print(f'Done writing to disk: {os.path.join(PROB_META_DIR, f"{spec_name}_attention.json")}')
