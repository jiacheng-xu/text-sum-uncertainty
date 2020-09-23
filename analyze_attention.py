import statistics

from data_collection import CUR_DIR, PROB_META_DIR, spec_name, MODEL_NAME, DATA_NAME


def open_data():
    pass


import os, random, pickle
import numpy as np

# we are going to show for each timestep, for each layer, what's the majority attention.
# marjority attention excluding roadmark tokens
# majority
from typing import List
import scipy

from collections import Counter

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

def attention_entrance(attentions: List[List[np.ndarray]], pred_distribution, logits: np.ndarray,
                       input_doc: np.ndarray, BOS_TOKEN):
    print("Example ..")
    timesteps = len(attentions)
    document_len = input_doc.shape[0]
    input_doc = input_doc.astype(np.int).tolist()
    logits = np.argmax(pred_distribution, axis=-1).tolist()
    dec_inp_logits = [BOS_TOKEN] + logits[:-1]
    pred_distribution = np.exp(pred_distribution)
    all_res = []
    for t in range(timesteps):
        attention = attentions[t]
        ent = entropy(pred_distribution[t])

        cur_inp = dec_inp_logits[t]
        cur_pred = logits[t]
        try:
            next_pred = logits[t + 1]
        except IndexError:
            next_pred = None
        if t - 1 >= 0:
            last_inp = dec_inp_logits[t - 1]
        else:
            last_inp = None

        rt_rs = analyze_attention_step(attention, t, document_len, last_inp, cur_inp, cur_pred, next_pred, ent,
                                       input_doc,
                                       dec_inp_logits)
        all_res += rt_rs
    return all_res


import json

if __name__ == '__main__':
    print("Looking at attention")

    if 'pegasus' in MODEL_NAME:
        from transformers import PegasusTokenizer

        bpe_tokenizer = PegasusTokenizer.from_pretrained(MODEL_NAME)
        EOS_TOK_IDs = [106, bpe_tokenizer.eos_token_id]  # <n>
        bos_token_id = 0
    else:
        raise NotImplementedError

    files = os.listdir(CUR_DIR)
    random.shuffle(files)
    # files = files[:20]
    results = []
    for f in files:
        with open(os.path.join(CUR_DIR, f), 'rb') as fd:
            data = pickle.load(fd)
        result = attention_entrance(data['attentions'], data['pred_distributions'], data['logits'], data['input_doc'], BOS_TOKEN=bos_token_id)
        results += result

    print("Start writing analysis result to disk...")
    print(len(results))
    with open(os.path.join(PROB_META_DIR, f"{spec_name}_attention.json"), 'w') as fd:
        json.dump(results, fd)
        print(f'Done writing to disk: {os.path.join(PROB_META_DIR, f"{spec_name}_attention.json")}')
