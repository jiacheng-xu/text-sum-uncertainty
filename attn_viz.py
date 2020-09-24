import numpy as np
from typing import List
import scipy
from data_collection import CUR_DIR, PROB_META_DIR, spec_name, MODEL_NAME, DATA_NAME
import os, random

from util import logger


def plot_single_head_attention(attention, src_seq, effective_doc_len, k=5):
    style_underscript = 'color:gray;font-size:10px'
    assert len(attention.shape) == 1
    ent_of_attention = scipy.stats.entropy(attention)
    topk_attention_position = attention.argsort()[-k:][::-1]
    words, attns = [], []
    cumulative_prob = 0
    for index in topk_attention_position:
        index = int(index)
        # if index == t:
        #     _word_ = "|?|"
        # else:
        #     _word_ = bpe_tokenizer.decode(src_seq[int(index)])
        _word_ = bpe_tokenizer.decode(src_seq[int(index)])
        print_word = f"{_word_} <sub style='{style_underscript}'> {int(index)} </sub>"
        _attention_ = attention[int(index)]
        words.append(print_word)
        if int(index) >= effective_doc_len:
            attns.append(-_attention_)
        else:
            attns.append(_attention_)
        cumulative_prob += _attention_
        if cumulative_prob > 0.7:
            break
    out = format_word_importances(words, attns)
    return out, ent_of_attention


def format_special_tokens(token):
    if token.startswith("<") and token.endswith(">"):
        return "#" + token.strip("<>")
    return token


def _get_color(attr):
    # clip values to prevent CSS errors (Values should be from [-1,1])
    attr = max(-1, min(1, attr))
    if attr > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * attr)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * attr)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def format_word_importances(words, importances):
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) <= len(importances)
    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        word = format_special_tokens(word)
        color = _get_color(importance)
        unwrapped_tag = '<mark style="background-color: {color}; opacity:1.0; \
                    line-height:1.75"><font color="black"> {word}\
                    </font></mark>'.format(
            color=color, word=word
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    return "".join(tags)


def plot_attentions(attention: np.ndarray, src_seq, word: str, effective_doc_len):
    # attention: 16, 16, src_len
    num_layer, num_heads, len_att = attention.shape
    trim_src_seq = src_seq[:len_att]
    data_for_timestep = []
    for layer_idx in range(num_layer):
        one_layer_attn = attention[layer_idx]
        row_data = []
        for jdx in range(num_heads):
            single_head_out, single_head_ent = plot_single_head_attention(one_layer_attn[jdx], trim_src_seq,
                                                                          effective_doc_len)
            row_data.append(single_head_out)
        data_for_one_layer = f"<tr><th>{layer_idx}</th>" + "".join(row_data) + "</tr>"
        data_for_timestep.append(data_for_one_layer)
    header_row = "".join([f"<th>{i}</th>" for i in range(num_heads)])
    header_nheads = f"<tr><th>{word}</th>{header_row}</tr>"
    table = f"<table rules='all'>{header_nheads}{''.join(data_for_timestep)}</table>"
    return table


def analyze_pred_dist_single_step(pred_distribution: np.ndarray, k=5):
    ent = scipy.stats.entropy(pred_distribution)
    level_of_ent = int(ent) * 3
    topk_idx = pred_distribution.argsort()[-k:][::-1]
    words, probs = [], []
    decoded_word = bpe_tokenizer.decode(int(topk_idx[0]))
    for index in topk_idx:
        _word_ = bpe_tokenizer.decode(int(index))
        _prob_ = pred_distribution[int(index)]
        words.append(_word_)
        probs.append(_prob_)
    out = format_word_importances(words, probs)

    logger.info(f"<p style='text-indent: {level_of_ent}0px'><strong>{decoded_word}</strong> Ent: {ent}</p>")
    logger.info((f"<p style='text-indent: {level_of_ent}0px'>{out}</p>"))


def meta_analyze_step(t, pred_distribution: np.ndarray,
                      attentions: np.ndarray,
                      input_doc: List[int],
                      dec_summary: List[int], EOS_TOKENs, show_attn: bool = True
                      ):
    word_bpe = dec_summary[t]
    word = bpe_tokenizer.decode(word_bpe)
    if word_bpe in EOS_TOKENs:
        return
    logger.info(f"<h3>Analyzing timestep {t}; Prediction <strong>{word}</strong></h3>")

    analyze_pred_dist_single_step(pred_distribution)
    if show_attn:
        table_in_html = plot_attentions(attentions, input_doc, word, len(input_doc))
        logger.info(table_in_html)
        logger.info("<br><br>")


# TLE: T:decoding timesteps. L:layer and head(16^2 or 12^2), E:encoding document length
def convert_enc_attn(attentions: List, merge_layer_head: bool = True):
    attentions = np.stack([np.stack([np.squeeze(head, axis=1) for head in layer]) for layer in attentions])  # 16,1,E
    if merge_layer_head:
        T, num_layer, num_head, Enc_len = attentions.shape
        A = np.reshape(attentions, (T, num_layer * num_head, Enc_len))
        return A
    else:
        return attentions


import pickle

from scipy.stats import entropy

if __name__ == '__main__':
    print("Looking at attention")

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

    BOS_TOKEN = 0
    for f in files:
        with open(os.path.join(CUR_DIR, f), 'rb') as fd:
            data = pickle.load(fd)
        attentions, pred_distb, logits, input_doc = data['attentions'], data['pred_distributions'], data['logits'], \
                                                    data['input_doc']
        timesteps = len(attentions)
        attentions = convert_enc_attn(attentions, merge_layer_head=False)  # T,L,L,E
        document_len = input_doc.shape[0]
        input_doc = input_doc.astype(np.int).tolist()
        logits = logits.tolist()
        dec_inp_logits = [BOS_TOKEN] + logits[:-1]
        pred_distb = np.exp(pred_distb)  # time step, vocab size
        pred_ent = entropy(pred_distb, axis=-1)
        for t in range(timesteps):
            meta_analyze_step(t, pred_distb[t], attentions[t], input_doc, logits, EOS_TOKENs=EOS_TOK_IDs)
        exit()
