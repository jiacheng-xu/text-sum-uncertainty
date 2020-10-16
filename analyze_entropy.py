import numpy
from scipy.stats import entropy

from util import parse_arg


def analyze_pred_dist_single_step(pred_distribution: numpy.ndarray, k=5):
    ent = entropy(pred_distribution)
    level_of_ent = int(ent) * 3
    topk_idx = pred_distribution.argsort()[-k:][::-1]
    words, probs = [], []
    decoded_word = bpe_tokenizer.decode(int(topk_idx[0]))
    for index in topk_idx:
        _word_ = bpe_tokenizer.decode(int(index))
        _prob_ = pred_distribution[int(index)]
        words.append(_word_)
        probs.append(_prob_)
    # out = format_word_importances(words, probs)
    # logger.info(f"<p style='text-indent: {level_of_ent}0px'><strong>{decoded_word}</strong> Ent: {ent}</p>")
    # logger.info((f"<p style='text-indent: {level_of_ent}0px'>{out}</p>"))


import os, pickle, random
import torch
from typing import List
import numpy as np


def comp_entropy(pred_distribution, nucleus_filter: bool = True, top_p=0.95):
    assert np.sum(pred_distribution) > 0.99
    assert np.sum(pred_distribution) < 1.01
    if nucleus_filter:
        empty_pred_distribution = np.zeros_like(pred_distribution)
        sorted_indices = np.argsort(pred_distribution)[::-1].tolist()  # the indices
        sorted_values = np.sort(pred_distribution)[::-1]  # the values
        cumulative_probs = np.cumsum(sorted_values)
        sorted_indices_to_remove = cumulative_probs > top_p  # if the i-th element in sorted_indices_to_remove is 1, it means pred_distribution[sorted_indices[i]] = 0
        sorted_indices_to_remove = sorted_indices_to_remove.tolist()
        sorted_indices_to_remove = [False] + sorted_indices_to_remove[:-1]
        sorted_values = sorted_values.tolist()
        for idx, indi_to_remove in enumerate(sorted_indices_to_remove):
            # if idx == 0:
            #     empty_pred_distribution[sorted_indices[idx]] = pred_distribution[sorted_indices[idx]]
            #     continue
            if not indi_to_remove:
                empty_pred_distribution[sorted_indices[idx]] = pred_distribution[sorted_indices[idx]]
            else:
                break
        empty_pred_distribution = empty_pred_distribution / np.sum(empty_pred_distribution)
        ent = float(entropy(empty_pred_distribution))
    else:
        ent = float(entropy(pred_distribution))
    return ent


def analyze_sentence(logit: List, inp_entropy: List,
                     pred_dist: numpy.ndarray,
                     input_doc, input_bigram, nucleus_filter: bool = True, top_p=0.9) -> List:
    # print(logit)
    viz_outputs = []
    for log, ent in zip(logit, inp_entropy):
        viz_outputs.append("{0}_{1:.1f}".format(bpe_tokenizer.decode(log), ent))
    print(" ".join(viz_outputs))
    cand_bigram = get_bigram(logit)
    l = len(logit)
    rt = []
    return_pos = []
    return_pos.append([0, l, comp_entropy(pred_dist[0], nucleus_filter, top_p)])
    for idx, big in enumerate(cand_bigram):
        t = big[1][0]
        ent = comp_entropy(pred_dist[t], nucleus_filter, top_p)
        tok = big[1][2]
        bigran, trigram = False, False
        if t >= 0:
            _can_big = f"{big[0][2]}_{big[1][2]}"
            if _can_big in input_bigram:
                bigran = True
            else:
                bigran = False

        rt.append(
            [t, l, ent, 0, tok, bigran, trigram]
        )
        return_pos.append([t, l, ent])
    return rt, return_pos


import string

punct = string.punctuation
punct_list = [c for c in punct if c not in ['-']]


def check_if_a_bpe_is_a_token(bpe_id):
    tok = bpe_tokenizer.decode(bpe_id)
    if len(tok) == 0:
        return False
    if tok.startswith(" "):
        return True
    if tok[0] in punct_list:
        return True
    if bpe_id == bpe_tokenizer.bos_token_id or bpe_id == bpe_tokenizer.eos_token_id:
        return True
    if tok[0].isupper():
        return True
    return False


def check_if_bpe_is_a_word(bpe_id):
    tok = bpe_tokenizer.convert_ids_to_tokens(bpe_id)
    if tok.startswith("▁"):
        return True
    if tok[0] in punct_list:
        return True
    return False


def get_bigram(logit_list: List[int]):
    indices = []
    # print(bpe_tokenizer.decode(logit_list))
    for idx, log in enumerate(logit_list):
        tok = bpe_tokenizer.decode(log)
        if isinstance(bpe_tokenizer, BartTokenizer):
            istoken = check_if_a_bpe_is_a_token(log)
        elif isinstance(bpe_tokenizer, GPT2Tokenizer):
            istoken = check_if_a_bpe_is_a_token(log)
        elif isinstance(bpe_tokenizer, PegasusTokenizer):
            istoken = check_if_bpe_is_a_word(log)
            # print(bpe_tokenizer.decode(log), istoken)
        else:
            raise NotImplementedError
        if istoken:
            indices.append(idx)
    last_digit = 0
    indices.pop(0)
    tokens = []
    for indi in indices:
        bpes = logit_list[last_digit:indi]
        tok = bpe_tokenizer.decode(bpes)
        tok = tok.strip()
        tokens.append((last_digit, indi, tok))
        last_digit = indi

    input_bigram = [(tokens[idx], tokens[idx + 1]) for idx in range(len(tokens) - 1)]
    return input_bigram


def analyze_prediction_entropy(logit_list, ent_list, input_doc: numpy.ndarray, eos_tokens=[50256],
                               pred_dist: numpy.ndarray = None, nucleus_filter: bool = True, top_p: float = 0.95):
    # 1) the general entropy distribution of all timesteps. get a sample of high/low entropy word prediction on two datasets.
    # 2) how entropy relates to the relative position of a sentence.
    # 3) characterize the copy/content selection/ EOS or not modes.
    # 4) does some part of hidden states indicate
    assert sum(pred_dist[0]) > 0.99
    assert sum(pred_dist[0]) < 1.01
    # logits = pred_dist.argmax(axis=-1)
    # logit_list = logits.tolist()
    # print(logit_list)
    # ent_list = ent.tolist()
    input_doc = input_doc.tolist()
    # for x in logit_list:
    #     print(f"{x} - {bpe_tokenizer.convert_ids_to_tokens(x)}")
    # print(f"Processing: Summary: {bpe_tokenizer.convert_ids_to_tokens(logit_list)}\n{bpe_tokenizer.decode(input_doc)}")
    input_bigram = get_bigram(input_doc)
    input_bigram = [f"{big[0][2]}_{big[1][2]}" for big in input_bigram]
    print(f"Bigram like {input_bigram[0]}")
    # input_bigram = [(input_doc[idx], input_doc[idx + 1]) for idx in range(len(input_doc) - 1)]
    # input_trigram = [(input_doc[idx], input_doc[idx + 1], input_doc[idx + 2]) for idx in range(len(input_doc) - 2)]
    # record sentence boundary
    indices = [i for i, x in enumerate(logit_list) if x in eos_tokens]
    outputs = []
    outputs_pos = []
    last_indi = 0
    print(f"Decode: {bpe_tokenizer.decode(logit_list)}")
    for indi in indices:
        indi = indi + 1
        if indi - last_indi < 3:
            break
        output, output_pos = analyze_sentence(logit_list[last_indi:indi],
                                              ent_list[last_indi:indi],
                                              pred_dist[last_indi:indi],
                                              input_doc, input_bigram,
                                              nucleus_filter=nucleus_filter, top_p=top_p)
        outputs += output
        outputs_pos += output_pos
        last_indi = indi
    return outputs, outputs_pos


import json

if __name__ == '__main__':
    print("Look at the entropy")
    # from data_collection import CUR_DIR, PROB_META_DIR, spec_name, MODEL_NAME

    args = parse_arg()

    model_files = os.listdir(args.cur_dir)

    random.shuffle(model_files)
    model_files = model_files[:args.max_sample_num]
    print(f"total len of files: {len(model_files)}")
    entropies = []
    max_probs = []
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
    try:
        outputs = []
        outputs_pos_entropy = []
        for f in model_files:
            with open(os.path.join(args.cur_dir, f), 'rb') as fd:
                data = pickle.load(fd)
            print(f"Finish loading {f}")
            try:
                pred_dist = [x.unsqueeze(0) for x in data['pred_distributions']]
                new_pred_dist = torch.cat(pred_dist, dim=0)
                pred_dist = new_pred_dist.numpy()  # shape: 90, 50257
                pred_real_dist = numpy.exp(pred_dist)
                input_doc = data['input_doc'].numpy()
                input_doc_mask = data['input_doc_mask'].numpy()
                effective_input_len = int(input_doc_mask.sum())
                input_doc = input_doc[:effective_input_len]
            except:
                logits = data['logits']
                if 'ent' in data:
                    ent = data['ent']
                else:
                    ent = None
                pred_real_dist = data['pred_distributions']
                pred_real_dist = numpy.exp(pred_real_dist)

                input_doc = data['input_doc']
                input_doc_mask = data['input_doc_mask']
                effective_input_len = int(input_doc_mask.sum())
                input_doc = input_doc[:effective_input_len]

            logits = logits.tolist()
            ent = ent.tolist()
            # BART remove SOS
            if 'bart' in args.model_name:
                bos = bpe_tokenizer.bos_token_id
                trimmed_logits, trimmed_ent, trimmed_pred_real_dist = [], [], []
                for idx, logi in enumerate(logits):
                    if logi == bos:
                        continue
                    trimmed_logits.append(logi)
                    trimmed_ent.append(ent[idx])
                    trimmed_pred_real_dist.append(pred_real_dist[idx])
                    if logi == bpe_tokenizer.eos_token_id:
                        break
                if len(trimmed_pred_real_dist) > 5:
                    trimmed_pred_real_dist = np.stack(trimmed_pred_real_dist, axis=0)
                else:
                    continue
                logits, ent, pred_real_dist = trimmed_logits, trimmed_ent, trimmed_pred_real_dist
            out, out_pos = analyze_prediction_entropy(logits, ent, input_doc, EOS_TOK_IDs, pred_real_dist,
                                                      nucleus_filter=args.nucleus, top_p=args.nuc_prob)
            outputs += out
            outputs_pos_entropy += out_pos
    except KeyboardInterrupt:
        print("interrupted")
    print(f"Entropy data in .json in {args.prob_meta_dir}")
    print(f"Nuc: {args.nucleus}")
    if args.nucleus:
        nc_label = str(args.nuc_prob)
    else:
        nc_label = ""
    print(f"writing Bigram entropy to {args.spec_name}{nc_label}_entropy.json")
    s = json.dumps(outputs)
    with open(os.path.join(args.prob_meta_dir, f"{args.spec_name}{nc_label}_entropy.json"), 'w') as fd:
        fd.write(s)

    print(f"writing position entropy to {args.spec_name}{nc_label}_pos_entropy.json")
    dump_outputs_pos_entropy = json.dumps(outputs_pos_entropy)
    with open(os.path.join(args.prob_meta_dir, f"{args.spec_name}{nc_label}_pos_entropy.json"), 'w') as fd:
        fd.write(dump_outputs_pos_entropy)
