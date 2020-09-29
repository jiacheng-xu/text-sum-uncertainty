import numpy, scipy

from util import parse_arg


def analyze_pred_dist_single_step(pred_distribution: numpy.ndarray, k=5):
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
    # out = format_word_importances(words, probs)
    # logger.info(f"<p style='text-indent: {level_of_ent}0px'><strong>{decoded_word}</strong> Ent: {ent}</p>")
    # logger.info((f"<p style='text-indent: {level_of_ent}0px'>{out}</p>"))


import os, pickle, random
import torch
from typing import List


def analyze_sentence(logit: List, entropy: List,
                     # pred_dist: numpy.ndarray,
                     input_doc, input_bigram, input_trigram) -> List:
    # print(logit)
    viz_outputs = []
    for log, ent in zip(logit, entropy):
        viz_outputs.append("{0}_{1:.1f}".format(bpe_tokenizer.decode(log), ent))
    print(" ".join(viz_outputs))
    cand_bigram = get_bigram(logit)
    # tokens = [bpe_tokenizer.decode(x) for x in logit]
    l = len(logit)
    rt = []
    for idx, big in enumerate(cand_bigram):
        t = big[1][0]
        # max_prob = float(pred_dist[t].max())
        # ent = float(scipy.stats.entropy(pred_dist[t]))
        ent = entropy[t]
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

    return rt


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
    return False


def check_if_bpe_is_a_word(bpe_id):
    tok = bpe_tokenizer.convert_ids_to_tokens(bpe_id)
    if tok.startswith("▁"):
        return True
    if tok[0] in punct_list:
        return True
    return False


from transformers import GPT2Tokenizer


def get_bigram(logit_list: List[int]):
    indices = []
    for idx, log in enumerate(logit_list):
        # tok = bpe_tokenizer.decode(log)
        if isinstance(bpe_tokenizer, GPT2Tokenizer):
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


def analyze_prediction_entropy(logits, ent, input_doc: numpy.ndarray, eos_tokens=[50256],
                               pred_dist: numpy.ndarray = None):
    # 1) the general entropy distribution of all timesteps. get a sample of high/low entropy word prediction on two datasets.
    # 2) how entropy relates to the relative position of a sentence.
    # 3) characterize the copy/content selection/ EOS or not modes.
    # 4) does some part of hidden states indicate

    # logits = pred_dist.argmax(axis=-1)
    logit_list = logits.tolist()
    # print(logit_list)
    ent_list = ent.tolist()
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
    last_indi = 0
    for indi in indices:
        indi = indi + 1
        if indi - last_indi < 3:
            break
        output = analyze_sentence(logit_list[last_indi:indi],
                                  ent_list[last_indi:indi],
                                  # pred_dist[last_indi:indi],
                                  input_doc, input_bigram,
                                  input_bigram)
        outputs += output
        last_indi = indi
    return outputs


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
        EOS_TOK_IDs = [106, bpe_tokenizer.eos_token_id]  # <n>
    else:
        raise NotImplementedError
    try:
        outputs = []
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
                ent = data['ent']
                # pred_real_dist = data['pred_distributions']
                # pred_real_dist = numpy.exp(pred_real_dist)
                input_doc = data['input_doc']
                input_doc_mask = data['input_doc_mask']
                effective_input_len = int(input_doc_mask.sum())
                input_doc = input_doc[:effective_input_len]
            out = analyze_prediction_entropy(logits, ent, input_doc, EOS_TOK_IDs)
            outputs += out
    except KeyboardInterrupt:
        print("interrupted")
    print(f"Entropy data in .json in {args.prob_meta_dir}")

    s = json.dumps(outputs)
    print(f"writing to {args.spec_name}_entropy.json")
    with open(os.path.join(args.prob_meta_dir, f"{args.spec_name}_entropy.json"), 'w') as fd:
        fd.write(s)
