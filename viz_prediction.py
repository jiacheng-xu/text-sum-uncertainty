import os
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description="Experiment for Text Sum Uncertainty for PEGASUS model.")
    parser.add_argument("--model_name", type=str,
                        help="The name of the PEGASUS model. Expect: google/pegasus-cnn_dailymail or google/pegasus-xsum.",
                        default='google/pegasus-cnn_dailymail')
    parser.add_argument("--data_name", help="Name of the dataset to use, xsum or cnn_dailymail.",
                        default='cnn_dailymail')
    parser.add_argument('--full_data', dest='feature', action='store_true')
    parser.add_argument('--min_data', dest='feature', action='store_false')
    parser.set_defaults(feature=True)
    parser.add_argument('--prob_meta_dir',
                        default='/mnt/data0/jcxu/data/prob_gpt',
                        help='Location to store outputs files.')
    parser.add_argument('--topk', default=5, help='Show the topk')
    args = parser.parse_args()

    spec_name = f"d_{args.data_name}-m_{args.model_name[-5:]}-full{int(args.feature)}"
    CUR_DIR = os.path.join(args.prob_meta_dir, spec_name)
    print(f"======= {CUR_DIR} =======")
    args.cur_dir = CUR_DIR
    args.spec_name = spec_name
    return args


import random
import pickle

from scipy.stats import entropy
import numpy as np


def viz_pred(tokenizer, pred_dist, input_doc, eos_tok, topk=5):
    print(f"Input Doc: {tokenizer.decode(input_doc[:400])}")
    T, vocab_sz = pred_dist.shape
    ent = entropy(pred_dist, axis=-1)
    eos_flag = False
    for t in range(T):
        cur_pred_distb = pred_dist[t]
        assert sum(cur_pred_distb)>0.99
        cur_ent = ent[t]

        indices = np.argsort(cur_pred_distb)[::-1].tolist()
        pred_tok = indices[0]
        if pred_tok in eos_tok:
            if eos_flag:
                break
            else:
                eos_flag = True
        else:
            eos_flag = False
        cand_pairs = []
        for k in range(topk):
            index = indices[k]
            token = tokenizer.decode(index)
            prob_val = cur_pred_distb[index]
            cand_pairs.append((prob_val, token))
        print_topk = "\t".join(['{:0.2f} {:>10}'.format(cand[0], cand[1]) for cand in cand_pairs])
        print('Ent: {:0.2f}\t{}'.format(cur_ent, print_topk)
              )


if __name__ == '__main__':
    args = parse_arg()

    fdir = os.path.join(args.cur_dir)
    files = os.listdir(fdir
                       )
    from transformers import PegasusTokenizer

    bpe_tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    EOS_TOK_IDs = [106, bpe_tokenizer.eos_token_id]  # <n>
    [print(bpe_tokenizer.decode(x)) for x in EOS_TOK_IDs]
    bos_token_id = 0
    random.shuffle(files)
    for f in files:
        with open(os.path.join(fdir, f), 'rb') as fd:
            data = pickle.load(fd)
        viz_pred(bpe_tokenizer, np.exp(data['pred_distributions']), data['input_doc'], topk=args.topk,
                 eos_tok=EOS_TOK_IDs)
