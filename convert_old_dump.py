import pickle
import os
import torch
import numpy as np
import scipy

from scipy.stats import entropy


def convert_to_mini_format(data_pack):
    rt = {}
    rt['logits'] = data_pack['logits'].astype(int)
    real_pred_dist = np.exp(data_pack['pred_distributions'])
    ent = entropy(real_pred_dist, axis=-1)
    rt['ent'] = ent
    rt['input_doc'] = data_pack['input_doc']
    rt['input_doc_mask'] = data_pack['input_doc_mask']
    rt['meta'] = data_pack['meta']
    return rt


if __name__ == '__main__':
    dir = '/mnt/data0/jcxu/data/prob_gpt/new/xsum'
    tgt_dir = '/mnt/data0/jcxu/data/prob_gpt/new/xsum-mini'
    # os.mkdir(tgt_dir)
    files = os.listdir(dir)
    for f in files:
        fpath = os.path.join(dir, f)
        with open(fpath, 'rb') as fd:
            data = pickle.load(fd)
        new_data_format = convert_to_mini_format(data)
        with open(os.path.join(tgt_dir, f), 'wb') as wfd:
            pickle.dump(new_data_format, wfd)
    print('done')
