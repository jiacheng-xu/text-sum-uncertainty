from datasets import list_datasets
from datasets import load_dataset
import torch
from datasets import load_dataset
from transformers import AutoTokenizer

import logging.config

level = logging.DEBUG
# create logger
logging.basicConfig(filename='example.html', filemode='w')
logger = logging.getLogger('sum')
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(level)

# create formatter
formatter = logging.Formatter('%(asctime)s-%(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger
logger.addHandler(ch)

# Datasets - xsum
dataset_meta = {
    'xsum': {
        'key_doc': 'document',
        'key_sum': 'summary'
    },
    'cnndm': {
        'key_doc': 'article',
        'key_sum': 'highlights'
    }
}

from transformers import BartTokenizer

import random
import string


def format_output(d):
    if 'T_R_1' in d:
        print("T_R_1, 2, L")
        print(f"{d['T_R_1']}\t{d['T_R_2']}\t{d['T_R_L']}\t")

    if 'M_R_1' in d:
        print("M_R_1, 2, L")
        print(f"{d['M_R_1']}\t{d['M_R_2']}\t{d['M_R_L']}\t")


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


import os


def read_file(fname):
    with open(fname, 'r') as fd:
        lines = fd.read().splitlines()
    return "\n".join(lines)


def yield_cnndm(src_dir='/mnt/data0/jcxu/data/cnndm/src', tgt_dir='/mnt/data0/jcxu/data/cnndm/tgt'):
    src_files = os.listdir(src_dir)
    prefix = [x.split(".")[0] for x in src_files if x.startswith('test')]
    random.shuffle(prefix)
    for p in prefix:
        src_article = read_file(os.path.join(src_dir, p + '.src'))
        tgt_sum = read_file(os.path.join(tgt_dir, p + '.tgt'))
        yield (src_article, tgt_sum)


def load_data(dataset_dir, data_name, tokenizer_name='bart-large-cnn',
              batch_size=7, split='test', max_sample_num=34, max_length=500):
    if data_name == 'xsum':
        dataset = load_dataset(data_name, cache_dir=dataset_dir, split=split)
        print("Assume only use one subset of the dataset")
        if len(dataset) > max_sample_num:
            dataset = dataset.shuffle()
    elif data_name == 'cnndm' or data_name == "cnn_dailymail":
        # dataset = load_dataset('cnn_dailymail', '3.0.0', cache_dir=dataset_dir, split=split)
        # import tensorflow_datasets as tfds
        # cnndm_dir = '/mnt/data0/user/data/better_cnndm/formal_data/test'
        dataset = yield_cnndm()
    else:
        raise NotImplementedError("Unkown dataset")

    if 'bart' in tokenizer_name:
        tokenizer = BartTokenizer.from_pretrained(tokenizer_name)
    elif 'gpt' in tokenizer_name:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_name)
    elif 'pegasus' in tokenizer_name:
        from transformers import PegasusTokenizer
        tokenizer = PegasusTokenizer.from_pretrained(tokenizer_name)
        print("Load PEGASUS tokenizer...")
    else:
        raise NotImplementedError

    cur_src_txt, cur_tgt_txt = [], []
    cnt = 0
    for example in dataset:
        if data_name == 'xsum':
            doc = example[dataset_meta[data_name]['key_doc']]
            summary = example[dataset_meta[data_name]['key_sum']]
        elif data_name == 'cnn_dailymail' or data_name == 'cnndm':
            doc, summary = example
        else:
            raise NotImplementedError
        cur_src_txt.append(doc)
        cur_tgt_txt.append(summary)
        if len(cur_src_txt) == batch_size:
            assert len(cur_src_txt) == len(cur_tgt_txt)
            batch = tokenizer.prepare_seq2seq_batch(cur_src_txt, tgt_texts=cur_tgt_txt, max_length=max_length,
                                                    truncation=True, padding='longest')

            yield batch
            cur_src_txt, cur_tgt_txt = [], []
        cnt += 1
        if cnt > max_sample_num:
            break

    # dataset = dataset.map(lambda e: (tokenizer(e[dataset_meta[data_name]['key_doc']]),
    #                                  tokenizer(e[dataset_meta[data_name]['key_sum']])), batched=True)
    # dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    # next(iter(dataloader))


from typing import Union, List
import numpy as np


def auto_detach_to_cpu(inp, dtype=np.float16) -> Union[np.ndarray, List[np.ndarray]]:
    if isinstance(inp, torch.Tensor):
        inp = inp.detach().cpu().numpy().astype(dtype)
        return inp
    tmp = []
    for x in inp:
        x = auto_detach_to_cpu(x)
        tmp.append(x)
    return tmp


def load_PEGASUS(mname):
    from transformers import PegasusTokenizer, PegasusForConditionalGeneration

    model = PegasusForConditionalGeneration.from_pretrained(mname)
    tokenizer = PegasusTokenizer.from_pretrained(mname)
    return model, tokenizer


if __name__ == '__main__':
    data_name = 'cnndm'
    data_name = 'xsum'
    DATASET_DIR = "/mnt/data0/jcxu/datasets"
    # model_name = 'facebook/bart-large-xsum'
    model_name = "google/pegasus-xsum"
    split = 'test'
    load_data(DATASET_DIR, data_name, model_name, split=split)
