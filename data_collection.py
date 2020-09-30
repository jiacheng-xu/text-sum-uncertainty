import os
import logging

from util import auto_detach_to_cpu, get_random_string
import numpy as np
import pickle

#
# # full_data = True  # shows everything include attention and hidden states
# full_data = False
#
# MODEL_NAME = "google/pegasus-cnn_dailymail"  # "google/pegasus-xsum"
# # MODEL_NAME = "google/pegasus-xsum"
#
# DATA_NAME = "cnn_dailymail"
# # DATA_NAME = 'xsum'
#
# PROB_META_DIR = '/mnt/data0/jcxu/data/prob_gpt'
# spec_name = f"d_{DATA_NAME}-m_{MODEL_NAME[-5:]}-full{int(full_data)}"
#
# CUR_DIR = os.path.join(PROB_META_DIR, spec_name)
# if not os.path.isdir(CUR_DIR):
#     os.mkdir(CUR_DIR)
#     logging.info(f"Making {CUR_DIR}")
#     print(f"======= {CUR_DIR} =======")

"""
if data_name == 'xsum':

    xsum_model = '/backup3/jcxu/fantastic-template/tmp_exps951s6lxj'
    xsum_path = '/backup3/jcxu/data/xsum/formal_data/test'

    # PROB_META_DIR = '/mnt/data0/jcxu/data/prob_gpt'
    # PROB_META_DIR = '/backup3/jcxu/data/prob_gpt'
    PROB_META_DIR = '/mnt/data0/jcxu/data/prob_gpt'

    model_to_use = xsum_model
    data_to_use = xsum_path

elif data_name == 'cnndm':

    cd_model = '/mnt/data0/jcxu/fantastic-template/tmp_expst6xy4zu_'
    cd_model = '/mnt/data0/jcxu/fantastic-template/tmp_sync/tmp_expsnvmn9u2u'

    cd_path = '/mnt/data0/jcxu/data/better_cnndm/formal_data/test'
    PROB_META_DIR = '/mnt/data0/jcxu/data/prob_gpt'
    # PROB_META_DIR = '/backup3/jcxu/data/prob_gpt'

    model_to_use = cd_model
    data_to_use = cd_path
else:
    raise NotImplementedError
"""

from scipy.stats import entropy


class DataCollector():
    def __init__(self, full_data: bool, cur_dir):
        self.pred_distributions = []
        self.attentions = []
        self.all_hidden_states = []
        self.logits = []
        self.input_doc = None
        self.input_doc_mask = None
        self.meta = None
        self.full_data = full_data
        self.cur_dir = cur_dir

    def add_meta(self, meta):
        self.meta = meta

    def add_input_doc(self, input_doc, input_doc_msk):
        self.input_doc = auto_detach_to_cpu(input_doc, dtype=np.int)
        self.input_doc_mask = auto_detach_to_cpu(input_doc_msk)

    def add_step(self, pred_distribution, all_hidden_states=None, attentions=None):
        self.pred_distributions.append(auto_detach_to_cpu(pred_distribution, dtype=np.float32))
        if self.full_data:
            self.all_hidden_states.append(auto_detach_to_cpu(all_hidden_states))
            self.attentions.append(auto_detach_to_cpu(attentions))

    def add_logit(self, logit):
        self.logits.append(auto_detach_to_cpu(logit, dtype=np.int))

    def write_to_disk_numpy(self):
        batchsz = self.input_doc.shape[0]
        for i in range(batchsz):
            _pred_dist = [x[i] for x in self.pred_distributions]
            _pred_dist = np.stack(_pred_dist, axis=0)
            ent = entropy(np.exp(_pred_dist), axis=-1)
            if self.full_data:
                # _hidden_states = [[y[i][np.newaxis, ...] for y in x] for x in self.all_hidden_states]
                # _hidden_states = [np.stack(x, axis=1) for x in self.all_hidden_states]
                # _hidden_states = np.stack(_hidden_states, axis=0)
                _hidden_states = None
                _attn = [[y[i] for y in x] for x in self.attentions]
            else:
                _hidden_states = None
                _attn = None
                _pred_dist = None

            # _attn = np.stack(_attn, axis=0)
            _logit = [x[i] for x in self.logits]
            _logit = np.stack(_logit, axis=0)
            if self.meta:
                _meta = self.meta[i]
                if 'name' in _meta:
                    fname = _meta['name']
                else:
                    fname = _meta['id']
            else:
                _meta = {"name": "", "id": ""}
                fname = get_random_string(8)
            f = f"model_output_{fname}.pt"

            with open(os.path.join(self.cur_dir, f), 'wb') as fd:
                pickle.dump(
                    {'pred_distributions': _pred_dist,
                     'attentions': _attn,
                     'all_hidden_states': _hidden_states,
                     'logits': _logit,
                     'input_doc': self.input_doc[i],
                     'input_doc_mask': self.input_doc_mask[i],
                     'meta': _meta,
                     'ent': ent

                     }, fd
                )
            logging.debug(f"writing {os.path.join(self.cur_dir, f)}")
            print(f"writing {os.path.join(self.cur_dir, f)}")
        self.__init__(full_data=self.full_data, cur_dir=self.cur_dir)
