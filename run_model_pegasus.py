import torch

from transformers import BatchEncoding, PreTrainedTokenizer
from typing import Optional, List
from argparse import Namespace

from data_collection import DataCollector
from util import load_PEGASUS, load_data
import logging


class SumGen(torch.nn.Module):
    def __init__(self, model, tokenizer: PreTrainedTokenizer,
                 use_cache=True, max_len=30, full_data=False):
        super().__init__()
        self.model = model
        self.output_attentions = True
        self.output_hidden_states = True

        self.tokenizer = tokenizer
        self.max_len = max_len
        self.use_cache = use_cache
        self.encoder = self.model.get_encoder()
        self.recorder = DataCollector(full_data=full_data)
        self.logsoftmax = torch.nn.LogSoftmax(-1)

    def save_data(self):
        pass

    def forward(self, input_doc, input_mask, tgt_sum=None):

        device = input_doc.device
        batch_size = input_doc.shape[0]
        seq_length = input_doc.shape[1]

        cur_len = 1
        has_eos = [False for _ in range(batch_size)]

        bos_token_id = 0
        decoded = [[bos_token_id] for _ in range(batch_size)]
        decoder_input_ids = torch.LongTensor(decoded).to(device)
        past_key_values = None
        encoder_outputs = self.encoder(input_doc, attention_mask=input_mask,
                                       return_dict=True)

        expanded_batch_idxs = (
            torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, 1)
                .view(-1)
                .to(device)
        )
        encoder_outputs["last_hidden_state"] = encoder_outputs.last_hidden_state.index_select(
            0, expanded_batch_idxs
        )
        self.recorder.add_input_doc(input_doc, input_mask)
        while cur_len < self.max_len and (not all(has_eos)):
            logging.debug(f"Current step: {cur_len}")
            cur_decoded, cur_past_key_values, cur_decoder_input_ids = self.forward_step(attn_mask=input_mask,
                                                                                        past_key_values=past_key_values,
                                                                                        decoder_input_ids=decoder_input_ids,
                                                                                        encoder_outputs=encoder_outputs
                                                                                        )
            # print('run normal')
            # cur_decoded is just a list with token id
            for idx, cur_dec_tok in enumerate(cur_decoded):
                if cur_dec_tok == self.tokenizer.eos_token_id:
                    has_eos[idx] = True
            if tgt_sum is None:
                # target = cur_decoder_input_ids[:, -1].unsqueeze(0)
                target = cur_decoded[0][0]  # assume batch size = 1
                print(f'target : {target}')
            else:
                pass
            past_key_values = cur_past_key_values
            decoder_input_ids = cur_decoder_input_ids
            cur_len += 1
        self.recorder.write_to_disk_numpy()
        logging.info("end of decoding")

    def forward_step(self,
                     attn_mask, past_key_values, decoder_input_ids, encoder_outputs,
                     ):
        """The forward pass for one single time step

        """

        model_inputs = {"input_ids": None,
                        "past_key_values": past_key_values,
                        "attention_mask": attn_mask,
                        "encoder_outputs": encoder_outputs,
                        "decoder_input_ids": decoder_input_ids,
                        }
        outputs = self.model.forward(**model_inputs,
                                     output_attentions=self.output_attentions,
                                     output_hidden_states=self.output_hidden_states,
                                     use_cache=self.use_cache,
                                     return_dict=True)
        if 'decoder_hidden_states' in outputs:
            pass

        if 'decoder_attentions' in outputs:
            pass

        if 'logits' in outputs:
            # logits are raw score before softmax
            logit = self.logsoftmax(outputs['logits'][:, -1, :])
            self.recorder.add_step(pred_distribution=logit)

        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)
        self.recorder.add_logit(next_token)
        cur_next_token = next_token.unsqueeze(-1)
        cur_decoded = cur_next_token.tolist()
        if "past_key_values" in outputs:
            past_key_values = outputs.past_key_values

        decoder_input_ids = torch.cat([decoder_input_ids, cur_next_token], dim=-1)

        return cur_decoded, past_key_values, decoder_input_ids


if __name__ == '__main__':
    print("Running experiment")
    from data_collection import full_data, MODEL_NAME, DATA_NAME, CUR_DIR

    max_sample_num = 1770
    batch_size = 10
    max_len = 35 if DATA_NAME == 'xsum' else 100
    DATASET_DIR = "/mnt/data0/jcxu/datasets"

    split = 'test'

    data_generator = load_data(DATASET_DIR, DATA_NAME,
                               tokenizer_name=MODEL_NAME,
                               split=split,
                               batch_size=batch_size,
                               max_length=300, max_sample_num=max_sample_num)
    model, tokenizer = load_PEGASUS(MODEL_NAME)
    device = torch.device('cuda:0')
    model = model.to(device)
    summary_gen_model = SumGen(model=model, tokenizer=tokenizer, full_data=full_data, max_len=max_len)
    for batch in data_generator:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        tgt = batch['labels'].to(device)
        summary_gen_model.forward(input_ids, attention_mask, tgt)
