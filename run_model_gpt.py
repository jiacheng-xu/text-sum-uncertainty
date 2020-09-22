from util import *

import logging
from allennlp.models import load_archive
from transum.configs.config_env import bpe_tokenizer, automatic_determine_dir, logger, flatten
from allennlp.predictors.predictor import Predictor
from transum.data.dataset_reader import FullSummarizationDatasetReader
from transum.prob.data_handler import data_feeder
from transum.modeling_mask_gpt import MaskGPT2Model

from allennlp.data.instance import Instance


class ModelProtocol():
    def __init__(self,
                 # num_data_points=2000,
                 # data_path='/mnt/data0/user/data/better_cnndm/formal_data/test',
                 archive='/mnt/data0/user/fantastic-template/tmp_expst6xy4zu_',
                 cuda_device=0,
                 trigram_blocking: bool = False,
                 batch_sz=50):
        # archive_name = '/mnt/data0/user/fantastic-template/tmp_expst6xy4zu_'  # vanilla GPT model

        f_handler = logging.FileHandler(f"eval_output_{archive.split('/')[-1]}.log")
        f_handler.setLevel(logging.DEBUG)
        logger.addHandler(f_handler)

        archive = load_archive(archive, cuda_device=cuda_device)
        archive.config.params['validation_iterator']['batch_size'] = batch_sz
        archive.model.custom_generator.trigram_blocking = trigram_blocking
        # archive.config.params['model']['decoding_timestep_min'] = 30
        # archive.config.params['model']['decoding_timestep_max'] = 80
        # archive.config.params['model']['decoding_timestep_seg'] = 6

        print(f"configuration: {archive.config.params}")
        predictor = Predictor.from_archive(
            archive, "sum_full", dataset_reader_to_load="validation"
        )
        self.predictor = predictor

        # metrics = setup_metrics()
        # predictor._model.custom_generator.trigram_blocking = False
        predictor._model.custom_generator.trigram_blocking = True
        # predictor._model.custom_generator.repetition_penalty = 3.0
        # predictor._model.custom_generator.top_k_sampling = 5
        # predictor._model.custom_generator.top_p_sampling = 0.3
        """
        dataset_reader_param = archive.config.params['dataset_reader']
        use_sent = dataset_reader_param['use_sent_mask']
        use_span = dataset_reader_param['use_span_mask']
        vanilla_gpt = dataset_reader_param['vanilla_gpt']

        instances = data_feeder(data_path, num_data_points)
        """

    def forward(self, instances):
        output = self.predictor.predict_batch_instance(instances)


if __name__ == '__main__':
    logger.info(f"Run the model (GPT)")
    batchsz = 20
    num_data_points = 200
    model = ModelProtocol(batch_sz=batchsz, archive=model_to_use)
    data_instances: List[Instance] = data_feeder(num_data_points=num_data_points, data_path=data_to_use)

    cur_t = 0
    total_len_instances = len(data_instances)
    all_outputs = []
    try:
        while cur_t < total_len_instances:
            outputs = model.predictor.predict_batch_instance(data_instances[cur_t:cur_t + batchsz])
            all_outputs += outputs
            cur_t += batchsz
    except KeyboardInterrupt:
        print("interrupted")
    # all_outputs = predictor.predict_batch_instance(instances)

    # outputs = [output['output'] for output in all_outputs]
    # add_output_to_metrics(metrics, outputs)
    out = model.predictor._model.get_metrics(reset=True)
    format_output(out)
    logger.info(out)
