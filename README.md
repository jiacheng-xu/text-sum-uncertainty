# Text Sum Uncertainty

**Code for "Understanding Neural Abstractive Summarization Models via Uncertainty" (EMNLP20, short)**

ArXiv preprint available at [here](https://arxiv.org/abs/2010.07882). 

## [Slide Deck](https://github.com/jiacheng-xu/text-sum-uncertainty/blob/master/slide.pdf)

Author: [Jiacheng Xu](https://www.cs.utexas.edu/~jcxu/), [Shrey Desai](https://shreydesai.github.io/), [Greg Durrett](https://www.cs.utexas.edu/~gdurrett/) from [TAUR Lab](http://taur.cs.utexas.edu/), UT Austin

Contact: jcxu at utexas.edu
## About
 In this work, 
 - We analyze summarization decoders by studying on the **entropy**, or uncertainty, of the model's token-level predictions.
 - Models examined: PEGASUS([paper](https://arxiv.org/abs/1912.08777), [model](https://huggingface.co/transformers/model_doc/pegasus.html#pegasusforconditionalgeneration)) and 
 BART([paper](https://arxiv.org/abs/1910.13461),[model](https://huggingface.co/transformers/model_doc/bart.html#bartforconditionalgeneration))
 - Datasets covered: CNN/DM and XSum
 - Quick start with models directly from [huggingface.co/transformers](https://github.com/huggingface/transformers) 
 
With the help of the methods we developed, we further investigate
- Correlation between prediction entropy & model behavior like COPY or GEN (Sec. 3)
- Sentence position connects to prediction entropy (Sec. 3) 
- Model behavior in different syntactic environments (Sec. 4)
- Coarse properties of attention and the how that correlates with model's prediction (Sec. 5) 


## Configuration

#### Hyper Parameters
In [`util.py`](https://github.com/jiacheng-xu/text-sum-uncertainty/blob/master/util.py), the function `parse_arg` defines all of the hyper-params used in this project.
 
| Param | Usage |
| ----------- | ----------- |
| prob_meta_dir | The location you save the model outputs. |
|max_len| Max decoding length. Set to 30 for XSum and 80 for CNN/DM. |
|device | Device name for Pytorch.|
|nuc_prob| Nucleus sampling prob threshold. Default: 0.95. |
|trunc_prob | Truncate the probability distribution (by default used in all of our experiments).|
|full_prob| Use the original probability distribution. |


#### Existing Configuration
To run the model, simply run `python run_model_pegasus.py` with one of the following parameter configuration. 

| Config Name | Parameters |
| ----------- | ----------- |
| run_model_pegasus_cnndm | --full_data |
| run_model_pegasus_xsum | --full_data --model_name google/pegasus-xsum --data_name xsum |
|run_model_bart_cnndm | --full_data --model_name facebook/bart-large-cnn |
|run_model_bart_xsum | --full_data --model_name facebook/bart-large-xsum --data_name xsum|

class `SumGen` in `run_model_pegasus.py` is the core decoding part. 