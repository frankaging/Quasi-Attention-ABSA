# Quasi-Attention-ABSA
Codebase for Context-Guided BERT for Targeted Aspect-Based Sentiment Analysis (AAAI2021)

## Contents

* [Citation](#Citation)
* [Quick start](#quick-start)
* [License](#license)

## Citation
```
@inproceedings{wu2020context,
  title={Context-Guided BERT for Targeted Aspect-Based Sentiment Analysis},
  author={Wu, Zhengxuan and Ong, Desmond C},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

## Quick start
### Download Pretrained BERT Model
You will have to download pretrained BERT model in order to execute the fine-tune pipeline. We recommand to use models provided by the official release on BERT from [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert). Note that their model is in tensorflow format. To convert tensorflow model to pytorch model, you can use the helper script to do that. For example,
```bash
cd code/
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```

### Datasets
We already preprocess the datasets for you. To be able to compare with the SOTA models,
we adapt the preprocess pipeline right from this previous [repo](https://github.com/HSLCY/ABSA-BERT-pair) where SOTA models are trained. To regenerate the dataset, please refer
to their paper and generate. Please also consider to cite their paper for this process.

### Train CG-BERT Model and QACG-BERT Models
Our (T)ABSA BERT models are adapted from [huggingface](https://github.com/huggingface/transformers) BERT model for text classification. If you want to take a look at the original model please search for [BertForSequenceClassification](https://github.com/huggingface/transformers/blob/master/src/transformers/modeling_bert.py). To train QACG-BERT model with semeval2014 dataset on GPU 0 and 1, you can do something like this,
```bash
cd code/
CUDA_VISIBLE_DEVICES=0,1 python run_classifier.py \
--task_name semeval_NLI_M \
--data_dir ../datasets/semeval2014/ \
--output_dir ../results/semeval2014/QACGBERT/ \
--model_type QACGBERT \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 24 \
--eval_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 30 \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../models/BERT-Google/bert_config.json \
--init_checkpoint ../models/BERT-Google/pytorch_model.bin \
--seed 123 \
--evaluate_interval 250
```
Please take a look at ``code/util/args_parser.py`` to find our different arguments you can pass with. And you can alsp take a look at ``code/util/processor.py`` to see how we process different datasets. We currently supports almost 10 different dataset loadings. You can create your own within 1 minute for loading data. You can specify your directories info above in the command.

### Analyze Attention Weights, Relevance and More
Once you have your model ready, save it to a location that you know (e.g., ``../results/semeval2014/QACGBERT/checkpoint.bin``). Our example code how to get relevance scores is in a jupyter notebook format, which is much easier to read. This is how you will open it,
```bash
cd code/notebook/
jupyter notebook
```
Inside ``visualization``, we provide an example on how to extract attention scores, gradient sensitivity scores!

## License

This repo has a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
