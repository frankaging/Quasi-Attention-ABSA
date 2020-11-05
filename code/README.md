# Context-Guided BERT for Targted Aspect-Based Sentiment Analysis 
In order to accurately compare our models with previous models, we adapted the 
evaluation pipeline for one of the best performing models by [Sun et. al. 2019](<https://github.com/HSLCY/ABSA-BERT-pair>).

You can refer to their pages for how to run and evaluate the model. I copied some
of their instructions on how to run and evaluate.

## Requirement
Run following commands to install all requirements:

```
pip3 install -r requirements.txt
```

## Step 1: Download Datasets

### SentiHood

The original dataset download link is failed [dataset released paper](<http://www.aclweb.org/anthology/C16-1146>). We use the [dataset mirror](<https://github.com/HSLCY/ABSA-BERT-pair/tree/master/data/sentihood>) from one previous work. See directory: `data/sentihood/`.

Run following commands, which will generate and format needed datasets:

```
cd generate/
bash make.sh sentihood
```

### SemEval 2014

Train Data is available in [SemEval-2014 ABSA Restaurant Reviews - Train Data](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-restaurant-reviews-train-data/479d18c0625011e38685842b2b6a04d72cb57ba6c07743b9879d1a04e72185b8/) and Gold Test Data is available in [SemEval-2014 ABSA Test Data - Gold Annotations](http://metashare.ilsp.gr:8080/repository/browse/semeval-2014-absa-test-data-gold-annotations/b98d11cec18211e38229842b2b6a04d77591d40acd7542b7af823a54fb03a155/). See directory: `data/semeval2014/`.

Run following commands, which will generate and format needed datasets:

```
cd generate/
bash make.sh semeval
```

## Step 2: prepare BERT-pytorch-model

Download [BERT-Base (Google's pre-trained models)](https://github.com/google-research/bert) and then convert a tensorflow checkpoint to a pytorch model.

For example:

```
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path uncased_L-12_H-768_A-12/pytorch_model.bin
```

## Step 3: train and evaluation

**TABSA** task on **SentiHood** dataset (This will run CG-BERT model by default):

```
python run_classifier_TABSA.py \
--task_name sentihood_NLI_M \
--data_dir ../data/sentihood/bert-pair/ \
--vocab_file ../data/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file ../data/uncased_L-12_H-768_A-12/bert_config.json \
--model_type ContextBERT \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--max_context_length 6 \
--train_batch_size 20 \
--learning_rate 2e-5 \
--num_train_epochs 60 \
--output_dir ../results/sentihood/NLI_M/FinalQuasi/ \
--seed 42 \
--init_checkpoint ../data/uncased_L-12_H-768_A-12/pytorch_model.bin \
--save_checkpoint_path ../results/sentihood/NLI_M/FinalQuasi/sentihood_checkpoint
```

For example, **ABSA** task on **SentiHood** dataset (This will run CG-BERT model by default):

```
python run_classifier_ABSA.py \
--task_name semeval_NLI_M \
--data_dir ../data/semeval2014/bert-pair/ \
--vocab_file ../data/uncased_L-12_H-768_A-12/vocab.txt \
--bert_config_file ../data/uncased_L-12_H-768_A-12/bert_config.json \
--model_type ContextBERT \
--eval_test \
--do_lower_case \
--max_seq_length 512 \
--max_context_length 6 \
--train_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 60 \
--output_dir ../results/semeval2014/NLI_M/ContextBERT \
--seed 42 \
--init_checkpoint ../data/uncased_L-12_H-768_A-12/pytorch_model.bin \
--save_checkpoint_path ../results/semeval2014/NLI_M/ContextBERT/semeval_checkpoint
```

To run QACG-BERT, currently you have to go to *./model/ContextBERT.py* and modify
L239, change it to quasi_forward, and you can now retrain the model. We will 
avoid this mannual modification later.