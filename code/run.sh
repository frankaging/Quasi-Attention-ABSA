# Model Convert
python convert_tf_checkpoint_to_pytorch.py \
--tf_checkpoint_path ../data/uncased_L-12_H-768_A-12/bert_model.ckpt \
--bert_config_file ../data/uncased_L-12_H-768_A-12/bert_config.json \
--pytorch_dump_path ../data/uncased_L-12_H-768_A-12/pytorch_model.bin

# Semeval ABSA Task
CUDA_VISIBLE_DEVICES=0 python run_classifier_ABSA.py \
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
--save_checkpoint_path ../results/semeval2014/NLI_M/ContextBERT/semeval_checkpoint_no_quasi

# Sentihood TABSA Task
CUDA_VISIBLE_DEVICES=0,1 python run_classifier_TABSA.py \
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

