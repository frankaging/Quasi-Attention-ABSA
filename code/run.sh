# example running command
CUDA_VISIBLE_DEVICES=5 python run_classifier.py \
--task_name semeval_NLI_M \
--data_dir ../datasets/semeval2014/ \
--output_dir ../results/semeval2014/QACGBERT-3/ \
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
--evaluate_interval 250 \
--context_standalone