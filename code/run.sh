# example running command
CUDA_VISIBLE_DEVICES=3,4,5,6,8,9 python run_classifier.py \
--task_name sentihood_NLI_M \
--data_dir ../datasets/sentihood/ \
--output_dir ../results/sentihood/QACGBERT-4/ \
--model_type QACGBERT \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 48 \
--eval_batch_size 48 \
--learning_rate 2e-5 \
--num_train_epochs 30 \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../models/BERT-Google/bert_config.json \
--init_checkpoint ../models/BERT-Google/pytorch_model.bin \
--seed 234 \
--context_standalone \
--evaluate_interval 250