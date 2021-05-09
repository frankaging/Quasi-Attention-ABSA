# best performing model results
CUDA_VISIBLE_DEVICES=1,4,5,8 python run_classifier.py \
--task_name sentihood_NLI_M \
--data_dir ../datasets/sentihood/ \
--output_dir ../results/sentihood/QACGBERT-reproduce/ \
--model_type QACGBERT \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 64 \
--eval_batch_size 256 \
--learning_rate 2e-5 \
--num_train_epochs 30 \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../models/BERT-Google/bert_config.json \
--init_checkpoint ../models/BERT-Google/pytorch_model.bin \
--seed 123 \
--evaluate_interval 25

# other models we tried
CUDA_VISIBLE_DEVICES=1,4,5,8 python run_classifier.py \
--task_name sentihood_NLI_M \
--data_dir ../datasets/sentihood/ \
--output_dir ../results/sentihood/QACGBERT-reproduce-other/ \
--model_type QACGBERT \
--do_lower_case \
--max_seq_length 128 \
--train_batch_size 64 \
--eval_batch_size 256 \
--learning_rate 2e-5 \
--num_train_epochs 30 \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../models/BERT-Google/bert_config.json \
--init_checkpoint ../models/BERT-Google/pytorch_model.bin \
--seed 123 \
--evaluate_interval 25 \
--context_standalone