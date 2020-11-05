# example running command
CUDA_VISIBLE_DEVICES=5,6,7 python run_classifier.py \
--task_name sentihood_NLI_M \
--data_dir ../datasets/sentihood/ \
--vocab_file ../models/BERT-Google/vocab.txt \
--bert_config_file ../data/BERT-Google/bert_config.json \
--model_type QACGBERT \
--do_lower_case \
--max_seq_length 512 \
--train_batch_size 24 \
--learning_rate 2e-5 \
--num_train_epochs 25 \
--output_dir ../results/sentihood/QACGBERT/ \
--init_checkpoint ../data/BERT-Google/pytorch_model.bin \
--seed 42