# Quasi-Attention-ABSA
Codebase for a new quasi-attention BERT model for (T)ABSA tasks

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