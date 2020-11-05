import pickle
import re
import os

import random
import numpy as np
import torch
from random import shuffle
import argparse
import pickle

import collections

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from model.BiLSTM import *
from model.BERT import *
from model.BERTSimple import *
from model.ContextBERT import *
from model.HeadwiseContextBERT import *

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from util.optimization import BERTAdam
from util.processor import (Sentihood_NLI_M_Processor,
                            Semeval_NLI_M_Processor)

from util.tokenization import *

from evaluation import *

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_len,
                 context_ids, context_len, target_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.seq_len = seq_len
        # extra fields to hold context
        self.context_ids = context_ids
        self.context_len = context_len
        self.target_id = target_id


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, max_context_length,
                                 # if this is true, the context will not be
                                 # appened into the inputs
                                 context_standalone):
    """Loads a data file into a list of `InputBatch`s."""

    unique_context_id_map = {'location - 1 - general':0,
                    'location - 1 - price':1,
                    'location - 1 - safety':2,
                    'location - 1 - transit location':3,
                    'location - 2 - general':4,
                    'location - 2 - price':5,
                    'location - 2 - safety':6,
                    'location - 2 - transit location':7}

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    features = []
    for (ex_index, example) in enumerate(tqdm(examples)):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
        # tokens of context
        tokens_context = None
        if example.text_b:
            tokens_context = tokenizer.tokenize(example.text_b)
            if "location - 1" in example.text_b:
                target_identity = 0
            else:
                target_identity = 1

        if tokens_b and not context_standalone:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[0:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        segment_ids = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append("[SEP]")
        segment_ids.append(0)

        if tokens_b and not context_standalone:
            for token in tokens_b:
                tokens.append(token)
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        context_ids = []
        if tokens_context:
            # context_ids = tokenizer.convert_tokens_to_ids(tokens_context)
            
            # let us encode context into single int
            context_ids = [unique_context_id_map[example.text_b]]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        seq_len = len(input_ids)
        context_len = len(context_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        # while len(context_ids) < max_context_length:
        #     context_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        # assert len(context_ids) == max_context_length
        # single int to rep context now
        assert len(context_ids) == 1

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        seq_len=seq_len,
                        # newly added context part
                        context_ids=context_ids,
                        context_len=context_len,
                        target_id=target_identity))
    
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def getModelOptimizerTokenizer(model_type, vocab_file, embed_file=None, 
                               bert_config_file=None, init_checkpoint=None,
                               label_list=None,
                               do_lower_case=True,
                               num_train_steps=None,
                               learning_rate=None,
                               base_learning_rate=None,
                               warmup_proportion=None):
    if embed_file is not None:
        # in case pretrain embeddings
        embeddings = pickle.load(open(embed_file, 'rb'))

    if model_type == "BiLSTM":
        logger.info("model = BiLSTM")
        tokenizer = WordLevelTokenizer(vocab_file=vocab_file)
        model = BiLSTM(pretrain_embeddings=embeddings, freeze=True)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        # if pretrain, we will load here
        if init_checkpoint is not None:
            logger.info("retraining with saved model.")
            checkpoint = torch.load(init_checkpoint, map_location='cpu')
            model.load_state_dict(checkpoint)
    elif model_type == "BERTSimple":
        logger.info("model = BERTSimple")
        tokenizer = WordLevelTokenizer(vocab_file=vocab_file)
        bert_config = BertConfig(hidden_size=300,
                                 num_hidden_layers=12,
                                 num_attention_heads=12,
                                 intermediate_size=3072,
                                 hidden_act="gelu",
                                 hidden_dropout_prob=0.1,
                                 attention_probs_dropout_prob=0.1,
                                 max_position_embeddings=512,
                                 type_vocab_size=2,
                                 initializer_range=0.02)
        if embed_file is None:
            raise ValueError("BERTSimple needs a pretrain embedding file.")
        model = \
            BertSimpleForSequenceClassification(bert_config,
                                                pretrain_embeddings=embeddings,
                                                num_labels=len(label_list),
                                                type_id_enable=True,
                                                position_enable=True)
        if init_checkpoint is not None:
            logger.info("retraining with saved model.")
            model.bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        # instead of BERTAdam, we use Adam to be able to perform gs on bias
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    elif model_type == "BERTPretrain":
        logger.info("model = BERTPretrain")
        if bert_config_file is not None:
            bert_config = BertConfig.from_json_file(bert_config_file)
        else:
            # default?
            bert_config = BertConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02
            )
        tokenizer = FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case, pretrain=False)
        # overwrite the vocab size to be exact. this also save space incase
        # vocab size is shrinked.
        bert_config.vocab_size = len(tokenizer.vocab)
        # model and optimizer
        model = BertForSequenceClassification(bert_config, len(label_list))

        if init_checkpoint is not None:
            logger.info("retraining with saved model.")
            model.bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'))
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
            
        optimizer = BERTAdam(optimizer_parameters,
                            lr=learning_rate,
                            warmup=warmup_proportion,
                            t_total=num_train_steps)
    elif model_type == "ContextBERT":
        logger.info("model = ContextBERT")
        # this is the model we develop
        tokenizer = FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case, pretrain=False)
        if bert_config_file is not None:
            bert_config = BertConfig.from_json_file(bert_config_file)
        else:
            # default?
            bert_config = BertConfig(
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=2,
                initializer_range=0.02
            )
        # overwrite the vocab size to be exact. this also save space incase
        # vocab size is shrinked.
        bert_config.vocab_size = len(tokenizer.vocab)
        # model and optimizer
        model = ContextAwareBertForSequenceClassification(
                    bert_config, len(label_list),
                    init_weight=True)
        if init_checkpoint is not None:
            logger.info("retraining with saved model.")
            # only load fields that are avaliable
            if "checkpoint" in init_checkpoint:
                # load full is it is not google BERT original pretrain
                model.load_state_dict(torch.load(init_checkpoint, map_location='cpu'), strict=False)
            else:
                model.bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'), strict=False)
        #######################################################################
        # Instead of BERTAdam, we use Adam to be able to perform gs on bias
        # we will have a smaller learning rate for BERT orignal parameters
        # and a higher learning rate for new parameters
        # orignal_bert = BertForSequenceClassification(bert_config, len(label_list))
        # original_params = []
        # exclude_params = ["classifier.weight", "classifier.bias"]
        # for params in orignal_bert.named_parameters():
        #     if params not in exclude_params:
        #         original_params.append(params[0])
        # no_decay = ['bias', 'gamma', 'beta']
        # base_params_no_decay = list(map(lambda x: x[1],
        #                             list(filter(lambda kv: kv[0] in original_params \
        #                             and any(nd in kv[0] for nd in no_decay),
        #                             model.named_parameters()))))
        # base_params_decay = list(map(lambda x: x[1],
        #                             list(filter(lambda kv: kv[0] in original_params \
        #                             and not any(nd in kv[0] for nd in no_decay),
        #                             model.named_parameters()))))
        # params = list(map(lambda x: x[1], 
        #                   list(filter(lambda kv: kv[0] not in original_params \
        #                                 or kv[0] in exclude_params,
        #                   model.named_parameters()))))

        # optimizer_parameters = [
        #     {'params': base_params_decay, 'weight_decay_rate': 0.01},
        #     {'params': base_params_no_decay, 'weight_decay_rate': 0.0},
        #     {'params': params, 'lr': learning_rate, 'weight_decay_rate': 0.01}]
        # optimizer = BERTAdam(optimizer_parameters,
        #                      lr=base_learning_rate,
        #                      warmup=warmup_proportion,
        #                      t_total=num_train_steps)

        # orignal_bert = BertForSequenceClassification(bert_config, len(label_list))
        # original_params = []
        # exclude_params = ["classifier.weight", "classifier.bias"]
        # for params in orignal_bert.named_parameters():
        #     if params not in exclude_params:
        #         original_params.append(params[0])
        # no_decay = ['bias', 'gamma', 'beta']
        # base_params_no_decay = list(map(lambda x: x[1],
        #                             list(filter(lambda kv: kv[0] in original_params \
        #                             and any(nd in kv[0] for nd in no_decay),
        #                             model.named_parameters()))))
        # base_params_decay = list(map(lambda x: x[1],
        #                             list(filter(lambda kv: kv[0] in original_params \
        #                             and not any(nd in kv[0] for nd in no_decay),
        #                             model.named_parameters()))))
        # params = list(map(lambda x: x[1], 
        #                   list(filter(lambda kv: kv[0] not in original_params \
        #                                 or kv[0] in exclude_params,
        #                   model.named_parameters()))))

        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
        optimizer = BERTAdam(optimizer_parameters,
                            lr=learning_rate,
                            warmup=warmup_proportion,
                            t_total=num_train_steps)
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        #######################################################################
    elif model_type == "HeadwiseContextBERT":
        logger.info("model = HeadwiseContextBERT")
        # this is the model we develop
        tokenizer = FullTokenizer(
            vocab_file=vocab_file, do_lower_case=do_lower_case, pretrain=False)
        if bert_config_file is not None:
            bert_config = BertConfig.from_json_file(bert_config_file)
        else:
            # default?
            bert_config = BertConfig()
        # overwrite the vocab size to be exact. this also save space incase
        # vocab size is shrinked.
        bert_config.vocab_size = len(tokenizer.vocab)
        # model and optimizer
        model = HeadwiseContextAwareBertForSequenceClassification(
                    bert_config, len(label_list),
                    init_weight=True)
        if init_checkpoint is not None:
            logger.info("retraining with saved model.")
            # only load fields that are avaliable
            if "checkpoint" in init_checkpoint:
                logger.info("retraining with a checkpoint model instead.")
                # load full is it is not google BERT original pretrain
                model.load_state_dict(torch.load(init_checkpoint, map_location='cpu'), strict=False)
            else:
                model.bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'), strict=False)
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_parameters = [
            {'params': [p for n, p in model.named_parameters() 
                if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
            {'params': [p for n, p in model.named_parameters() 
                if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
            ]
            
        optimizer = BERTAdam(optimizer_parameters,
                            lr=learning_rate,
                            warmup=warmup_proportion,
                            t_total=num_train_steps)
    else:
        logger.info("***** Not Support Model Type *****")
    return model, optimizer, tokenizer

def Train(args):

    # whether include the head specializations
    if args.head_sp_loss:
        logger.info("include headwise specialization loss")
    else:
        logger.info("NOT include headwise specialization loss")

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device %s n_gpu %d distributed training %r", device, n_gpu, bool(args.local_rank != -1))

    if args.accumulate_gradients < 1:
        raise ValueError("Invalid accumulate_gradients parameter: {}, should be >= 1".format(
                            args.accumulate_gradients))

    args.train_batch_size = int(args.train_batch_size / args.accumulate_gradients)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.bert_config_file is not None:
        bert_config = BertConfig.from_json_file(args.bert_config_file)
        if args.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length {} because the BERT model was only trained up to sequence length {}".format(
                args.max_seq_length, bert_config.max_position_embeddings))

    # not preloading
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    # prepare dataloaders
    processors = {
        "sentihood_NLI_M":Sentihood_NLI_M_Processor,
        "semeval_NLI_M":Semeval_NLI_M_Processor
    }

    processor = processors[args.task_name]()
    label_list = processor.get_labels()

    # training setup
    train_examples = None
    num_train_steps = None
    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size * args.num_train_epochs)

    # model and optimizer
    model, optimizer, tokenizer = \
        getModelOptimizerTokenizer(model_type=args.model_type,
                                   vocab_file=args.vocab_file,
                                   embed_file=args.embed_file,
                                   bert_config_file=args.bert_config_file,
                                   init_checkpoint=args.init_checkpoint,
                                   label_list=label_list,
                                   do_lower_case=True,
                                   num_train_steps=num_train_steps,
                                   learning_rate=args.learning_rate,
                                   base_learning_rate=args.base_learning_rate,
                                   warmup_proportion=args.warmup_proportion)

    # training set
    train_features = convert_examples_to_features(
        train_examples, label_list, args.max_seq_length,
        tokenizer, args.max_context_length,
        args.context_standalone)

    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
    all_seq_len = torch.tensor([[f.seq_len] for f in train_features], dtype=torch.long)
    all_context_ids = torch.tensor([f.context_ids for f in train_features], dtype=torch.long)
    all_context_len = torch.tensor([[f.context_len] for f in train_features], dtype=torch.long)
    target_id_convert = [[0,1,2,3], [4,5,6,7]]
    all_target_ids = torch.tensor([target_id_convert[f.target_id] for f in train_features], dtype=torch.long)

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_label_ids, all_seq_len, all_context_ids,
                               all_context_len, all_target_ids)
    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    # test set
    if args.eval_test:
        test_examples = processor.get_test_examples(args.data_dir)
        test_features = convert_examples_to_features(
            test_examples, label_list, args.max_seq_length,
            tokenizer, args.max_context_length,
            args.context_standalone)

        all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
        all_seq_len = torch.tensor([[f.seq_len] for f in test_features], dtype=torch.long)
        all_context_ids = torch.tensor([f.context_ids for f in test_features], dtype=torch.long)
        all_context_len = torch.tensor([[f.context_len] for f in test_features], dtype=torch.long)
        target_id_convert = [[0,1,2,3], [4,5,6,7]]
        all_target_ids = torch.tensor([target_id_convert[f.target_id] for f in test_features], dtype=torch.long)

        test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                  all_label_ids, all_seq_len, all_context_ids,
                                  all_context_len, all_target_ids)
        test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # train
    model.to(device)
    output_log_file = os.path.join(args.output_dir, "log.txt")
    print("output_log_file=",output_log_file)

    if args.save_checkpoint_path:
        with open(output_log_file, "a+") as writer:
            if args.eval_test:
                writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
            else:
                writer.write("epoch\tglobal_step\tloss\n")
    else:
        with open(output_log_file, "w") as writer:
            if args.eval_test:
                writer.write("epoch\tglobal_step\tloss\ttest_loss\ttest_accuracy\n")
            else:
                writer.write("epoch\tglobal_step\tloss\n")
    
    global_step = 0
    epoch=0

    # training epoch to eval
    eval_freq_train = 100
    grads_in_norm_list = []
    for _ in trange(int(args.num_train_epochs), desc="Epoch"):
        model.to(device)
        epoch+=1
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # truncate to save space and computing resource
            input_ids, input_mask, segment_ids, label_ids, seq_lens, \
                context_ids, context_lens, all_target_ids = batch
            max_seq_lens = max(seq_lens)[0]
            input_ids = input_ids[:,:max_seq_lens]
            input_mask = input_mask[:,:max_seq_lens]
            segment_ids = segment_ids[:,:max_seq_lens]
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            seq_lens = seq_lens.to(device)
            # context fields
            context_ids = context_ids.to(device)
            # all_target_ids = all_target_ids.to(device)

            loss, _, _, _, _ = \
                model(input_ids, segment_ids, input_mask, seq_lens,
                                  device=device, labels=label_ids,
                                  context_ids=context_ids,
                                  context_lens=context_lens,
                                  include_headwise=args.head_sp_loss,
                                  headwise_weight=args.head_sp_loss_lambda)
            if n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            loss.backward()
            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()    # We have accumulated enought gradients
                model.zero_grad()
                global_step += 1

        # save for each time point
        if args.save_checkpoint_path:
            torch.save(model.state_dict(), args.save_checkpoint_path + ".bin")

        save_pred_file = os.path.join(args.output_dir, "test_ep_"+str(epoch)+".txt")

        attention_scores_list = []
        # eval_test
        if args.eval_test:
            model.eval()
            test_loss, test_accuracy = 0, 0
            nb_test_steps, nb_test_examples = 0, 0

            with open(save_pred_file,"w") as f_test:
                for input_ids, input_mask, segment_ids, label_ids, seq_lens, \
                    context_ids, context_lens, all_target_ids in test_dataloader:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # truncate to save space and computing resource
                    max_seq_lens = max(seq_lens)[0]
                    input_ids = input_ids[:,:max_seq_lens]
                    input_mask = input_mask[:,:max_seq_lens]
                    segment_ids = segment_ids[:,:max_seq_lens]
                    input_ids = input_ids.to(device)
                    input_mask = input_mask.to(device)
                    segment_ids = segment_ids.to(device)
                    label_ids = label_ids.to(device)
                    seq_lens = seq_lens.to(device)
                    # context fields
                    context_ids = context_ids.to(device)
                    # all_target_ids = all_target_ids.to(device)

                    # intentially with gradient
                    tmp_test_loss, logits, _, _, _ = \
                        model(input_ids, segment_ids, input_mask, seq_lens,
                                device=device, labels=label_ids,
                                context_ids=context_ids,
                                context_lens=context_lens,
                                include_headwise=args.head_sp_loss,
                                headwise_weight=args.head_sp_loss_lambda)

                    logits = F.softmax(logits, dim=-1)
                    logits = logits.detach().cpu().numpy()
                    label_ids = label_ids.to('cpu').numpy()
                    outputs = np.argmax(logits, axis=1)
                    for output_i in range(len(outputs)):
                        f_test.write(str(outputs[output_i]))
                        for ou in logits[output_i]:
                            f_test.write(" "+str(ou))
                        f_test.write("\n")
                    tmp_test_accuracy=np.sum(outputs == label_ids)

                    test_loss += tmp_test_loss.mean().item()
                    test_accuracy += tmp_test_accuracy

                    nb_test_examples += input_ids.size(0)
                    nb_test_steps += 1

            test_loss = test_loss / nb_test_steps
            test_accuracy = test_accuracy / nb_test_examples


        result = collections.OrderedDict()
        if args.eval_test:
            result = {'epoch': epoch,
                    'global_step': global_step,
                    'loss': tr_loss/nb_tr_steps,
                    'test_loss': test_loss,
                    'test_accuracy': test_accuracy}
        else:
            result = {'epoch': epoch,
                    'global_step': global_step,
                    'loss': tr_loss/nb_tr_steps}

        logger.info("***** Eval results *****")
        with open(output_log_file, "a+") as writer:
            for key in result.keys():
                logger.info("  %s = %s\n", key, str(result[key]))
                writer.write("%s\t" % (str(result[key])))
            writer.write("\n")

        logger.info("***** Metrices results *****")
        # we print out eval results directly
        result = collections.OrderedDict()
        if args.task_name in ["sentihood_NLI_M"]:
            y_true = get_y_true(args.task_name)
            y_pred, score = get_y_pred(args.task_name, save_pred_file)
            aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
            aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
            aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
            result = {'aspect_strict_Acc': aspect_strict_Acc,
                    'aspect_Macro_F1': aspect_Macro_F1,
                    'aspect_Macro_AUC': aspect_Macro_AUC,
                    'sentiment_Acc': sentiment_Acc,
                    'sentiment_Macro_AUC': sentiment_Macro_AUC}
        else:
            y_true = get_y_true(args.task_name)
            y_pred, score = get_y_pred(args.task_name, args.pred_data_dir)
            aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
            sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
            sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
            sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
            result = {'aspect_P': aspect_P,
                    'aspect_R': aspect_R,
                    'aspect_F': aspect_F,
                    'sentiment_Acc_4_classes': sentiment_Acc_4_classes,
                    'sentiment_Acc_3_classes': sentiment_Acc_3_classes,
                    'sentiment_Acc_2_classes': sentiment_Acc_2_classes}

        with open(output_log_file, "a+") as writer:
            for key in result.keys():
                logger.info("  %s = %s\n", key, str(result[key]))
                writer.write("%s\t" % (str(result[key])))
            writer.write("\n")


def router(args):
    Train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        choices=["sentihood_NLI_M", "sentihood_QA_M"],
                        help="The name of the task to train.")
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--vocab_file",
                        default=None,
                        type=str,
                        required=True,
                        help="The vocabulary file that the model was trained on.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model checkpoints will be written.")
    parser.add_argument('--model_type', 
                        type=str,
                        default=None,
                        required=True,
                        help='type of model to train')   
    parser.add_argument('--head_sp_loss', 
                        default=False,
                        action='store_true',
                        help='whether to include head specialization loss lambda.')    
    ## Other parameters
    parser.add_argument("--context_standalone",
                        default=False,
                        action='store_true',
                        help="Whether the seperate the context from inputs.")
    parser.add_argument('--head_sp_loss_lambda', 
                        type=float,
                        default=1.0,
                        required=False,
                        help='head specialization loss lambda.')    
    parser.add_argument("--bert_config_file",
                        default=None,
                        type=str,
                        help="The config json file corresponding to the pre-trained BERT model. \n"
                             "This specifies the model architecture.")
    parser.add_argument("--embed_file",
                        default=None,
                        type=str,
                        help="The embedding file that the model was trained on.")
    parser.add_argument("--init_checkpoint",
                        default=None,
                        type=str,
                        help="Initial checkpoint (usually from a pre-trained model).")
    parser.add_argument("--save_checkpoint_path",
                        default=None,
                        type=str,
                        help="path to save checkpoint (usually from a pre-trained model).")
    parser.add_argument("--eval_test",
                        default=False,
                        action='store_true',
                        help="Whether to run eval on the test set.")                    
    parser.add_argument("--do_lower_case",
                        default=False,
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--max_context_length",
                        default=6,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--train_batch_size",
                        default=20,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=20,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--base_learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        default=False,
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--accumulate_gradients",
                        type=int,
                        default=1,
                        help="Number of steps to accumulate gradient on (divide the batch_size and accumulate)")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', 
                        type=int, 
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumualte before performing a backward/update pass.")                       
    args = parser.parse_args()
    router(args)