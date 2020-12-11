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

from model.CGBERT import *
from model.QACGBERT import *

from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler, WeightedRandomSampler
from tqdm import tqdm, trange

from util.optimization import BERTAdam
from util.processor import (Sentihood_NLI_M_Processor,
                            Semeval_NLI_M_Processor)

from util.tokenization import *

from util.evaluation import *

import logging
logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s', 
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

processors = {
    "sentihood_NLI_M":Sentihood_NLI_M_Processor,
    "semeval_NLI_M":Semeval_NLI_M_Processor
}

context_id_map_sentihood = {'location - 1 - general':0,
                            'location - 1 - price':1,
                            'location - 1 - safety':2,
                            'location - 1 - transit location':3,
                            'location - 2 - general':4,
                            'location - 2 - price':5,
                            'location - 2 - safety':6,
                            'location - 2 - transit location':7}

context_id_map_semeval= {'price':0,
                         'anecdotes':1,
                         'food':2,
                         'ambience':3,
                         'service':4}

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

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id, seq_len,
                 context_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.seq_len = seq_len
        # extra fields to hold context
        self.context_ids = context_ids

def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, max_context_length,
                                 # if this is true, the context will not be
                                 # appened into the inputs
                                 context_standalone, args):
    """Loads a data file into a list of `InputBatch`s."""

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
            # let us encode context into single int
            if args.task_name == "sentihood_NLI_M":
                context_ids = [context_id_map_sentihood[example.text_b]]
            else:
                context_ids = [context_id_map_semeval[example.text_b]]

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        seq_len = len(input_ids)
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        while len(context_ids) < max_context_length:
            context_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(context_ids) == max_context_length

        label_id = label_map[example.label]

        features.append(
                InputFeatures(
                        input_ids=input_ids,
                        input_mask=input_mask,
                        segment_ids=segment_ids,
                        label_id=label_id,
                        seq_len=seq_len,
                        # newly added context part
                        context_ids=context_ids))
    
    return features

def make_weights_for_balanced_classes(labels, nclasses, fixed=False):
    if fixed:
        weight = [0] * len(labels)  
        if nclasses == 3:                                     
            for idx, val in enumerate(labels):
                if val == 0:
                    weight[idx] = 0.2
                elif val == 1:
                    weight[idx] = 0.4
                elif val == 2:
                    weight[idx] = 0.4  
            return weight
        else:
            for idx, val in enumerate(labels):
                if val == 0:
                    weight[idx] = 0.2
                else:
                    weight[idx] = 0.4
            return weight 
    else:
        count = [0] * nclasses                                                      
        for item in labels:                                                         
            count[item] += 1                                                     
        weight_per_class = [0.] * nclasses                                      
        N = float(sum(count))                                                   
        for i in range(nclasses):                                                   
            weight_per_class[i] = N/float(count[i])                                 
        weight = [0] * len(labels)                                              
        for idx, val in enumerate(labels):                                          
            weight[idx] = weight_per_class[val]                                 
        return weight  

def getModelOptimizerTokenizer(model_type, vocab_file,
                               bert_config_file=None, init_checkpoint=None,
                               label_list=None,
                               do_lower_case=True,
                               num_train_steps=None,
                               learning_rate=None,
                               base_learning_rate=None,
                               warmup_proportion=None,
                               init_lrp=False):

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
    logger.info("*** Model Config ***")
    logger.info(bert_config.to_json_string())
    # overwrite the vocab size to be exact. this also save space incase
    # vocab size is shrinked.
    bert_config.vocab_size = len(tokenizer.vocab)
    # model and optimizer
    if model_type == "CGBERT":
        logger.info("model = CGBERT")
        model = CGBertForSequenceClassification(
                    bert_config, len(label_list),
                    init_weight=True)
    elif model_type == "QACGBERT":
        logger.info("model = QACGBERT")
        model = QACGBertForSequenceClassification(
                    bert_config, len(label_list),
                    init_weight=True,
                    init_lrp=init_lrp)
    else:
        assert False
    if init_checkpoint is not None:
        logger.info("retraining with saved model.")
        # only load fields that are avaliable
        if "checkpoint" in init_checkpoint:
            logger.info("loading a best checkpoint, not BERT pretrain.")
            # we need to add handling logic specially for parallel gpu trainign
            state_dict = torch.load(init_checkpoint, map_location='cpu')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:] # remove 'module.' of dataparallel
                    new_state_dict[name]=v
                else:
                    new_state_dict[k]=v
            model.load_state_dict(new_state_dict)
        else:
            model.bert.load_state_dict(torch.load(init_checkpoint, map_location='cpu'), strict=False)
    no_decay = ['bias', 'gamma', 'beta']
    block_list = ['context_for_q',
                  'context_for_k',
                  'lambda_q_context_layer', 
                  'lambda_k_context_layer',
                  'lambda_q_query_layer',
                  'lambda_k_key_layer',
                  'classifier']
    optimizer_parameters = [
        {'params': [p for n, p in model.named_parameters() 
            if not any(nd in n for nd in no_decay) and not any(bl in n for bl in block_list)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in model.named_parameters() 
            if any(nd in n for nd in no_decay) and not any(bl in n for bl in block_list)], 'weight_decay_rate': 0.0}
        ]
    # ok, let us here try different learning rates for newly added layers,
    # it is just 6 linear layers.
    if model_type == "CGBERT":
        optimizer_parameters.append(
            {'params': model.classifier.parameters(), 'lr': 1e-4},
        )
    elif model_type == "QACGBERT":
        new_lr = 2e-4
        optimizer_parameters.append(
            {'params': model.classifier.parameters(), 'lr': new_lr},
        )
        for layer_module in model.bert.encoder.layer:
            optimizer_parameters.extend([
                {'params': layer_module.attention.self.context_for_q.parameters(), 'lr': new_lr},
                {'params': layer_module.attention.self.context_for_k.parameters(), 'lr': new_lr},
                {'params': layer_module.attention.self.lambda_q_context_layer.parameters(), 'lr': new_lr},
                {'params': layer_module.attention.self.lambda_k_context_layer.parameters(), 'lr': new_lr},
                {'params': layer_module.attention.self.lambda_q_query_layer.parameters(), 'lr': new_lr},
                {'params': layer_module.attention.self.lambda_k_key_layer.parameters(), 'lr': new_lr},
            ])
    else:
        assert False

    optimizer = BERTAdam(optimizer_parameters,
                        lr=learning_rate,
                        warmup=warmup_proportion,
                        t_total=num_train_steps)
    return model, optimizer, tokenizer

def system_setups(args):
    # system related setups
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

    output_log_file = os.path.join(args.output_dir, "log.txt")
    print("output_log_file=",output_log_file)

    if args.task_name == "sentihood_NLI_M":
        with open(output_log_file, "w") as writer:
            writer.write("epoch\tglobal_step\tloss\tt_loss\tt_acc\tstrict_acc\tf1\tauc\ts_acc\ts_auc\n")
    else:
        with open(output_log_file, "w") as writer:
            writer.write("epoch\tglobal_step\tloss\tt_loss\tt_acc\taspect_P\taspect_R\taspect_F\ts_acc_4\ts_acc_3\ts_acc_2\n")

    return device, n_gpu, output_log_file

def data_and_model_loader(device, n_gpu, args, sampler="randomWeight"):
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
        args.context_standalone, args)

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

    train_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                               all_label_ids, all_seq_len, all_context_ids)
    if args.local_rank == -1:
        if sampler == "random":
            train_sampler = RandomSampler(train_data)
        else:
            # consider switching to a weighted sampler
            if args.task_name == "semeval_NLI_M":
                sampler_weights = make_weights_for_balanced_classes(all_label_ids, 5)
            else:
                sampler_weights = make_weights_for_balanced_classes(all_label_ids, 3)
            train_sampler = WeightedRandomSampler(sampler_weights, len(train_data), replacement=True)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, 
                                  batch_size=args.train_batch_size)

    # test set
    test_examples = processor.get_test_examples(args.data_dir)
    test_features = convert_examples_to_features(
        test_examples, label_list, args.max_seq_length,
        tokenizer, args.max_context_length,
        args.context_standalone, args)

    all_input_ids = torch.tensor([f.input_ids for f in test_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in test_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in test_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in test_features], dtype=torch.long)
    all_seq_len = torch.tensor([[f.seq_len] for f in test_features], dtype=torch.long)
    all_context_ids = torch.tensor([f.context_ids for f in test_features], dtype=torch.long)

    test_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                                all_label_ids, all_seq_len, all_context_ids)
    test_dataloader = DataLoader(test_data, batch_size=args.eval_batch_size, shuffle=False)

    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)
    model.to(device)

    return model, optimizer, train_dataloader, test_dataloader

def evaluate_fast(test_dataloader, model, device, n_gpu, args):
    """
    evaluate only and not recording anything
    """
    model.eval()
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    pbar = tqdm(test_dataloader, desc="Iteration")
    y_true, y_pred, score = [], [], []
    # we don't need gradient in this case.
    with torch.no_grad():
        for _, batch in enumerate(pbar):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # truncate to save space and computing resource
            input_ids, input_mask, segment_ids, label_ids, seq_lens, \
                context_ids = batch
            max_seq_lens = max(seq_lens)[0]
            input_ids = input_ids[:,:max_seq_lens]
            input_mask = input_mask[:,:max_seq_lens]
            segment_ids = segment_ids[:,:max_seq_lens]

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            seq_lens = seq_lens.to(device)
            context_ids = context_ids.to(device)

            # intentially with gradient
            tmp_test_loss, logits, _, _, _, _ = \
                model(input_ids, segment_ids, input_mask, seq_lens,
                        device=device, labels=label_ids,
                        context_ids=context_ids)

            logits = F.softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            outputs = np.argmax(logits, axis=1)
            tmp_test_accuracy=np.sum(outputs == label_ids)

            y_true.append(label_ids)
            y_pred.append(outputs)
            score.append(logits)

            test_loss += tmp_test_loss.mean().item()
            test_accuracy += tmp_test_accuracy

            nb_test_examples += input_ids.size(0)
            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples

    # we follow previous works in calculating the metrics
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    score = np.concatenate(score, axis=0)

    logger.info("***** Fast Evaluation results *****")
    result = collections.OrderedDict()
    # handling corner case for a checkpoint start
    result = {'test_loss': test_loss,
              'test_accuracy': test_accuracy}

    # for ABSA tasks, we need more evaluations
    if args.task_name == "sentihood_NLI_M":
        aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
        aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
        aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
        result = {'aspect_strict_Acc': aspect_strict_Acc,
                  'aspect_Macro_F1': aspect_Macro_F1,
                  'aspect_Macro_AUC': aspect_Macro_AUC,
                  'sentiment_Acc': sentiment_Acc,
                  'sentiment_Macro_AUC': sentiment_Macro_AUC}
    else:
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

    for key in result.keys():
        logger.info("  %s = %s\n", key, str(result[key]))

    return -1


def evaluate(test_dataloader, model, device, n_gpu, nb_tr_steps, tr_loss, epoch, 
             global_step, output_log_file, global_best_acc, args):

    model.eval()
    test_loss, test_accuracy = 0, 0
    nb_test_steps, nb_test_examples = 0, 0
    pbar = tqdm(test_dataloader, desc="Iteration")
    y_true, y_pred, score = [], [], []
    # we don't need gradient in this case.
    with torch.no_grad():
        for _, batch in enumerate(pbar):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            # truncate to save space and computing resource
            input_ids, input_mask, segment_ids, label_ids, seq_lens, \
                context_ids = batch
            max_seq_lens = max(seq_lens)[0]
            input_ids = input_ids[:,:max_seq_lens]
            input_mask = input_mask[:,:max_seq_lens]
            segment_ids = segment_ids[:,:max_seq_lens]

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            label_ids = label_ids.to(device)
            seq_lens = seq_lens.to(device)
            context_ids = context_ids.to(device)

            # intentially with gradient
            tmp_test_loss, logits, _, _, _, _ = \
                model(input_ids, segment_ids, input_mask, seq_lens,
                        device=device, labels=label_ids,
                        context_ids=context_ids)

            logits = F.softmax(logits, dim=-1)
            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()
            outputs = np.argmax(logits, axis=1)
            tmp_test_accuracy=np.sum(outputs == label_ids)

            y_true.append(label_ids)
            y_pred.append(outputs)
            score.append(logits)

            test_loss += tmp_test_loss.mean().item()
            test_accuracy += tmp_test_accuracy

            nb_test_examples += input_ids.size(0)
            nb_test_steps += 1

        test_loss = test_loss / nb_test_steps
        test_accuracy = test_accuracy / nb_test_examples

    # we follow previous works in calculating the metrics
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    score = np.concatenate(score, axis=0)

    logger.info("***** Evaluation results *****")
    result = collections.OrderedDict()
    # handling corner case for a checkpoint start
    if nb_tr_steps == 0:
        loss_tr = 0.0
    else:
        loss_tr = tr_loss/nb_tr_steps

    # for ABSA tasks, we need more evaluations
    if args.task_name == "sentihood_NLI_M":
        aspect_strict_Acc = sentihood_strict_acc(y_true, y_pred)
        aspect_Macro_F1 = sentihood_macro_F1(y_true, y_pred)
        aspect_Macro_AUC, sentiment_Acc, sentiment_Macro_AUC = sentihood_AUC_Acc(y_true, score)
        result = {'epoch': epoch,
                  'global_step': global_step,
                  'loss': loss_tr,
                  'test_loss': test_loss,
                  'test_accuracy': test_accuracy,
                  'aspect_strict_Acc': aspect_strict_Acc,
                  'aspect_Macro_F1': aspect_Macro_F1,
                  'aspect_Macro_AUC': aspect_Macro_AUC,
                  'sentiment_Acc': sentiment_Acc,
                  'sentiment_Macro_AUC': sentiment_Macro_AUC}
    else:
        aspect_P, aspect_R, aspect_F = semeval_PRF(y_true, y_pred)
        sentiment_Acc_4_classes = semeval_Acc(y_true, y_pred, score, 4)
        sentiment_Acc_3_classes = semeval_Acc(y_true, y_pred, score, 3)
        sentiment_Acc_2_classes = semeval_Acc(y_true, y_pred, score, 2)
        result = {'epoch': epoch,
                  'global_step': global_step,
                  'loss': loss_tr,
                  'test_loss': test_loss,
                  'test_accuracy': test_accuracy,
                  'aspect_P': aspect_P,
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

    # save for each time point
    if args.output_dir:
        torch.save(model.state_dict(), args.output_dir + "checkpoint_" + str(global_step) + ".bin")
        if args.task_name == "sentihood_NLI_M":
            if aspect_strict_Acc > global_best_acc:
                torch.save(model.state_dict(), args.output_dir + "best_checkpoint.bin")
                global_best_acc = aspect_strict_Acc
        else:
            if aspect_F > global_best_acc:
                torch.save(model.state_dict(), args.output_dir + "best_checkpoint.bin")
                global_best_acc = aspect_F

    return global_best_acc

def step_train(train_dataloader, test_dataloader, model, optimizer, 
               device, n_gpu, evaluate_interval, global_step, 
               output_log_file, epoch, global_best_acc, args):
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    pbar = tqdm(train_dataloader, desc="Iteration")
    for step, batch in enumerate(pbar):
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # truncate to save space and computing resource
        input_ids, input_mask, segment_ids, label_ids, seq_lens, \
            context_ids = batch
        max_seq_lens = max(seq_lens)[0]
        input_ids = input_ids[:,:max_seq_lens]
        input_mask = input_mask[:,:max_seq_lens]
        segment_ids = segment_ids[:,:max_seq_lens]

        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)
        seq_lens = seq_lens.to(device)
        context_ids = context_ids.to(device)

        loss, _, _, _, _, _ = \
            model(input_ids, segment_ids, input_mask, seq_lens,
                            device=device, labels=label_ids,
                            context_ids=context_ids)
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
        pbar.set_postfix({'train_loss': loss.tolist()})

        if global_step % evaluate_interval == 0:
            logger.info("***** Evaluation Interval Hit *****")
            global_best_acc = evaluate(test_dataloader, model, device, n_gpu, nb_tr_steps, tr_loss, epoch, 
                                       global_step, output_log_file, global_best_acc, args)

    return global_step, global_best_acc