# coding=utf-8

from __future__ import absolute_import, division, print_function

import copy
import json
import math

import six
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import collections
from functools import partial

from util.lrp import *

# access global vars here
global func_inputs
global func_activations
func_inputs = collections.defaultdict(list)
func_activations = collections.defaultdict(list)

def get_inputivation(name):
    def hook(model, input, output):
        func_inputs[name] = [_in for _in in input]
    return hook

def get_activation(name):
    def hook(model, input, output):
        print("?")
        func_activations[name] = output
    return hook

def get_activation_multi(name):
    def hook(model, input, output):
        func_activations[name] = [_out for _out in output]
    return hook

# TODO: make this init as a part of the model init
def init_hooks_lrp(model):
    """
    Initialize all the hooks required for full lrp for BERT model.
    """
    # in order to backout all the lrp through layers
    # you need to register hooks here.
    model.classifier.register_forward_hook(
        get_inputivation('model.classifier'))
    model.classifier.register_forward_hook(
        get_activation('model.classifier'))

    model.bert.embeddings.register_forward_hook(
        get_activation('model.bert.embeddings'))

    layer_module_index = 0
    for module_layer in model.bert.encoder.layer:
        layer_name_self = 'model.bert.encoder.layer.' + str(layer_module_index) + \
                          '.attention.self'
        module_layer.attention.self.register_forward_hook(
            get_activation_multi(layer_name_self))
        layer_module_index += 1

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

class BERTIntermediate(nn.Module):
    def __init__(self, config):
        super(BERTIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = gelu

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                vocab_size=32000,
                hidden_size=768,
                num_hidden_layers=12,
                num_attention_heads=12,
                intermediate_size=3072,
                hidden_act="gelu",
                hidden_dropout_prob=0.1,
                attention_probs_dropout_prob=0.1,
                max_position_embeddings=512,
                type_vocab_size=16,
                initializer_range=0.02,
                full_pooler=False): # this is for transformer-like BERT
        """Constructs BertConfig.

        Args:
            vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.full_pooler = full_pooler

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size=None)
        for (key, value) in six.iteritems(json_object):
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BERTLayerNorm(nn.Module):
    def __init__(self, config, variance_epsilon=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(BERTLayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(config.hidden_size))
        self.beta = nn.Parameter(torch.zeros(config.hidden_size))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta

class BERTEmbeddings(nn.Module):
    def __init__(self, config):
        super(BERTEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

def mask(seq_len):
    batch_size = len(seq_len)
    max_len = max(seq_len)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    for i in range(mask.size()[0]):
        mask[i,:seq_len[i]] = 1
    return mask

class ContextBERTPooler(nn.Module):
    def __init__(self, config):
        super(ContextBERTPooler, self).__init__()
        # new fields
        self.attention_gate = nn.Sequential(nn.Linear(config.hidden_size, 32),
                              nn.ReLU(),
                              nn.Dropout(config.hidden_dropout_prob),
                              nn.Linear(32, 1))
        # old fileds
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, attention_mask):
        #######################################################################
        # In pooling, we are using a local context attention mechanism, instead
        # of just pooling the first [CLS] elements
        # attn_scores = self.attention_gate(hidden_states)
        # extended_attention_mask = attention_mask.unsqueeze(dim=-1)
        # attn_scores = \
        #     attn_scores.masked_fill(extended_attention_mask == 0, -1e9)
        # attn_scores = F.softmax(attn_scores, dim=1)
        # # attened embeddings
        # hs_pooled = \
        #     torch.matmul(attn_scores.permute(0,2,1), hidden_states).squeeze(dim=1)

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        hs_pooled = hidden_states[:, 0]
        #######################################################################

        #return first_token_tensor
        pooled_output = self.dense(hs_pooled)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class ContextBERTSelfAttention(nn.Module):
    def __init__(self, config):
        super(ContextBERTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # learnable context integration factors
        # enforce initialization to zero as to leave the pretrain model
        # unperturbed in the beginning
        self.context_for_q = nn.Linear(config.hidden_size, self.attention_head_size)
        self.context_for_k = nn.Linear(config.hidden_size, self.attention_head_size)

        self.lambda_q_context_layer = nn.Linear(self.attention_head_size, 1, bias=False)
        self.lambda_q_query_layer = nn.Linear(self.attention_head_size, 1, bias=False)
        self.lambda_k_context_layer = nn.Linear(self.attention_head_size, 1, bias=False)
        self.lambda_k_key_layer = nn.Linear(self.attention_head_size, 1, bias=False)

        # zero-centered activation function, specifically for re-arch fine tunning
        self.lambda_act = nn.Sigmoid()

        self.quasi_act = nn.Sigmoid()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask,
                # optional parameters for saving context information
                device=None, context_embedded=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        ######################################################################
        # Dot product attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores) # [0, 1]

        # Quasi-attention Integration with context
        context_embedded_q = self.context_for_q(context_embedded)
        context_embedded_q_extend = \
            torch.stack(self.num_attention_heads * [context_embedded_q], dim=1)
        context_embedded_k = self.context_for_k(context_embedded)
        context_embedded_k_extend = \
            torch.stack(self.num_attention_heads * [context_embedded_k], dim=1)

        quasi_attention_scores = torch.matmul(context_embedded_q_extend, context_embedded_k_extend.transpose(-1, -2))
        quasi_attention_scores = quasi_attention_scores / math.sqrt(self.attention_head_size)
        quasi_attention_scores = quasi_attention_scores + attention_mask
        quasi_scalar = 1.0
        quasi_attention_scores = 1.0 * quasi_scalar * self.quasi_act(quasi_attention_scores) # [-1, 0]

        # Quasi-gated control
        lambda_q_context = self.lambda_q_context_layer(context_embedded_q_extend)
        lambda_q_query = self.lambda_q_query_layer(query_layer)
        lambda_q = self.quasi_act(lambda_q_context + lambda_q_query)
        lambda_k_context = self.lambda_k_context_layer(context_embedded_k_extend)
        lambda_k_key = self.lambda_k_key_layer(key_layer)
        lambda_k = self.quasi_act(lambda_k_context + lambda_k_key)
        lambda_q_scalar = 1.0
        lambda_k_scalar = 1.0
        lambda_context = lambda_q_scalar*lambda_q + lambda_k_scalar*lambda_k
        lambda_context = (1 - lambda_context)
        quasi_attention_prob = lambda_context * quasi_attention_scores
        new_attention_probs = attention_probs + quasi_attention_prob
        ######################################################################

        value_layer = self.transpose_for_scores(mixed_value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        new_attention_probs = self.dropout(new_attention_probs)

        context_layer = torch.matmul(new_attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        return context_layer, new_attention_probs, attention_probs, quasi_attention_prob

class BERTSelfOutput(nn.Module):
    def __init__(self, config):
        super(BERTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BERTOutput(nn.Module):
    def __init__(self, config):
        super(BERTOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BERTLayerNorm(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class ContextBERTAttention(nn.Module):
    def __init__(self, config):
        super(ContextBERTAttention, self).__init__()
        self.self = ContextBERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask,
                # optional parameters for saving context information
                device=None, context_embedded=None):
        self_output, new_attention_probs, attention_probs, quasi_attention_prob = \
            self.self.forward(input_tensor, attention_mask,
                              device, context_embedded)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, new_attention_probs, attention_probs, quasi_attention_prob

class ContextBERTLayer(nn.Module):
    def __init__(self, config):
        super(ContextBERTLayer, self).__init__()
        self.attention = ContextBERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask,
                # optional parameters for saving context information
                device=None, context_embedded=None):
        attention_output, new_attention_probs, attention_probs, quasi_attention_prob = \
            self.attention(hidden_states, attention_mask,
                           device, context_embedded)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, new_attention_probs, attention_probs, quasi_attention_prob

class ContextBERTEncoder(nn.Module):
    def __init__(self, config):
        super(ContextBERTEncoder, self).__init__()

        deep_context_transform_layer = \
            nn.Linear(2*config.hidden_size, config.hidden_size)
        self.context_layer = \
            nn.ModuleList([copy.deepcopy(deep_context_transform_layer) for _ in range(config.num_hidden_layers)])  

        layer = ContextBERTLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])    

    def forward(self, hidden_states, attention_mask,
                # optional parameters for saving context information
                device=None, context_embeddings=None):
        #######################################################################
        # Here, we can try other ways to incoperate the context!
        # Just make sure the output context_embedded is in the 
        # shape of (batch_size, seq_len, d_hidden).
        all_encoder_layers = []
        all_new_attention_probs = []
        all_attention_probs = []
        all_quasi_attention_prob = []
        layer_index = 0
        for layer_module in self.layer:
            # update context
            deep_context_hidden = torch.cat([context_embeddings, hidden_states], dim=-1)
            deep_context_hidden = self.context_layer[layer_index](deep_context_hidden)
            deep_context_hidden += context_embeddings
            # BERT encoding
            hidden_states, new_attention_probs, attention_probs, quasi_attention_prob = \
                    layer_module(hidden_states, attention_mask,
                                 device, deep_context_hidden)
            all_encoder_layers.append(hidden_states)
            all_new_attention_probs.append(new_attention_probs.data)
            all_attention_probs.append(attention_probs.data)
            all_quasi_attention_prob.append(quasi_attention_prob.data)
            layer_index += 1
        #######################################################################
        return all_encoder_layers, all_new_attention_probs, all_attention_probs, all_quasi_attention_prob

    def backward_lat(self, attention_scores):
        print("in encoder")

        layer_name_self = 'model.bert.encoder.layer.' + str(0) + '.attention.self'
        for k in func_activations.keys():
            print(k)
        print(len(func_activations[layer_name_self]))

        pass

class ContextBertModel(nn.Module):
    """ Context-aware BERT base model
    """
    def __init__(self, config: BertConfig):
        """Constructor for BertModel.

        Args:
            config: `BertConfig` instance.
        """
        super(ContextBertModel, self).__init__()
        self.embeddings = BERTEmbeddings(config)
        self.encoder = ContextBERTEncoder(config)
        self.pooler = ContextBERTPooler(config)

        self.context_embeddings = nn.Embedding(2*4, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                # optional parameters for saving context information
                device=None, context_ids=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, from_seq_length]
        # So we can broadcast to [batch_size, num_heads, to_seq_length, from_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids)

        #######################################################################
        # Context embeddings
        seq_len = embedding_output.shape[1]
        context_embedded = self.context_embeddings(context_ids).squeeze(dim=1)
        context_embedding_output = torch.stack(seq_len*[context_embedded], dim=1)
        #######################################################################

        all_encoder_layers, all_new_attention_probs, all_attention_probs, all_quasi_attention_prob = \
            self.encoder(embedding_output, extended_attention_mask,
                         device,
                         context_embedding_output)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output, attention_mask)
        return pooled_output, all_new_attention_probs, all_attention_probs, all_quasi_attention_prob

    def backward_lat(self, attention_scores):
        self.encoder.backward_lat(attention_scores)

        return attention_scores

class QACGBertForSequenceClassification(nn.Module):
    """Proposed Context-Aware Bert Model for Sequence Classification
    """
    def __init__(self, config, num_labels, init_weight=False, init_lrp=False):
        super(QACGBertForSequenceClassification, self).__init__()
        self.bert = ContextBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_head = config.num_attention_heads
        self.config = config
        if init_weight:
            print("init_weight = True")
            def init_weights(module):
                if isinstance(module, (nn.Linear, nn.Embedding)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                elif isinstance(module, BERTLayerNorm):
                    module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                    module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
                if isinstance(module, nn.Linear):
                    if module.bias is not None:
                        module.bias.data.zero_()
            self.apply(init_weights)

        #######################################################################
        # Let's do special handling of initialization of newly added layers
        # for newly added layer, we want it to be "invisible" to the
        # training process in the beginning and slightly diverge from the
        # original model. To illustrate, let's image we add a linear layer
        # in the middle of a pretrain network. If we initialize the weight
        # randomly by default, then it will effect the output of the 
        # pretrain model largely so that it lost the point of importing
        # pretrained weights.
        #
        # What we do instead is that we initialize the weights to be a close
        # diagonal identity matrix, so that, at the beginning, for the 
        # network, it will be bascially copying the input hidden vectors as
        # the output, and slowly diverge in the process of fine tunning. 
        # We turn the bias off to accomedate this as well.
        init_perturbation = 1e-2
        for layer_module in self.bert.encoder.layer:
            layer_module.attention.self.lambda_q_context_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
            layer_module.attention.self.lambda_k_context_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
            layer_module.attention.self.lambda_q_query_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
            layer_module.attention.self.lambda_k_key_layer.weight.data.normal_(mean=0.0, std=init_perturbation)

        #######################################################################
        if init_lrp:
            print("init_lrp = True")
            init_hooks_lrp(self)

    def forward(self, input_ids, token_type_ids, attention_mask, seq_lens,
                device=None, labels=None,
                # optional parameters for saving context information
                context_ids=None):

        pooled_output, _, _, _ = \
            self.bert(input_ids, token_type_ids, attention_mask,
                      device, context_ids)
        
        pooled_output = self.dropout(pooled_output)

        logits = \
            self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits

    def backward_gradient(self, sensitivity_grads):
        classifier_out = func_activations['model.classifier']
        embedding_output = func_activations['model.bert.embeddings']
        sensitivity_grads = torch.autograd.grad(classifier_out, embedding_output, 
                                                grad_outputs=sensitivity_grads)[0]
        return sensitivity_grads

    def backward_lat(self, attention_scores):

        self.bert.backward_lat(attention_scores)

        # # for lat, we only cares about the attention weights
        # num_layers = self.config.num_hidden_layers
        # num_heads = self.config.num_attention_heads

        # # we simply repeat for all heads as they start off equally
        # lat_a_h = []
        # lat_a_self_h = []
        # lat_a_quasi_h = []
        # for h in range(num_heads):
        #     pre_a_h = attention_scores.unsqueeze(1) # span out for seq_len
        #     pre_a_self_h = attention_scores.unsqueeze(1) # span out for seq_len
        #     pre_a_quasi_h = attention_scores.unsqueeze(1) # span out for seq_len
        #     for i in reversed(range(num_layers)):
                
        #         layer_name_self = 'model.bert.encoder.layer.' + str(i) + '.attention.self'
        #         print(func_activations[layer_name_self])
        #         a_h = func_activations[layer_name_self][1][:,h] # b, l, l
        #         a_self_h = func_activations[layer_name_self][2][:,h]
        #         a_quasi_h = func_activations[layer_name_self][3][:,h]
        #         # propagate
        #         pre_a_h = torch.matmul(pre_a_h, a_h)
        #         pre_a_self_h = torch.matmul(pre_a_self_h, a_self_h)
        #         pre_a_quasi_h = torch.matmul(pre_a_quasi_h, a_quasi_h)
        #     # collect for heads
        #     lat_a_h.append(pre_a_h.data)
        #     lat_a_self_h.append(pre_a_h.data)
        #     lat_a_quasi_h.append(pre_a_h.data)

        # lat_a_h = torch.cat(lat_a_h, dim=1).sum(dim=1) # b, l
        # lat_a_self_h = torch.cat(lat_a_self_h, dim=1).sum(dim=1)
        # lat_a_quasi_h = torch.cat(lat_a_quasi_h, dim=1).sum(dim=1)

        return attention_scores