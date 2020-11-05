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

from model.BERT import *

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
        self.context_for_q = nn.Linear(self.attention_head_size, self.attention_head_size)
        self.context_for_k = nn.Linear(self.attention_head_size, self.attention_head_size)

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

        memo_bundle = {}

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        mixed_query_layer = self.transpose_for_scores(mixed_query_layer)
        mixed_key_layer = self.transpose_for_scores(mixed_key_layer)

        ######################################################################
        # Integrate context embeddings into attention calculation
        # Note that for this model, the context is shared and is the same for
        # every head.
        # Q_context = (1-lambda_Q) * Q + lambda_Q * Context_Q
        # K_context = (1-lambda_K) * K + lambda_K * Context_K

        context_embedded = self.transpose_for_scores(context_embedded)

        context_embedded_q = self.context_for_q(context_embedded)
        lambda_q_context = self.lambda_q_context_layer(context_embedded_q)
        lambda_q_query = self.lambda_q_query_layer(mixed_query_layer)
        lambda_q = lambda_q_context + lambda_q_query
        lambda_q = self.lambda_act(lambda_q)
        contextualized_query_layer = \
            (1 - lambda_q) * mixed_query_layer + lambda_q * context_embedded_q
        # memo_bundle["lambda_q"] = lambda_q

        context_embedded_k = self.context_for_k(context_embedded)
        lambda_k_context = self.lambda_k_context_layer(context_embedded_k)
        lambda_k_key = self.lambda_k_key_layer(mixed_key_layer)
        lambda_k = lambda_k_context + lambda_k_key
        lambda_k = self.lambda_act(lambda_k)
        contextualized_key_layer = \
            (1 - lambda_k) * mixed_key_layer + lambda_k * context_embedded_k
        memo_bundle["lambda_q"] = lambda_q.clone().data
        memo_bundle["lambda_k"] = lambda_k.clone().data
        ######################################################################

        value_layer = self.transpose_for_scores(mixed_value_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(contextualized_query_layer, contextualized_key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs, memo_bundle

    def quasi_forward(self, hidden_states, attention_mask,
                      # optional parameters for saving context information
                      device=None, context_embedded=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        memo_bundle = {}
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
        new_attention_probs = attention_probs + lambda_context * quasi_attention_scores

        memo_bundle["lambda_q"] = lambda_q
        memo_bundle["lambda_k"] = lambda_k
        memo_bundle["lambda_context"] = lambda_context
        memo_bundle["attention_probs"] = attention_probs
        memo_bundle["quasi_attention_scores"] = quasi_attention_scores
        memo_bundle["new_attention_probs"] = new_attention_probs

        ######################################################################

        value_layer = self.transpose_for_scores(mixed_value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        new_attention_probs = self.dropout(new_attention_probs)

        context_layer = torch.matmul(new_attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, new_attention_probs, memo_bundle

class ContextBERTAttention(nn.Module):
    def __init__(self, config):
        super(ContextBERTAttention, self).__init__()
        self.self = ContextBERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask,
                # optional parameters for saving context information
                device=None, context_embedded=None):
        self_output, attention_probs, memo_bundle = \
                      self.self.forward(input_tensor, attention_mask,
                                device, context_embedded)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs, memo_bundle

class ContextBERTLayer(nn.Module):
    def __init__(self, config):
        super(ContextBERTLayer, self).__init__()
        self.attention = ContextBERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask,
                # optional parameters for saving context information
                device=None, context_embedded=None):
        attention_output, attention_probs, memo_bundle = \
                           self.attention(hidden_states, attention_mask,
                                          device, context_embedded)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs, memo_bundle

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
                device=None, context_embeddings=None, context_lens=None):
        #######################################################################
        # Here, we can try other ways to incoperate the context!
        # Just make sure the output context_embedded is in the 
        # shape of (batch_size, seq_len, d_hidden).
        all_encoder_layers = []
        all_encoder_attention_scores = []
        all_encoder_memo_bundle = []
        layer_index = 0
        for layer_module in self.layer:
            # update context
            deep_context_hidden = torch.cat([context_embeddings, hidden_states], dim=-1)
            deep_context_hidden = self.context_layer[layer_index](deep_context_hidden)
            deep_context_hidden += context_embeddings
            # BERT encoding
            hidden_states, attention_probs, memo_bundle = \
                            layer_module(hidden_states, attention_mask,
                                         device, deep_context_hidden)
            all_encoder_layers.append(hidden_states)
            all_encoder_attention_scores.append(attention_probs)
            all_encoder_memo_bundle.append(memo_bundle)
            layer_index += 1
        #######################################################################
        return all_encoder_layers, all_encoder_attention_scores, all_encoder_memo_bundle

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
                device=None,
                context_ids=None, context_lens=None):
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

        all_encoder_layers, all_encoder_attention_scores, all_encoder_memo_bundle = \
                             self.encoder(embedding_output, extended_attention_mask,
                                          device,
                                          context_embedding_output, context_lens)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output, attention_mask)
        return all_encoder_layers, all_encoder_attention_scores, \
               pooled_output, context_embedded, all_encoder_memo_bundle, embedding_output

class ContextAwareBertForSequenceClassification(nn.Module):
    """Proposed Context-Aware Bert Model for Sequence Classification
    """
    def __init__(self, config, num_labels, init_weight=False):
        super(ContextAwareBertForSequenceClassification, self).__init__()
        self.bert = ContextBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.num_head = config.num_attention_heads
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

    def head_specialization_loss(self, head_attention_distribution,
                                 num_specialization=4):
        pass

    def set_context_feature(self, unique_context_feature):
        self.unique_context_feature = unique_context_feature

    def forward(self, input_ids, token_type_ids, attention_mask, seq_lens,
                device=None, labels=None,
                # optional parameters for saving context information
                context_ids=None, context_lens=None,
                include_headwise=False,
                headwise_weight=None,
                all_target_ids=None,
                unique_context_feature_ids=None, unique_context_lens=None):
        _, attention_scores, \
        pooled_output, context_embedded, \
        all_encoder_memo_bundle, embedding_output = \
            self.bert(input_ids, token_type_ids, attention_mask,
                      device,
                      context_ids, context_lens)
        
        pooled_output = self.dropout(pooled_output)
        # logits = \
        #     self.classifier(torch.cat([pooled_output, context_embedded], dim=-1))
        logits = \
            self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            if include_headwise:
                pass
            return loss, logits, embedding_output, all_encoder_memo_bundle, None
        else:
            return logits