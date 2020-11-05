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
        self.context_for_q = nn.Linear(config.hidden_size, self.attention_head_size)
        self.context_for_k = nn.Linear(config.hidden_size, self.attention_head_size)
        self.head_context = nn.Linear(2*config.hidden_size, self.attention_head_size)
        self.head_context_for_q = nn.Linear(2*self.attention_head_size, self.attention_head_size)
        self.head_context_for_k = nn.Linear(2*self.attention_head_size, self.attention_head_size)

        self.lambda_q_context_layer = nn.Linear(self.attention_head_size, 1, bias=False)
        self.lambda_q_query_layer = nn.Linear(self.attention_head_size, 1, bias=False)
        self.lambda_k_context_layer = nn.Linear(self.attention_head_size, 1, bias=False)
        self.lambda_k_key_layer = nn.Linear(self.attention_head_size, 1, bias=False)

        # zero-centered activation function, specifically for re-arch fine tunning
        self.lambda_act = nn.Tanh()

        self.quasi_act = nn.Sigmoid()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask,
                # optional parameters for saving context information
                device=None, context_embedded=None,
                head_context_embedded=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)

        ######################################################################
        # Integrate context embeddings into attention calculation
        # Note that for this model, the context is shared and is the same for
        # every head.
        # Q_context = (1-lambda_Q) * Q + lambda_Q * Context_Q
        # K_context = (1-lambda_K) * K + lambda_K * Context_K

        # head special
        batch_size = hidden_states.shape[0]
        head_context_embedded_q = self.context_for_q(head_context_embedded)
        head_context_embedded_k = self.context_for_k(head_context_embedded)

        context_embedded_q = self.context_for_q(context_embedded)
        context_embedded_q_extend = \
            torch.stack(self.num_attention_heads * [context_embedded_q], dim=1)
        context_embedded_q_extend = torch.cat([head_context_embedded_q, context_embedded_q_extend], dim=-1)
        context_embedded_q_extend = self.head_context_for_q(context_embedded_q_extend)

        lambda_q_context = self.lambda_q_context_layer(context_embedded_q_extend)
        lambda_q_query = self.lambda_q_query_layer(query_layer)
        lambda_q = lambda_q_context + lambda_q_query
        lambda_q = self.lambda_act(lambda_q)
        contextualized_query_layer = \
            (1 - lambda_q) * query_layer + lambda_q * context_embedded_q_extend

        context_embedded_k = self.context_for_k(context_embedded)
        context_embedded_k_extend = \
            torch.stack(self.num_attention_heads * [context_embedded_k], dim=1)
        context_embedded_k_extend = torch.cat([head_context_embedded_k, context_embedded_k_extend], dim=-1)
        context_embedded_k_extend = self.head_context_for_k(context_embedded_k_extend)

        lambda_k_context = self.lambda_k_context_layer(context_embedded_k_extend)
        lambda_k_key = self.lambda_k_key_layer(key_layer)
        lambda_k = lambda_k_context + lambda_k_key
        lambda_k = self.lambda_act(lambda_k)
        contextualized_key_layer = \
            (1 - lambda_k) * key_layer + lambda_k * context_embedded_k_extend
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
        return context_layer, attention_probs

    def quasi_forward(self, hidden_states, attention_mask,
                      # optional parameters for saving context information
                      device=None, context_embedded=None,
                      head_context_embedded=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        ######################################################################

        # head special
        batch_size = hidden_states.shape[0]

        context_embedded_extend = \
            torch.stack(self.num_attention_heads * [context_embedded], dim=1)

        context_embedded_extend = torch.cat([head_context_embedded, context_embedded_extend], dim=-1)
        context_embedded_extend = self.head_context(context_embedded_extend)

        # Dot product attention
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_embedded_fused_k = torch.cat([context_embedded_extend, key_layer], dim=-1)
        context_embedded_k_extend = self.head_context_for_k(context_embedded_fused_k)

        # Quasi-attention Integration with context
        quasi_attention_scores = torch.matmul(query_layer, context_embedded_k_extend.transpose(-1, -2))
        quasi_attention_scores = quasi_attention_scores / math.sqrt(self.attention_head_size)
        quasi_attention_scores = quasi_attention_scores + attention_mask
        quasi_scalar = 1.0
        quasi_attention_base = quasi_scalar * self.quasi_act(quasi_attention_scores)
        quasi_attention_scores = -1.0 * quasi_attention_base

        # Quasi-gated control
        # lambda_q_context = self.lambda_q_context_layer(context_embedded_q_extend)
        lambda_q_query = self.lambda_q_query_layer(query_layer)
        lambda_q = self.lambda_act(lambda_q_query) # -1, 1; 0
        lambda_k_context = self.lambda_k_context_layer(context_embedded_k_extend)
        lambda_k_key = self.lambda_k_key_layer(key_layer)
        lambda_k = self.lambda_act(lambda_k_context + lambda_k_key) # -1, 1; 0
        lambda_context = torch.abs(lambda_q*lambda_k)
        attention_probs = (1-lambda_context) * attention_probs + lambda_context * quasi_attention_scores

        ######################################################################

        value_layer = self.transpose_for_scores(mixed_value_layer)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, quasi_attention_base

class ContextBERTAttention(nn.Module):
    def __init__(self, config):
        super(ContextBERTAttention, self).__init__()
        self.self = ContextBERTSelfAttention(config)
        self.output = BERTSelfOutput(config)

    def forward(self, input_tensor, attention_mask,
                # optional parameters for saving context information
                device=None, context_embedded=None,
                head_context_embedded=None):
        self_output, attention_probs = \
                      self.self(input_tensor, attention_mask,
                                              device, context_embedded,
                                              head_context_embedded)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs

class ContextBERTLayer(nn.Module):
    def __init__(self, config):
        super(ContextBERTLayer, self).__init__()
        self.attention = ContextBERTAttention(config)
        self.intermediate = BERTIntermediate(config)
        self.output = BERTOutput(config)

    def forward(self, hidden_states, attention_mask,
                # optional parameters for saving context information
                device=None, context_embedded=None,
                head_context_embedded=None):
        attention_output, attention_probs = \
                           self.attention(hidden_states, attention_mask,
                                          device, context_embedded,
                                          head_context_embedded)
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs

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
                device=None, context_embeddings=None, context_lens=None,
                head_context_embedded=None):
        #######################################################################
        # Here, we can try other ways to incoperate the context!
        # Just make sure the output context_embedded is in the 
        # shape of (batch_size, seq_len, d_hidden).
        all_encoder_layers = []
        all_encoder_attention_scores = []
        layer_index = 0
        for layer_module in self.layer:
            # update context
            deep_context_hidden = torch.cat([context_embeddings, hidden_states], dim=-1)
            deep_context_hidden = self.context_layer[layer_index](deep_context_hidden)
            deep_context_hidden = context_embeddings
            # BERT encoding
            hidden_states, attention_probs = \
                            layer_module(hidden_states, attention_mask,
                                         device, deep_context_hidden,
                                         head_context_embedded)
            all_encoder_layers.append(hidden_states)
            all_encoder_attention_scores.append(attention_probs)
            layer_index += 1
        #######################################################################
        return all_encoder_layers, all_encoder_attention_scores

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

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size
        self.context_embeddings = nn.Embedding(2*4, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                # optional parameters for saving context information
                device=None,
                context_ids=None, context_lens=None,
                all_target_ids=None):
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
        batch_size = embedding_output.shape[0]
        seq_len = embedding_output.shape[1]
        context_embedded = self.context_embeddings(context_ids).squeeze(dim=1)
        context_embedding_output = torch.stack(seq_len*[context_embedded], dim=1)

        # Extra context for head specialization
        if all_target_ids is not None:
            head_context_embedded = self.context_embeddings(all_target_ids)
            head_context_embedded = torch.stack(seq_len*[head_context_embedded], dim=1)
            head_context_embedded = \
                torch.stack(3*[head_context_embedded], dim=3).reshape(batch_size, -1, 
                                                                    self.num_attention_heads, 
                                                                    self.attention_head_size).contiguous().transpose(1,2).contiguous()
        else:
            same_target_ids = torch.tensor([0,0,0,1,1,1,2,2,3,3,4,4], dtype=torch.long).to(device)
            head_context_embedded = self.context_embeddings(same_target_ids)
            head_context_embedded = torch.stack(seq_len*[head_context_embedded], dim=0)
            head_context_embedded = torch.stack(batch_size*[head_context_embedded], dim=0)

            head_context_embedded = head_context_embedded.transpose(1,2).contiguous()

        #######################################################################

        all_encoder_layers, all_encoder_attention_scores = \
                             self.encoder(embedding_output, extended_attention_mask,
                                          device,
                                          context_embedding_output, context_lens,
                                          head_context_embedded)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output, attention_mask)
        return all_encoder_layers, all_encoder_attention_scores, \
               pooled_output, context_embedded

class HeadwiseContextAwareBertForSequenceClassification(nn.Module):
    """Proposed Context-Aware Bert Model for Sequence Classification
    """
    def __init__(self, config, num_labels, init_weight=False):
        super(HeadwiseContextAwareBertForSequenceClassification, self).__init__()
        self.bert = ContextBertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.classifier = nn.Linear(config.hidden_size*2, num_labels)
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
        init_perturbation = 1e-5
        for layer_module in self.bert.encoder.layer:
            layer_module.attention.self.lambda_q_context_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
            layer_module.attention.self.lambda_k_context_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
            layer_module.attention.self.lambda_q_query_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
            layer_module.attention.self.lambda_k_key_layer.weight.data.normal_(mean=0.0, std=init_perturbation)
        #######################################################################

    def head_specialization_loss(self, head_attention_distribution,
                                 num_specialization=4):

        head_per_sp = int(self.num_head / num_specialization)
        head_group = []
        for i in range(num_specialization):
            head_index = []
            for j in range(head_per_sp):
                head_index.append(i*head_per_sp + j)
            head_group.append(head_index)
        
        head_attention_distribution_concat = torch.stack(head_attention_distribution, dim=0).transpose(0,1).contiguous()
        loss = nn.MSELoss()
        print(head_attention_distribution_concat)

        out_group_loss = 0.0
        # out_group -> disagreement
        for i in range(0, len(head_group)):
            for j in range(i+1, len(head_group)):
                for l in range(0, len(head_group[i])):
                    for m in range(0, len(head_group[j])):
                        left = head_attention_distribution_concat[:,:,head_group[i][l]]
                        right = head_attention_distribution_concat[:,:,head_group[j][m]]
                        sim = loss(left, right)
                        out_group_loss += sim
        
        out_group_loss = -1.0 * out_group_loss
        print(out_group_loss)

        return out_group_loss

    def set_context_feature(self, unique_context_feature):
        self.unique_context_feature = unique_context_feature

    def forward(self, input_ids, token_type_ids, attention_mask, seq_lens,
                device=None, labels=None,
                # optional parameters for saving context information
                context_ids=None, context_lens=None,
                include_headwise=False,
                headwise_weight=None,
                all_target_ids=None):
        _, all_encoder_attention_scores, \
        pooled_output, context_embedded = \
            self.bert(input_ids, token_type_ids, attention_mask,
                      device,
                      context_ids, context_lens,
                      all_target_ids=all_target_ids)
        # print(all_encoder_attention_scores)
        pooled_output = self.dropout(pooled_output)
        logits = \
            self.classifier(torch.cat([pooled_output, context_embedded], dim=-1))
        # logits = \
        #     self.classifier(pooled_output)
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            # loss += self.head_specialization_loss(all_encoder_attention_scores)
            return loss, logits
        else:
            return logits