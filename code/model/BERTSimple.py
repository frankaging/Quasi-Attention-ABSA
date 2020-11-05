from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math, copy, time
from torch.autograd import Variable
import torch.nn.functional as F

from model.BERT import *
from torch import nn
from torch.nn import CrossEntropyLoss

class BERTSimpleEmbeddings(nn.Module):
    def __init__(self, config, pretrain_embeddings,
                 type_id_enable=True,
                 position_enable=True):
        super(BERTSimpleEmbeddings, self).__init__()
        """Construct the embedding module from word, position and token_type embeddings.
        """
        self.word_embeddings = \
            nn.Embedding.from_pretrained(pretrain_embeddings,
                                         freeze=True, padding_idx=0)

        self.type_id_enable = type_id_enable
        self.position_enable = position_enable
        if type_id_enable:
            print("type_id_enable = True")
            self.token_type_embeddings = \
                nn.Embedding(config.type_vocab_size, config.hidden_size)
            self.token_type_embeddings.weight.data.normal_(
                mean=0.0, std=config.initializer_range)
        if position_enable:
            print("position_enable = True")
            self.position_embeddings = \
                nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.position_embeddings.weight.data.normal_(
                mean=0.0, std=config.initializer_range)

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
        embeddings = words_embeddings
        if self.type_id_enable:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        if self.position_enable:
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertModelSimple(nn.Module):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModelSimple(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config: BertConfig,
                 pretrain_embeddings,
                 type_id_enable=True,
                 position_enable=True):
        """Constructor for BertModelSimple.

        Args:
            config: `BertConfig` instance.
        """
        super(BertModelSimple, self).__init__()
        self.embeddings = BERTSimpleEmbeddings(config, pretrain_embeddings,
                                               type_id_enable=type_id_enable,
                                               position_enable=position_enable)

        self.encoder = BERTEncoder(config)
        self.pooler = BERTPooler(config)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
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
        all_encoder_layers = self.encoder(embedding_output, extended_attention_mask)
        sequence_output = all_encoder_layers[-1]
        pooled_output = self.pooler(sequence_output)
        return all_encoder_layers, pooled_output


class BertSimpleForSequenceClassification(nn.Module):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 2, 0]])

    config = BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    num_labels = 2

    model = BertForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, pretrain_embeddings, num_labels,
                 type_id_enable=True,
                 position_enable=True,
                 init_weight=False):
        super(BertSimpleForSequenceClassification, self).__init__()
        self.bert = BertModelSimple(config, pretrain_embeddings,
                                    type_id_enable, position_enable)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)

        if init_weight:
            print("init_weight = True")
            def init_weights(module):
                # Embedding module is pretrain, we will need to bypass here to avoid
                # overloading to zeros.
                if isinstance(module, (nn.Linear)):
                    # Slightly different from the TF version which uses truncated_normal for initialization
                    # cf https://github.com/pytorch/pytorch/pull/5617
                    module.weight.data.normal_(mean=0.0, std=config.initializer_range)
                elif isinstance(module, BERTLayerNorm):
                    module.beta.data.normal_(mean=0.0, std=config.initializer_range)
                    module.gamma.data.normal_(mean=0.0, std=config.initializer_range)
                if isinstance(module, nn.Linear):
                    module.bias.data.zero_()
            self.apply(init_weights)

    def set_context_feature(self, unique_context_feature):
        self.unique_context_feature = unique_context_feature

    def forward(self, input_ids, token_type_ids, attention_mask, seq_lens,
                device=None, labels=None):
        _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits