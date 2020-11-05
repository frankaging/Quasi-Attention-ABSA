import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss

def mask(seq_len):
    batch_size = len(seq_len)
    max_len = max(seq_len)
    mask = torch.zeros(batch_size, max_len, dtype=torch.float)
    for i in range(mask.size()[0]):
        mask[i,:seq_len[i]] = 1
    return mask

class BiLSTMPooler(nn.Module):
    def __init__(self, hidden_size):
        super(BiLSTMPooler, self).__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        #return first_token_tensor
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

'''
Model class
'''
class BiLSTM(nn.Module):
    def __init__(self,
                 pretrain_embeddings=None,
                 freeze=True,
                 embedding_dim=300,
                 hidden_dim=150,
                 output_dim=3,
                 bidirectional=True):
        super(BiLSTM, self).__init__()
        # Word embedding layers
        self.embeddings = nn.Embedding.from_pretrained(
                            pretrain_embeddings,
                            freeze=freeze,
                            padding_idx=0)

        # BiLSTM as the encoder for each utterance.
        self.hidden_dim = hidden_dim
        self.encoder = nn.LSTM(embedding_dim, hidden_dim,
                               batch_first=True, bidirectional=bidirectional)
        self.num_directions = 2 if bidirectional else 1
        self.lstm_layers = 1

        # Pooler
        self.pooler = BiLSTMPooler(hidden_dim*2)

        # For specific decoding token, we will decode the label as MLP.
        out_hidden_dim = 64
        self.output_dim = output_dim
        self.decoder_in_dim = hidden_dim*2
        self.decoder = nn.Sequential(nn.Linear(self.decoder_in_dim,
                                               out_hidden_dim),
                                     nn.ReLU(),
                                     nn.Dropout(0.1),
                                     nn.Linear(out_hidden_dim, self.output_dim))
        self.dropout = nn.Dropout(0.1)

    def set_context_feature(self, unique_context_feature):
        self.unique_context_feature = unique_context_feature

    def forward(self, input_ids, token_type_ids, attention_mask, lengths, device=None, labels=None):
        batch_num = input_ids.shape[0]

        # embeddings
        embedding_output = self.embeddings(input_ids)

        # pack inputs
        lengths = lengths.squeeze(dim=-1)
        inputs_packed = pack_padded_sequence(embedding_output, lengths,
                                     batch_first=True, enforce_sorted=False)
        # bi-lstm encoder
        self.h, self.c = \
            (Variable(torch.zeros(self.lstm_layers * self.num_directions,
                                  batch_num, self.hidden_dim).to(device)),
             Variable(torch.zeros(self.lstm_layers * self.num_directions,
                                  batch_num, self.hidden_dim).to(device)))
        outputs, (_, _) = self.encoder(inputs_packed, (self.h, self.c))
        outputs_unpacked, _ = \
            pad_packed_sequence(outputs, batch_first=True)

        # pooler
        pooled_output = self.pooler(outputs_unpacked)

        # mlp decoder
        logits = self.decoder(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss, logits
        else:
            return logits