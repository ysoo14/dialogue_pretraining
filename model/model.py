import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertConfig

class DialogueEncoder(nn.Module):
    def __init__(self, bert_config):
        super(DialogueEncoder, self).__init__()
        self.bert = BertModel(bert_config)
    
    def forward(self, input_embeds, umask):
        return self.bert(inputs_embeds=input_embeds, encoder_attention_mask=umask)

class Model_MNS(nn.Module): #matching the number of speakers
    def __init__(self, hidden_dim, n_layer, dropout, n_heads, intermediate_dim, n_class, device):
        super(Model_MNS, self).__init__()
        self.bert_config = BertConfig(hidden_size=hidden_dim,
                                      num_hidden_layers=n_layer,
                                      num_attention_heads=n_heads,
                                      hidden_dropout_prob=dropout,
                                      intermediate_size=intermediate_dim,
                                      max_position_embeddings=512)
        self.cls_token = torch.LongTensor([0]).to(device)
        self.sep_token = torch.LongTensor([1]).to(device)
        self.token_embedding = nn.Embedding(2, hidden_dim)
        self.dialogue_encoder = DialogueEncoder(bert_config=self.bert_config)
        self.classifier=nn.Linear(hidden_dim, n_class)
        self.activation = nn.ReLU()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)
        self.device = device

    def add_token(self, utterances):
        batch_size = utterances.shape[1]
        cls_t = self.token_embedding(self.cls_token)
        cls_t = cls_t.repeat(1, batch_size, 1)
        sep_t = self.token_embedding(self.sep_token)
        sep_t = sep_t.repeat(1, batch_size, 1)

        result = torch.cat((cls_t, utterances, sep_t), dim=0)

        result = result.to(self.device)

        return result

    def forward(self, utterances, umask): #utterance (seq_len, batch_size, dim)
        utterances = self.add_token(utterances)
        utterances = utterances.permute(1,0,2)
        outputs = self.dialogue_encoder(input_embeds=utterances, umask=umask)

        s_token = outputs[1]
        
        linear_output = self.classifier(s_token)
        activation_output = self.activation(linear_output)

        softmax_output = self.log_softmax(activation_output)

        return softmax_output  


class Model_SCP(nn.Module):
    def __init__(self, hidden_dim, n_layer, dropout, n_heads, intermediate_dim, n_class, device):
        super(Model_SCP, self).__init__()
        self.bert_config = BertConfig(hidden_size=hidden_dim,
                                      num_hidden_layers=n_layer,
                                      num_attention_heads=n_heads,
                                      hidden_dropout_prob=dropout,
                                      intermediate_size=intermediate_dim)
        self.cls_token = torch.LongTensor([0]).to(device)
        self.sep_token = torch.LongTensor([1]).to(device)
        self.token_embedding = nn.Embedding(2, hidden_dim)
        self.dialogue_encoder = DialogueEncoder(bert_config=self.bert_config)
        self.classifier=nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = device

    def add_token(self, contexts, utt1, utt2):
        contexts = contexts.squeeze(2)
        batch_size = contexts.shape[1]
        cls_t = self.token_embedding(self.cls_token)
        cls_t = cls_t.repeat(1, batch_size, 1)
        sep_t = self.token_embedding(self.sep_token)
        sep_t = sep_t.repeat(1, batch_size, 1)

        result = torch.cat((cls_t, contexts, sep_t, utt1, sep_t, utt2, sep_t), dim=0)

        result = result.to(self.device)

        return result

    def forward(self, contexts, utt1, utt2, umask): #utterance (seq_len, batch_size, dim)
        utterances = self.add_token(contexts, utt1, utt2)
        utterances = utterances.permute(1,0,2)
        outputs = self.dialogue_encoder(input_embeds=utterances, umask=umask)

        s_token = outputs[1]
        
        linear_output = self.classifier(s_token)
        output = self.sigmoid(linear_output)
        output = output.squeeze(1)
        return output