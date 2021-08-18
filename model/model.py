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
    def __init__(self, hidden_dim, n_layer, dropout, n_heads, intermediate_dim, n_class):
        super(Model_MNS, self).__init__()
        self.bert_config = BertConfig(hidden_size=hidden_dim,
                                      num_hidden_layers=n_layer,
                                      num_attention_heads=n_heads,
                                      hidden_dropout_prob=dropout,
                                      intermediate_size=intermediate_dim)
        self.dialogue_encoder = DialogueEncoder(bert_config=self.bert_config)
        self.classifier=nn.Linear(hidden_dim, n_class)
        self.activation = nn.ReLU()
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, utterances, umask):
        utterances = utterances.permute(1,0,2)
        outputs = self.dialogue_encoder(input_embeds=utterances, umask=umask)

        s_token = outputs[1]
        
        linear_output = self.classifier(s_token)
        activation_output = self.activation(linear_output)

        softmax_output = self.log_softmax(activation_output)

        return softmax_output  