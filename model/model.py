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
        utterances = utterances.squeeze(2)
        utterances = self.add_token(utterances)
        utterances = utterances.permute(1,0,2)
        outputs = self.dialogue_encoder(input_embeds=utterances, umask=umask)

        s_token = outputs[1]
        
        linear_output = self.classifier(s_token)
        activation_output = self.activation(linear_output)

        softmax_output = self.log_softmax(activation_output)

        return softmax_output  


class Model_SCP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layer, dropout, n_heads, intermediate_dim, n_class, device):
        super(Model_SCP, self).__init__()
        self.bert_config = BertConfig(hidden_size=hidden_dim,
                                      num_hidden_layers=n_layer,
                                      num_attention_heads=n_heads,
                                      hidden_dropout_prob=dropout,
                                      intermediate_size=intermediate_dim)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.cls_token = torch.LongTensor([0]).to(device)
        self.sep_token = torch.LongTensor([1]).to(device)
        self.token_embedding = nn.Embedding(2, hidden_dim)
        self.dialogue_encoder = DialogueEncoder(bert_config=self.bert_config)
        self.fc1=nn.Linear(hidden_dim, int(hidden_dim/2))
        self.fc2=nn.Linear(int(hidden_dim/2), 1)
        #self.fc3=nn.Linear(100, 1)

        self.sigmoid = nn.Sigmoid()
        self.device = device

    def add_token(self, contexts, utt1, utt2):
        batch_size = contexts.shape[1]
        cls_t = self.token_embedding(self.cls_token)
        cls_t = cls_t.repeat(1, batch_size, 1)
        sep_t = self.token_embedding(self.sep_token)
        sep_t = sep_t.repeat(1, batch_size, 1)
 
        result = torch.cat((cls_t, contexts, sep_t, utt1, sep_t, utt2, sep_t), dim=0)
      
        result = result.to(self.device)

        return result
    def extending_umask(self, umask):
        batch_size = umask.shape[0]

        s_token_mask = torch.ones(batch_size, 1)
        h_token_mask = torch.ones(batch_size, 5)

        s_token_mask = s_token_mask.to(self.device)
        h_token_mask = h_token_mask.to(self.device)

        extended_umask = torch.cat((s_token_mask, umask, h_token_mask), dim=1)

        return extended_umask

    def forward(self, contexts, utt1, utt2, umask): #utterance (seq_len, batch_size, dim)
        contexts = contexts.squeeze(2)

        n_contexts = torch.cat([self.input_layer(utterance.unsqueeze(0)) for utterance in contexts], dim=0)
        
        utt1 = self.input_layer(utt1)
        utt2 = self.input_layer(utt2)

        #utterances = self.add_token(n_contexts, utt1, utt2)
        utterances = torch.cat((utt1, utt2), dim=1)
        utterances = utterances.permute(1,0,2)
        umask = umask.permute(1, 0)
        umask = self.extending_umask(umask)

        outputs = self.dialogue_encoder(input_embeds=utterances, umask=umask)

        s_token = outputs[1]
        
        linear_output1 = self.fc1(s_token)
        linear_output2 = self.fc2(linear_output1)
        #linear_output3 = self.fc3(linear_output2)

        output = self.sigmoid(linear_output2)
        output = output.squeeze(1)

        return output