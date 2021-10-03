import torch
import torch.nn as nn
from torch.nn  import CrossEntropyLoss, MSELoss, NLLLoss

import torch.nn.functional as F

from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, BertConfig, BertTokenizer, BertForPreTraining, BertLMHeadModel
from transformers.models.bert.modeling_bert import BertPredictionHeadTransform

from copy import deepcopy

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
        self.cls_token = torch.LongTensor([0]).cuda(device)
        self.sep_token = torch.LongTensor([1]).cuda(device)
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

        result = result.cuda(self.device)

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
        self.cls_token = torch.LongTensor([0]).cuda(device)
        self.sep_token = torch.LongTensor([1]).cuda(device)
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

        s_token_mask = s_token_mask.cuda(self.device)
        h_token_mask = h_token_mask.cuda(self.device)

        extended_umask = torch.cat((s_token_mask, umask, h_token_mask), dim=1)

        return extended_umask

    def forward(self, contexts, utt1, utt2, umask): #utterance (seq_len, batch_size, dim)
        contexts = contexts.squeeze(2)

        n_contexts = torch.cat([self.input_layer(utterance.unsqueeze(0)) for utterance in contexts], dim=0)
        
        utt1 = self.input_layer(utt1)
        utt2 = self.input_layer(utt2)

        utterances = self.add_token(n_contexts, utt1, utt2)
        #utterances = torch.cat((utt1, utt2), dim=1)
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

class Model_SAE(nn.Module): #speaker autoencoder
    def __init__(self, device, vocab_size=30522, hidden_size=768, num_hidden_layers=12, num_attention_heads=12, intermediate_dim=3072):
        super(Model_SAE, self).__init__()

        self.utt_pretrained_encoder = BertModel.from_pretrained('bert-base-uncased')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_config = BertConfig(vocab_size=vocab_size, hidden_size=hidden_size, num_hidden_layers=num_hidden_layers,
                                         num_attention_heads=num_attention_heads, intermediate_dim=intermediate_dim)
        self.utt_encoder = BertModel(self.bert_config)
        self.speaker_embedding = nn.Embedding(3, hidden_size)
        self.context_encoder = DialogueEncoder(self.bert_config)
        self.context_mlm_trans = BertPredictionHeadTransform(self.bert_config)
        self.dropout = nn.Dropout(self.bert_config.hidden_dropout_prob)
        self.append_linear = nn.Linear(hidden_size*2, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = device
        # self.decoder_config = deepcopy(self.bert_config)
        # self.decoder_config.is_decoder=True
        # self.decoder_config.add_cross_attention=True
        # self.decoder = BertLMHeadModel(self.decoder_config)
    
    def utt_encoding(self, utts, utt_masks, pretrained=True):
        batch_size, max_ctx_len, max_utt_len = utts.size()
        utts = utts.view(-1, max_utt_len)
        utt_masks = utt_masks.view(-1, max_utt_len)
        if pretrained == True:
            outputs = self.utt_pretrained_encoder(utts, utt_masks)
        else:
            outputs = self.utt_encoder(utts, utt_masks)
        utts_encodings = outputs[1]
        utts_encodings = utts_encodings.view(batch_size, max_ctx_len, -1)
        return utts_encodings

    def context_encoding(self, utts, utts_masks, dialogue_masks, masked_speakers, mode='append'):
        encoded_speaker = self.speaker_embedding(masked_speakers)
        utt_encodings = self.utt_encoding(utts, utts_masks)

        if mode == 'append':
            final_encodings = torch.cat((utt_encodings, encoded_speaker), dim=2)
            final_encodings = self.append_linear(final_encodings)
        else:
            final_encodings = utt_encodings + encoded_speaker

        outputs = self.context_encoder(final_encodings, dialogue_masks)

        context_hiddens = outputs[0]
        pooled_output = outputs[1]

        return context_hiddens, pooled_output

    def forward(self, utts, utt_masks, dialogue_masks, masked_speakers, labels, label_speakers, label_masks, mode='append'): #dialogues (batch, length, size)
        label_speakers_long = label_speakers.type(torch.LongTensor)
        label_speakers_long = label_speakers_long.cuda(self.device)

        context_hiddens, _ = self.context_encoding(utts, utt_masks, dialogue_masks, masked_speakers)
        predict_encodings = self.context_mlm_trans(self.dropout(context_hiddens))

        with torch.no_grad():
            label_encodings = self.utt_encoding(labels, label_masks)
            speaker_embedded = self.speaker_embedding(label_speakers_long)

            if mode == 'append':
                label_encodings = torch.cat((label_encodings, speaker_embedded), dim=2)
                label_encodings = self.append_linear(label_encodings)
            else:
                label_encodings = label_encodings + speaker_embedded

        loss_mlm = MSELoss()(predict_encodings, label_encodings) # [num_selected_utts x dim]

        predict_encodings = self.linear(predict_encodings)

        predict_speakers = self.sigmoid(predict_encodings).squeeze(2)

        loss_speakers = torch.nn.BCELoss()(predict_speakers, label_speakers)

        final_loss = loss_mlm + loss_speakers

        return final_loss, loss_mlm, loss_speakers
