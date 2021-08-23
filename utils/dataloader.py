import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import pickle
import pandas as pd

class Dataset_MNS(Dataset):
    def __init__(self, path):
        self.dataset= pickle.load(open(path, 'rb'), encoding='latin1') 

        self.ids = self.dataset['vids']
        self.utterances = self.dataset['utterances']
        self.encoded_utterances = self.dataset['encoded_utterance']
        self.speakers = self.dataset['speakers']
        self.num_utterances = self.dataset['num_utterances']
        self.num_speakers = self.dataset['num_speakers']

        self.len = len(self.ids)

    def __getitem__(self, index):        
        encoded_utterances = self.encoded_utterances[index]
        num_speaker = []
        utterances = torch.cat([utterance for utterance in encoded_utterances], dim=0)
        num_speaker.append(self.num_speakers[index])
        return utterances,\
               torch.FloatTensor(self.speakers[index]),\
               torch.LongTensor(num_speaker),\
               torch.FloatTensor([1]*(self.num_utterances[index]+2)),\
               self.num_utterances[index],\
               index        

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i<4 else dat[i] for i in dat]

class Dataset_SCP(Dataset):
    def __init__(self, path):
        self.dataset= pickle.load(open(path, 'rb'), encoding='latin1') 

        self.contexts = self.dataset['contexts']
        self.utterance1 = self.dataset['utt1s']
        self.utterance2 = self.dataset['utt2s']
        self.labels = self.dataset['labels']
        
        self.len = len(self.contexts)

    def __getitem__(self, index):
        label = [] 
        label.append(self.labels[index])
        return torch.FloatTensor(self.contexts[index]),\
               torch.FloatTensor(self.utterance1[index]),\
               torch.FloatTensor(self.utterance2[index]),\
               torch.FloatTensor([1]*(len(self.contexts[index])+6)),\
               torch.FloatTensor(label),\
               index        

    def __len__(self):
        return self.len
    
    def collate_fn(self, data):
        dat = pd.DataFrame(data)

        return [pad_sequence(dat[i]) if i<5 else dat[i] for i in dat]