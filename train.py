import argparse
import numpy as np, pickle, time, argparse
import pickle
import time

import torch
import torch.nn as nn
import torch.optim as optim

from model.model import Model_MNS
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler
from tqdm import tqdm
from utils.dataloader import Dataset

np.random.seed(1234)

def get_data_loader(path, multigpu, batch_size=16, num_workers=0, pin_memory=False):
    dataset = Dataset(path)
    n_dataset = dataset.len
    train_len = int(n_dataset * 0.8)
    valid_len = int(n_dataset * 0.1)
    test_len = n_dataset - train_len - valid_len
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_len, valid_len, test_len])
    n_gpu = torch.cuda.device_count()

    if multigpu:
        train_sampler = DistributedSampler(train_dataset)
        valid_sampler = DistributedSampler(valid_dataset)
        test_sampler = DistributedSampler(test_dataset)
    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = RandomSampler(valid_dataset)
        test_sampler = RandomSampler(test_dataset)

    train_dataloader = DataLoader(dataset=train_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  sampler=train_sampler, 
                                  collate_fn=dataset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    valid_dataloader = DataLoader(dataset=valid_dataset, 
                                  batch_size=batch_size, 
                                  shuffle=False, 
                                  sampler=valid_sampler, 
                                  collate_fn=dataset.collate_fn,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)

    test_dataloader = DataLoader(dataset=test_dataset, 
                                 batch_size=batch_size, 
                                 shuffle=False, 
                                 sampler=test_sampler, 
                                 collate_fn=dataset.collate_fn,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)

    return train_dataloader, valid_dataloader, test_dataloader

def train_or_eval_model(model, loss_function, dataloader, epoch, device, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in tqdm(dataloader):
        if train:
            optimizer.zero_grad()
        utterances, speakers, label, umask, num_utterances, _ = data

        u = utterances.to(device)
        label = label.to(device)
        umask = umask.to(device)

        log_prob = model(u, umask) # seq_len, batch, n_classes

        print(log_prob.shape)
        # lp_ = log_prob.transpose(0,1).contiguous().view(-1,log_prob.size()[2]) # batch*seq_len, n_classes
        # labels_ = label.view(-1) # batch*seq_len

        label = label.view(-1)
        loss = loss_function(log_prob, label)

        pred_ = torch.argmax(log_prob,1) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        losses.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses),4)
    avg_accuracy = round(accuracy_score(labels,preds)*100,2)
    avg_fscore = round(f1_score(labels,preds,average='weighted')*100,2)
    return avg_loss, avg_accuracy, labels, preds, avg_fscore, epoch 

if __name__ == '__main__' :
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='MNS', metavar='TASK',
                        help='training_object')
    parser.add_argument('--multigpu', default=False, metavar='multi-gpu',
                        help='multi-gpu')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=60, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')

    args = parser.parse_args()

    print(args) 

    args.cuda = torch.cuda.is_available()
    n_epochs   = args.epochs

    if args.cuda:
        print('Running on GPU')
        device = torch.device("cuda")
    else:
        print('Running on CPU')
        device = torch.device("cpu")

    hidden_dim=1024
    n_layer=12
    dropout=0.3
    n_heads=8
    intermediate_dim=hidden_dim*4
    n_class=8

    if args.task == 'MNS':
        model = Model_MNS(hidden_dim=hidden_dim, n_layer=n_layer, 
                          dropout=dropout, n_heads=n_heads, 
                          intermediate_dim=intermediate_dim, 
                          n_class=n_class) #hidden_dim, n_layer, dropout, n_heads, intermediate_dim, n_class

    else:
        model = Model_MNS(hidden_dim=hidden_dim, n_layer=n_layer, 
                          dropout=dropout, n_heads=n_heads, 
                          intermediate_dim=intermediate_dim, 
                          n_class=n_class) #hidden_dim, n_layer, dropout, n_heads, intermediate_dim, n_class

    model = model.to(device)
    if args.multigpu:
        model = nn.parallel.DistributedDataParallel(model)

    path = r'./dataset/data.pkl'

    train_dataloader, valid_dataloader, test_dataloader = get_data_loader(path, args.multigpu, batch_size=32, num_workers=0, pin_memory=False)

    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)
    
    loss_function = torch.nn.NLLLoss()

    for e in range(n_epochs):
        start_time = time.time()
        train_loss, train_acc, _,_,_,train_fscore, _= train_or_eval_model(model, loss_function,
                                               train_dataloader, e, device, optimizer, True)
        valid_loss, valid_acc, _,_,_,val_fscore, _= train_or_eval_model(model, loss_function, valid_dataloader, e, device)
        test_loss, test_acc, test_label, test_pred, test_mask, test_fscore, epoch= train_or_eval_model(model, loss_function, test_dataloader, e, device)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask =\
                    test_loss, test_label, test_pred, test_mask

            #path = './weights/' + args.task + '/model_' + str(epoch) + '.pt'
            path = './weights/' + args.task + '/model.pt'
            torch.save(model.state_dict(), path)

        print('epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore{} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))