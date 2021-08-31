import argparse
import logging
import numpy as np, pickle, time, argparse
import pickle5 as pickle
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim

from model.model import Model_MNS, Model_SCP
from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, classification_report, precision_recall_fscore_support

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.sampler import RandomSampler

from tqdm.auto import tqdm
from tqdm import trange
from utils.dataloader import Dataset_SCP, Dataset_MNS

np.random.seed(1234)

def get_data_loader(path, args, batch_size=16, num_workers=0, pin_memory=False):
    if args.task == 'MNS':
        dataset = Dataset_MNS(path)
    else:
        dataset = Dataset_SCP(path)

    n_dataset = dataset.len

    train_len = int(n_dataset * 0.8)
    valid_len = int(n_dataset * 0.1)
    test_len = n_dataset - train_len - valid_len
    train_dataset, valid_dataset, test_dataset = random_split(dataset, [train_len, valid_len, test_len])
    
    multigpu = args.multigpu

    if multigpu:
        train_sampler = DistributedSampler(train_dataset, num_replicas=args.world_size, rank=args.gpu)
        valid_sampler = DistributedSampler(valid_dataset, num_replicas=args.world_size, rank=args.gpu)
        test_sampler = DistributedSampler(test_dataset, num_replicas=args.world_size, rank=args.gpu)
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
    for step, data in tqdm(enumerate(dataloader)):
        if train:
            optimizer.zero_grad()
        utterances, speakers, label, umask, num_utterances, _ = data

        u = utterances.to(device)
        label = label.to(device)
        umask = umask.to(device)

        log_prob = model(u, umask) # seq_len, batch, n_classes

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

def train_or_eval_model2(model, loss_function, dataloader, epoch, device, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in tqdm(dataloader, position=0, leave=True):
        if train:
            optimizer.zero_grad()
        contexts, utt1s, utt2s, umasks, label, _ = data
        contexts = contexts.to(device)
        utt1s = utt1s.to(device)
        utt2s = utt2s.to(device)
        label = label.to(device)
        umasks = umasks.to(device)
        log_prob = model(contexts, utt1s, utt2s, umasks) # seq_len, batch, n_classes
        label = label.view(-1)
        loss = loss_function(log_prob, label)

        pred_ = torch.round(log_prob) # batch*seq_len
        preds.append(pred_.data.cpu().numpy())
        labels.append(label.data.cpu().numpy())

        losses.append(loss.item())
        if train:
            loss.backward()
            optimizer.step()

        del contexts, utt1s, utt2s, label
        del loss
        del pred_
        del log_prob

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
    else:
        return float('nan'), float('nan'), [], [], [], float('nan'),[]

    avg_loss = round(np.sum(losses),4)
    avg_accuracy = round(accuracy_score(labels,preds)*100,2)
    avg_fscore = round(f1_score(labels,preds,average='weighted')*100,2)
    return avg_loss, avg_accuracy, labels, preds, avg_fscore, epoch 

def main(args):
    args.node = 1
    args.cuda = torch.cuda.is_available()

    if args.cuda:
        print('Running on GPU')
        device = torch.device("cuda")
    else:
        print('Running on CPU')
        device = torch.device("cpu")

    logger = logging.getLogger()

    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node * args.node
    mp.spawn(main_worker, nprocs=ngpus_per_node, 
             args=(ngpus_per_node, args, device, logger))
    
    
def main_worker(gpu, ngpus_per_node, args, device, logger):
    args.gpu = gpu
    args.rank = gpu
    torch.cuda.set_device(args.gpu)
    
    print("Use GPU: {} for training".format(args.gpu))
    #args.rank = args.rank * ngpus_per_node + gpu

    dist.init_process_group(backend='nccl', 
                            init_method='tcp://127.0.0.1:3456',
                            world_size=args.world_size, 
                            rank=gpu)
    hidden_dim=768
    n_layer=12
    dropout=0.3
    n_heads=12
    intermediate_dim=hidden_dim*4

    batch_size = args.batch_size
    num_worker = 4 * ngpus_per_node

    batch_size = int(batch_size / ngpus_per_node)
    num_worker = int(num_worker / ngpus_per_node)

    if args.task == 'MNS':
        n_class=9
        model = Model_MNS(hidden_dim=hidden_dim, n_layer=n_layer, 
                          dropout=dropout, n_heads=n_heads, 
                          intermediate_dim=intermediate_dim, 
                          n_class=n_class,
                          device=device) #hidden_dim, n_layer, dropout, n_heads, intermediate_dim, n_class

        path = r'./dataset/data.pkl'
        train_dataloader, valid_dataloader, test_dataloader = get_data_loader(path, args, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
        loss_function = torch.nn.NLLLoss()

    else:
        n_class=2
        model = Model_SCP(hidden_dim=hidden_dim, n_layer=n_layer, 
                          dropout=dropout, n_heads=n_heads, 
                          intermediate_dim=intermediate_dim, 
                          n_class=n_class,
                          device=device) #hidden_dim, n_layer, dropout, n_heads, intermediate_dim, n_class

        path = r'./dataset/data2_base.pkl'
        train_dataloader, valid_dataloader, test_dataloader = get_data_loader(path, args, batch_size=batch_size, num_workers=num_worker, pin_memory=True)
        loss_function = torch.nn.BCELoss()

    model = model.to(device)

    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True)

    optimizer = optim.AdamW(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)
    

    best_loss, best_label, best_pred, best_mask = None, None, None, None

    n_epochs = args.epochs

    for e in tqdm(range(n_epochs), position=0, leave=True):
        start_time = time.time()
        if args.task =='MNS':
            train_loss, train_acc, _,_,train_fscore, _= train_or_eval_model(model, loss_function,
                                                train_dataloader, e, device, optimizer, True)
            valid_loss, valid_acc, _,_,val_fscore, _= train_or_eval_model(model, loss_function, valid_dataloader, e, device)
            test_loss, test_acc, test_label, test_pred, test_fscore, epoch= train_or_eval_model(model, loss_function, test_dataloader, e, device)
        else:
            train_loss, train_acc, _,_,train_fscore, _= train_or_eval_model2(model, loss_function, train_dataloader, e, device, optimizer, True)
            valid_loss, valid_acc, _,_,val_fscore, _= train_or_eval_model2(model, loss_function, valid_dataloader, e, device)
            test_loss, test_acc, test_label, test_pred, test_fscore, epoch= train_or_eval_model2(model, loss_function, test_dataloader, e, device)
        
        if gpu == 0:
            logger.setLevel(logging.DEBUG)
            log_path = './logs/' + args.task +'_output.log'
            file_handler = logging.FileHandler(log_path)
            logger.addHandler(file_handler)
            logger.debug('epoch {} train_loss {} train_acc {} train_fscore{} valid_loss {} valid_acc {} val_fscore{} test_loss {} test_acc {} test_fscore {} time {}'.\
                format(e+1, train_loss, train_acc, train_fscore, valid_loss, valid_acc, val_fscore,\
                        test_loss, test_acc, test_fscore, round(time.time()-start_time,2)))

            if best_loss == None or best_loss > test_loss:
                best_loss = test_loss

                #path = './weights/' + args.task + '/model_' + str(epoch) + '.pt'
                path = './weights/' + args.task + '_model.pt'
                torch.save(model.state_dict(), path)
                best_epoch = e

    logger.debug(best_epoch)
    
if __name__ == '__main__' :
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--task', default='MNS', metavar='TASK',
                        help='training_object')
    parser.add_argument('--multigpu', default=True, metavar='multi-gpu',
                        help='multi-gpu')
    parser.add_argument('--lr', type=float, default=0.00005, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.00001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--batch-size', type=int, default=32, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs')
    parser.add_argument('--class-weight', action='store_true', default=True,
                        help='class weight')
    parser.add_argument("--local_rank", type=int, default=0)


    args = parser.parse_args()
    main(args)