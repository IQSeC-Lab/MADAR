import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time, random
from tqdm import tqdm
from sklearn.metrics import classification_report

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict


def make_family_based_dict(X_train, Y_train, Y_train_family, task_month, global_family_dict):
    count = 0
    for x_ind, x_sample in enumerate(X_train):
        count += 1
        #print(x_ind, Y_train[x_ind])

        if Y_train[x_ind] == 0:
            global_family_dict["goodware"].append(x_sample)
        if Y_train[x_ind] == 1:
            if Y_train_family[x_ind] == '':
                global_family_dict["others_family"].append(x_sample)
            else:
                global_family_dict[Y_train_family[x_ind]].append(x_sample)

    print(f'Task {task_month} and #-of new samples stored {count}')
    
    return global_family_dict


    
    
def get_family_labeled_task_test_data(data_dir, task_years, mlp_net=False):
    
    X_te, Y_te, Y_te_family = get_family_labeled_year_data(data_dir, task_years[-1], train=False)
    
    for year in task_years[:-1]:
        pre_X_te, pre_Y_te, pre_Y_te_family = get_family_labeled_year_data(data_dir, year, train=False)
        X_te, Y_te, Y_te_family = np.concatenate((X_te, pre_X_te)), np.concatenate((Y_te, pre_Y_te)),\
                                np.concatenate((Y_te_family, pre_Y_te_family))
        

    X_test, Y_test, Y_test_family = X_te, Y_te, Y_te_family
    print(f'X_test {X_test.shape} Y_test {Y_test.shape} Y_te_family {Y_te_family.shape}')
    
    return X_test, Y_test, Y_test_family


    
def get_family_labeled_year_data(data_dir, year, train=True):
    
    if train:
        data_dir = data_dir + '/'
        XY_train = np.load(data_dir + str(year) + '_Domain_AZ_Train_Transformed.npz', allow_pickle=True)
        X_tr, Y_tr, Y_tr_family = XY_train['X_train'], XY_train['Y_train'], XY_train['Y_tr_family']
        print(f'X_train {X_tr.shape} Y_train {Y_tr.shape} Y_tr_family {Y_tr_family.shape}')
        
        return X_tr, Y_tr, Y_tr_family
    else:
        data_dir = data_dir + '/'
        XY_test = np.load(data_dir + str(year) + '_Domain_AZ_Test_Transformed.npz', allow_pickle=True)
        X_test, Y_test, Y_test_family = XY_test['X_test'], XY_test['Y_test'], XY_test['Y_te_family']

        return X_test, Y_test, Y_test_family 






def get_replay_samples(global_family_dict, num_samples_per_malware_family):
    pre_malware_samples = []

    cnt = 0
    for k in global_family_dict.keys():
        if k != 'goodware':
            cnt += 1
            if num_samples_per_malware_family > len(global_family_dict[k]):
                selected_family_samples = random.sample(global_family_dict[k], len(global_family_dict[k]))
            else:
                selected_family_samples = random.sample(global_family_dict[k], num_samples_per_malware_family)

            #print(selected_family_samples)
            for sample in selected_family_samples:
                pre_malware_samples.append(sample)
                
    if len(global_family_dict['goodware']) < len(pre_malware_samples):
        pre_goodware_samples = random.sample(global_family_dict['goodware'], len(global_family_dict['goodware']))
    else:
        pre_goodware_samples = random.sample(global_family_dict['goodware'], len(pre_malware_samples))

    samples_to_replay = np.concatenate((np.array(pre_goodware_samples), np.array(pre_malware_samples)))
    labels_to_replay = np.concatenate((np.zeros(len(pre_goodware_samples)), np.ones(len(pre_malware_samples))))


    print(f'X_replay {samples_to_replay.shape} Y_replay {labels_to_replay.shape}')
    print(f'Replay {len(pre_malware_samples)} malware samples of {len(global_family_dict.keys()) -1} families')
    print(f'and Replay {len(pre_goodware_samples)} goodware samples')
    
    
    return samples_to_replay, labels_to_replay





def get_replay_samples_first(global_family_dict, num_samples_per_malware_family):
    pre_malware_samples = []

    cnt = 0
    for k in global_family_dict.keys():
        if k != 'goodware':
            cnt += 1
            if num_samples_per_malware_family > len(global_family_dict[k]):
                selected_family_samples = global_family_dict[k] 
                #random.sample(global_family_dict[k], len(global_family_dict[k]))
            else:
                selected_family_samples = global_family_dict[k][:num_samples_per_malware_family]
                #random.sample(global_family_dict[k], num_samples_per_malware_family)

            #print(selected_family_samples)
            for sample in selected_family_samples:
                pre_malware_samples.append(sample)
                
    if len(global_family_dict['goodware']) < len(pre_malware_samples):
        pre_goodware_samples = global_family_dict['goodware'] 
        #random.sample(global_family_dict['goodware'], len(global_family_dict['goodware']))
    else:
        pre_goodware_samples = global_family_dict['goodware'][:len(pre_malware_samples)] 
        #random.sample(global_family_dict['goodware'], len(pre_malware_samples))

    samples_to_replay = np.concatenate((np.array(pre_goodware_samples), np.array(pre_malware_samples)))
    labels_to_replay = np.concatenate((np.zeros(len(pre_goodware_samples)), np.ones(len(pre_malware_samples))))


    print(f'X_replay {samples_to_replay.shape} Y_replay {labels_to_replay.shape}')
    print(f'Replay {len(pre_malware_samples)} malware samples of {len(global_family_dict.keys()) -1} families')
    print(f'and Replay {len(pre_goodware_samples)} goodware samples')
    
    
    return samples_to_replay, labels_to_replay


def get_replay_samples_last(global_family_dict, num_samples_per_malware_family):
    pre_malware_samples = []

    cnt = 0
    for k in global_family_dict.keys():
        if k != 'goodware':
            cnt += 1
            if num_samples_per_malware_family > len(global_family_dict[k]):
                selected_family_samples = global_family_dict[k] 
                #random.sample(global_family_dict[k], len(global_family_dict[k]))
            else:
                selected_family_samples = global_family_dict[k][-num_samples_per_malware_family:]
                #random.sample(global_family_dict[k], num_samples_per_malware_family)

            #print(selected_family_samples)
            for sample in selected_family_samples:
                pre_malware_samples.append(sample)
                
    if len(global_family_dict['goodware']) < len(pre_malware_samples):
        pre_goodware_samples = global_family_dict['goodware']
        #random.sample(global_family_dict['goodware'], len(global_family_dict['goodware']))
    else:
        pre_goodware_samples = global_family_dict['goodware'][-len(pre_malware_samples):]
        #random.sample(global_family_dict['goodware'], len(pre_malware_samples))

    samples_to_replay = np.concatenate((np.array(pre_goodware_samples), np.array(pre_malware_samples)))
    labels_to_replay = np.concatenate((np.zeros(len(pre_goodware_samples)), np.ones(len(pre_malware_samples))))


    print(f'X_replay {samples_to_replay.shape} Y_replay {labels_to_replay.shape}')
    print(f'Replay {len(pre_malware_samples)} malware samples of {len(global_family_dict.keys()) -1} families')
    print(f'and Replay {len(pre_goodware_samples)} goodware samples')
    
    
    return samples_to_replay, labels_to_replay



def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def get_partial_data(X, Y, replay_portion):
    indx = [i for i in range(len(Y))]
    random.shuffle(indx)

    replay_data_size = int(len(indx)*replay_portion)
    replay_index = indx[:replay_data_size]

    X_train = X[replay_index]
    Y_train = Y[replay_index]
    
    return X_train, Y_train


def get_month_data(data_dir, month, train=True):
    
    if train:
        data_dir = data_dir + str(month) + '/'
        XY_train = np.load(data_dir + 'XY_train.npz')
        X_tr, Y_tr = XY_train['X_train'], XY_train['Y_train']
        
        #indx = [i for i in range(len(Y_tr))]
        #random.shuffle(indx)

        #train_size = int(len(indx)*0.9)
        #trainset = indx[:train_size]
        #validset = indx[train_size:]

        # Separate the training set
        #X_train = X_tr[trainset]
        #Y_train = Y_tr[trainset]

        # Separate the valid set
        #X_valid = X_tr[validset]
        #Y_valid = Y_tr[validset]        
        #return X_train, Y_train, X_valid, Y_valid
        return X_tr, Y_tr
    else:
        data_dir = data_dir + str(month) + '/'
        XY_test = np.load(data_dir + 'XY_test.npz')
        X_test, Y_test = XY_test['X_test'], XY_test['Y_test']

        return X_test, Y_test        
        
def get_baseline_test_data(data_dir, task_months):
    
    X_te, Y_te = get_month_data(data_dir, task_months[-1], train=False)
    
    for month in task_months[:-1]:
        pre_X_te, pre_Y_te = get_month_data(data_dir, month, train=False)
        X_te, Y_te = np.concatenate((X_te, pre_X_te)), np.concatenate((Y_te, pre_Y_te))
        
    #X_te, Y_te  = reshape_features(X_te, Y_te)
    print(f'X_test {X_te.shape} Y_test {Y_te.shape}')
    
    return X_te, Y_te
    
def get_baseline_training_data(data_dir, task_months):
    
    X_tr, Y_tr, X_val, Y_val = get_month_data(data_dir, task_months[-1])
    
    #print(f'Current Task month {task_months[-1]} data X {X_tr.shape} Y {Y_tr.shape}')
    for month in task_months[:-1]:
        pre_X_tr, pre_Y_tr, pre_X_val, pre_Y_val = get_month_data(data_dir, month)
        #print(f'previous month {month} data X {pre_X_tr.shape} Y {pre_Y_tr.shape}')
        X_tr, Y_tr = np.concatenate((X_tr, pre_X_tr)), np.concatenate((Y_tr, pre_Y_tr))
        X_val, Y_val = np.concatenate((X_val, pre_X_val)), np.concatenate((Y_val, pre_Y_val))

        
    #X_tr, Y_tr  = reshape_features(X_tr, Y_tr)
    #X_val, Y_val  = reshape_features(X_val, Y_val)
    
    print(f'X_train {X_tr.shape} Y_train {Y_tr.shape}\n')
    print(f'X_valid {X_val.shape} Y_valid {Y_val.shape}\n')
    
    return X_tr, Y_tr, X_val, Y_val 
        
        
def get_task_partial_joint_training_data(data_dir, task_months, replay_portion, mlp_net=False):
    
    #X_tr, Y_tr, X_val, Y_val = get_month_data(data_dir, task_months[-1])
    X_tr, Y_tr = get_month_data(data_dir, task_months[-1])
    print(f'Current Task month {task_months[-1]} data X {X_tr.shape} Y {Y_tr.shape}')
    for month in task_months[:-1]:
        #pre_X_tr, pre_Y_tr, pre_X_val, pre_Y_val = get_month_data(data_dir, month)
        pre_X_tr, pre_Y_tr = get_month_data(data_dir, month)
        pre_X_tr, pre_Y_tr = get_partial_data(pre_X_tr, pre_Y_tr, replay_portion)
        
        print(f'previous month {month} data X {pre_X_tr.shape} Y {pre_Y_tr.shape}')
        
        X_tr, Y_tr = np.concatenate((X_tr, pre_X_tr)), np.concatenate((Y_tr, pre_Y_tr))
        #X_val, Y_val = np.concatenate((X_val, pre_X_val)), np.concatenate((Y_val, pre_Y_val))
    
    if not mlp_net:
        X_train, Y_train  = reshape_features(X_tr, Y_tr)
        X_valid, Y_valid = reshape_features(X_val, Y_val)
    else:
        X_train, Y_train  = X_tr, Y_tr
        #X_valid, Y_valid = X_val, Y_val
    print(f'X_train {X_train.shape} Y_train {Y_train.shape}\n')
    #print(f'X_valid {X_valid.shape} Y_valid {Y_valid.shape}\n')
    
    #return X_train, Y_train, X_valid, Y_valid
    return X_train, Y_train

def get_current_task_joint_test_data(data_dir, current_task):
    
    X_te, Y_te = get_month_data(data_dir, current_task, train=False)
    
    X_test, Y_test = reshape_features(X_te, Y_te)
    
    print(f'X_test {X_test.shape} Y_test {Y_test.shape}')
    
    return X_test, Y_test

def get_task_test_data(data_dir, task_months, mlp_net=False):
    
    X_te, Y_te = get_month_data(data_dir, task_months[-1], train=False)
    
    for month in task_months[:-1]:
        pre_X_te, pre_Y_te = get_month_data(data_dir, month, train=False)
        X_te, Y_te = np.concatenate((X_te, pre_X_te)), np.concatenate((Y_te, pre_Y_te))
        
    if not mlp_net:
        X_test, Y_test  = reshape_features(X_te, Y_te)
    else:
        X_test, Y_test  = X_te, Y_te
    print(f'X_test {X_test.shape} Y_test {Y_test.shape}')
    
    return X_test, Y_test



        
def get_task_training_data(data_dir, task_months):
    
    X_tr, Y_tr = get_month_data(data_dir, task_months[-1])
    
    for month in task_months[:-1]:
        pre_X_tr, pre_Y_tr = get_month_data(data_dir, month)
        X_tr, Y_tr = np.concatenate((X_tr, pre_X_tr)), np.concatenate((Y_tr, pre_Y_tr))
        
    X_train, Y_train  = reshape_features(X_tr, Y_tr)
    
    print(f'X_train {X_train.shape} Y_train {Y_train.shape}')
    
    return X_train, Y_train

def get_current_task_joint_training_data(data_dir, current_task):
    X_tr, Y_tr = get_month_data(data_dir, current_task)
    
    X_train, Y_train  = reshape_features(X_tr, Y_tr)
    
    print(f'X_train {X_train.shape} Y_train {Y_train.shape}')
    
    return X_train, Y_train

def reshape_features(X_, Y_):
    
    X = []
    for i in X_:
        tmp = np.zeros(2401)
        tmp[:len(i)] = i
        X.append(tmp)
    X = np.array(X)
    
    X = np.array(X.reshape(X.shape[0],1,49,49), np.float32)
    
    Y = np.array(Y_, np.int32)
    
    return X, Y


def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
        
        
        

        
        
def get_dataloader(X, y, batch_size, train_data=True):
    # manage class imbalance issue
    # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
    # https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/17
    y = np.array(y,dtype=int)
    
    class_sample_count = np.array(
        [len(np.where(y == t)[0]) for t in np.unique(y)])
                    
    weight = 1. / class_sample_count
    #print(weight)
    samples_weight = np.array([weight[t] for t in y])

    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    X_ = torch.from_numpy(X).type(torch.FloatTensor)
    y_ = torch.from_numpy(y).type(torch.FloatTensor)
    
    data_tensored = torch.utils.data.TensorDataset(X_,y_)    
    
    if train_data:
                    trainloader = torch.utils.data.DataLoader(data_tensored, batch_size = batch_size,
                                                              num_workers=1, sampler=sampler, drop_last=True)
                    return trainloader
    else:
                    validloader = torch.utils.data.DataLoader(data_tensored, batch_size = batch_size,
                                                              num_workers=1, sampler=sampler, drop_last=False)
                    return validloader
            
            



                
class LRScheduler():
    """
    Learning rate scheduler. If the validation loss does not decrease for the 
    given number of `patience` epochs, then the learning rate will decrease by
    by given `factor`.
    """
    def __init__(
        self, optimizer, patience=7, min_lr=1e-6, factor=0.25
    ):
        """
        new_lr = old_lr * factor

        :param optimizer: the optimizer we are using
        :param patience: how many epochs to wait before updating the lr
        :param min_lr: least lr value to reduce to while updating
        :param factor: factor by which the lr should be updated
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( 
                self.optimizer,
                mode='min',
                patience=self.patience,
                factor=self.factor,
                min_lr=self.min_lr,
                verbose=True
            )

    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)

