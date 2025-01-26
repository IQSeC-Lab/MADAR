import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time, random
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader

from az_utils import *



def validation(model, validloader, batch_size, criterion, device):
  

    model.eval()
    valid_loss = []
    valid_acc = []
    
    with torch.no_grad():
        #for idx, data in prog_bar:
        for x_batch, y_batch in tqdm(validloader):
            #x_batch, y_batch = data
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            
            loss = criterion(y_pred, y_batch.view(-1,1))
            acc = binary_acc(y_pred, y_batch)
            valid_loss.append(loss.item())
            valid_acc.append(acc.item())
            
    return np.mean(valid_loss), np.mean(valid_acc)


def binary_acc(y_pred_tag, y_):
    correct_results_sum = (torch.round(y_pred_tag).squeeze(1) == y_).sum().float()
    acc = correct_results_sum/y_.shape[0]
    #acc = acc * 100
    
    return acc



class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        
    def __call__(self, path, opt_path ,epoch, val_loss, model, optimizer):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(path, opt_path, epoch, val_loss, model, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(path, opt_path, epoch, val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, path, opt_path, epoch, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.delete_previous_saved_model(path, opt_path)
        path = path + 'best_model_epoch_' + str(epoch) + '.pt'
        print(f'{path}')
        torch.save(model.state_dict(), path)
        
        opt_path = opt_path + 'best_optimizer_epoch_' + str(epoch) + '.pt'
        print(f'{opt_path}')
        torch.save(optimizer.state_dict(), opt_path)
        
        self.val_loss_min = val_loss
        
    def delete_previous_saved_model(self, path, opt_path):
        saved_models = os.listdir(path)
        for prev_model in saved_models:
            prev_model = path + prev_model
            #print(prev_model)
            if os.path.isfile(prev_model):
                os.remove(prev_model)
            else: pass
        
        saved_opt = os.listdir(opt_path)
        for prev_opt in saved_opt:
            prev_opt = opt_path + prev_opt
            
            if os.path.isfile(prev_opt):
                os.remove(prev_opt)
            else: pass


def training_early_stopping(model, save_path, opt_save_path, X_train, y_train, X_valid, y_valid, 
                            patience, batch_size, device, optimizer, num_epoch,             
                            criterion, replay_type, current_task, exp, earlystopping=True):
 
    
    trainloader = get_dataloader(X_train, y_train, batch_size, train_data=True)
    
    #try:
    #    validloader = get_dataloader(X_valid, y_valid, batch_size, train_data=False)
    #except:
    X_valid = torch.from_numpy(X_valid).type(torch.FloatTensor)
    y_valid = torch.from_numpy(y_valid).type(torch.FloatTensor)
    valid = torch.utils.data.TensorDataset(X_valid, y_valid)    
    validloader = torch.utils.data.DataLoader(valid, batch_size = batch_size, shuffle = False)    

    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    
    early_stopping = EarlyStopping(patience=patience, verbose=True)
    
    
    start_train = time.time()
    for epoch in range(1,num_epoch+1):
        print(f"Epoch {epoch} of {num_epoch}")
        epoch_train_loss, epoch_train_acc = epoch_training(model, trainloader, batch_size, criterion, optimizer, device)
        epoch_valid_loss, epoch_valid_acc = validation(model, validloader, batch_size, criterion, device)
        
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        valid_loss.append(epoch_valid_loss)
        valid_acc.append(epoch_valid_acc)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.4f}")
        print(f'Val Loss: {epoch_valid_loss:.4f}, Val Acc: {epoch_valid_acc:.4f}')

        
        #lr_scheduler(epoch_valid_loss)
        if earlystopping:
            early_stopping(save_path, opt_save_path, epoch, epoch_valid_loss, model, optimizer)

            if early_stopping.early_stop:
                print("Early stopping")
                break
                
    end = time.time()
    print(f"Training time: {(end-start_train)/60:.3f} minutes")
    
    #best_model = os.listdir(save_path)
    #best_epoch = int(best_model[0].split('_')[3].split('.')[0])
    return (end-start_train)/60, epoch, train_loss, valid_loss 








def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Ember_MLP_Net(nn.Module):
    def __init__(self, input_features):
        super(Ember_MLP_Net, self).__init__()
        
        self.fc1 = nn.Linear(input_features, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.act1 = nn.ReLU()
        self.fc1_drop = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.act2 = nn.ReLU()
        self.fc2_drop = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.act3 = nn.ReLU()
        self.fc3_drop = nn.Dropout(p=0.5)        
        
        self.fc4 = nn.Linear(256, 128)
        self.fc4_bn = nn.BatchNorm1d(128)
        self.act4 = nn.ReLU()
        self.fc4_drop = nn.Dropout(p=0.5)  
        
        self.fc_last = nn.Linear(128, 1) 
        self.out = nn.Sigmoid()
        
        #self.activate = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.act1(x) 
        x = self.fc1_drop(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.act2(x) 
        x = self.fc2_drop(x)
        
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = self.act3(x) 
        x = self.fc3_drop(x)
        
        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = self.act4(x)
        x = self.fc4_drop(x)
        
        x = self.fc_last(x)
        x = self.out(x)
        return x



class Ember_Net(nn.Module):
    def __init__(self):
        super(Ember_Net, self).__init__()
        
        input_channel = 64
        self.conv1 = nn.Conv2d(1, input_channel*2, kernel_size=2)
        
        self.conv1_drop = nn.Dropout2d(p=0.1, inplace=True)
        
        self.conv2 = nn.Conv2d(input_channel*2, input_channel*4, kernel_size=4, padding=4)
        self.conv2_drop = nn.Dropout2d(p=0.1, inplace=True)
        
        self.conv3 = nn.Conv2d(input_channel*4, input_channel*8, kernel_size=4, padding=4)
        self.conv3_drop = nn.Dropout2d(p=0.1, inplace=True)
        
        self.conv4 = nn.Conv2d(input_channel*8, input_channel*16, kernel_size=4, padding=4)
        self.conv4_drop = nn.Dropout2d(p=0.1, inplace=True)
        
        self.fc1 = nn.Linear(input_channel*16*7*7, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        
        self.fc2 = nn.Linear(256, 128)
        self.fc2_bn = nn.BatchNorm1d(128)
        
        self.fc3 = nn.Linear(128, 1)
        self.out = nn.Sigmoid()

    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv1_drop(x)
        x = F.max_pool2d(x, kernel_size=2)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv3_drop(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.conv4_drop(x)
        x = F.max_pool2d(x, kernel_size=2)
        
        #print(x.shape)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        
        x = self.fc1(x)
        #print(x.shape)
        x = self.fc1_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.7)
        
        
        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training, p=0.7)
        
        
        x = self.fc3(x)
        x = self.out(x)
        return x
    



def training(model, X_train, y_train, batch_size, device, optimizer, num_epoch,\
             criterion, replay_type, current_task, save_dir, exp):
    

    
    # manage class imbalance issue
    # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
    # https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/17
    class_sample_count = np.array(
        [len(np.where(y_train == t)[0]) for t in np.unique(y_train)])
    weight = 1. / class_sample_count
    #print(weight)
    samples_weight = np.array([weight[t] for t in y_train])

    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.FloatTensor)
    
    train = torch.utils.data.TensorDataset(X_train,y_train)    
    trainloader = torch.utils.data.DataLoader(train, batch_size = batch_size, num_workers=5, sampler=sampler, drop_last=True)    
    
    model.train()
    for epoch in range(1,num_epoch+1):
        with tqdm(trainloader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            for inputs, labels in tepoch:
                tepoch.update(1)
                inputs, labels = inputs.to(device), labels.to(device) 

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = model(inputs).to(device)
                loss = criterion(outputs, labels.view(-1,1))

                #print(labels.shape, outputs.unsqueeze(1))
                #print(torch.round(outputs).squeeze(1).shape,labels.shape)

                acc = binary_acc(outputs, labels)

                loss.backward()
                optimizer.step()
                tepoch.set_postfix({'Training Loss': loss.item(), 'Accuracy': acc.item()})
                #tepoch.set_postfix({'Train loss': loss.item()})
                time.sleep(0.0001)
        #print(loss)
    PATH = save_dir + 'training_runs/' + str(current_task) + '/' + str(replay_type) + '_' + str(exp) +'_model.pt'
    create_parent_folder(PATH)
    print(f'model save path {PATH}')
    torch.save(model.state_dict(), PATH)
    

def testing(model, X_test, Y_test, batch_size, device):
    X_te = torch.from_numpy(X_test).type(torch.FloatTensor)
    y_te = torch.from_numpy(Y_test).type(torch.FloatTensor)
    test = torch.utils.data.TensorDataset(X_te, y_te)   
    testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)    

    model.eval()
    y_pred_list = []
    y_true_list = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(testloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)

            y_pred_tag = torch.round(y_test_pred).squeeze(1).float()
            y_pred_list += list(y_pred_tag.cpu().numpy())
            y_true_list += list(y_batch.cpu().numpy())

    correct_test_results = (np.array(y_pred_list) == np.array(y_true_list)).sum()
    #acc = correct_test_results/len(y_true_list)
    #acc = acc * 100
    cls_report = classification_report(np.array(y_true_list), np.array(y_pred_list))
    print(cls_report)
    return cls_report




def testing_aucscore(model, X_test, Y_test, batch_size, device):
    #X_te = torch.from_numpy(X_test).type(torch.FloatTensor)
    #y_te = torch.from_numpy(Y_test).type(torch.FloatTensor) 
    
    testloader = get_dataloader(X_test, Y_test, batch_size, train_data=False)   
    
    model.eval()
    y_pred_list = []
    y_true_list = []
    test_acc = []
    with torch.no_grad():
        for x_batch, y_batch in tqdm(testloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            tmp_test_acc = binary_acc(y_test_pred, y_batch)
            test_acc.append(tmp_test_acc.item())
            
            y_pred_tag = torch.round(y_test_pred).squeeze(1)
            y_pred_list += list(y_pred_tag.cpu().numpy())
            y_true_list += list(y_batch.cpu().numpy())
        
            
    correct_test_results = (np.array(y_pred_list) == np.array(y_true_list)).sum()
    acc = correct_test_results/len(y_true_list)
    
    from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
    
    correct_labels, predicted_labels = np.array(y_true_list), np.array(y_pred_list)
    
    roc_auc = roc_auc_score(correct_labels, predicted_labels)
    precision = precision_score(correct_labels, predicted_labels, average='micro')
    recall = recall_score(correct_labels, predicted_labels, average='micro')
    f1score = f1_score(correct_labels, predicted_labels, average='macro')
    
    return np.mean(test_acc), roc_auc, precision, recall, f1score
    # return np.mean(test_acc)


def epoch_training(model, trainloader, batch_size, criterion, optimizer, device):
    
    running_loss = []
    running_acc = []
    
    model.train()
    #prog_bar = tqdm(enumerate(trainloader), total=int(len(trainloader)/trainloader.batch_size))
    
    #for idx, data in trainloader:
    for x_batch, y_batch in tqdm(trainloader):
        #inputs, labels = data
        x_batch, y_batch = x_batch.to(device), y_batch.to(device) 
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(x_batch).to(device)
        loss = criterion(outputs, y_batch.view(-1,1))

        acc = binary_acc(outputs, y_batch)

        running_loss.append(loss.item())
        running_acc.append(acc.item())

        loss.backward()
        optimizer.step()
    return np.mean(running_loss), np.mean(running_acc)