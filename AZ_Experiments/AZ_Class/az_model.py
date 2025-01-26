import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time, random
from tqdm import tqdm
from sklearn.metrics import classification_report, roc_auc_score, f1_score

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AZ_MLP_Net(nn.Module):
    def __init__(self, input_features, n_classes):
        super(AZ_MLP_Net, self).__init__()
        self.input_feats_length = input_features
        self.output_classes = n_classes
        
        
        self.fc01 = nn.Linear(self.input_feats_length, 2048)
        self.fc01_bn = nn.BatchNorm1d(2048)
        self.fc01_drop = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc1_drop = nn.Dropout(p=0.5)
        
        self.fc2 = nn.Linear(1024, 512)
        self.fc2_bn = nn.BatchNorm1d(512)
        self.fc2_drop = nn.Dropout(p=0.5)
        
        self.fc3 = nn.Linear(512, 256)
        self.fc3_bn = nn.BatchNorm1d(256)
        self.fc3_drop = nn.Dropout(p=0.5)        
        
        self.fc4 = nn.Linear(256, 128)
        self.fc4_bn = nn.BatchNorm1d(128)
        self.fc4_drop = nn.Dropout(p=0.5)  
        
        self.fc_last = nn.Linear(128, self.output_classes) 
        
        self.activate = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        #print(x.shape)
        
        x = self.fc01(x)
        x = self.fc01_bn(x)
        x = self.activate(x) 
        x = self.fc01_drop(x)
        
        
        x = self.fc1(x)
        x = self.fc1_bn(x)
        x = self.activate(x) 
        x = self.fc1_drop(x)

        x = self.fc2(x)
        x = self.fc2_bn(x)
        x = self.activate(x) 
        x = self.fc2_drop(x)
        
        x = self.fc3(x)
        x = self.fc3_bn(x)
        x = self.activate(x) 
        x = self.fc3_drop(x)
        
        x = self.fc4(x)
        x = self.fc4_bn(x)
        x = self.activate(x) 
        x = self.fc4_drop(x)
        
        x = self.fc_last(x)
        return x 



def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))

def get_dataloader(X, y, batch_size, train_data=True):
    # manage class imbalance issue
    # https://discuss.pytorch.org/t/how-to-handle-imbalanced-classes/11264/2
    # https://discuss.pytorch.org/t/some-problems-with-weightedrandomsampler/23242/17
    class_sample_count = np.array(
        [len(np.where(y == t)[0]) for t in np.unique(y)])
                    
    weight = 1. / class_sample_count
    #print(weight)
    samples_weight = np.array([weight[t] for t in y])

    samples_weight = torch.from_numpy(samples_weight).float()
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight), replacement=True)

    X_ = torch.from_numpy(X).type(torch.FloatTensor)
    y_ = torch.from_numpy(y).type(torch.long) #.type(torch.FloatTensor)
    #print(type(y_))
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


        
def testing_aucscore(model, X_test, Y_test, batch_size, device):
    X_te = torch.from_numpy(X_test).type(torch.FloatTensor)
    y_te = torch.from_numpy(Y_test).type(torch.FloatTensor)
    test = torch.utils.data.TensorDataset(X_te, y_te)    
    testloader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)    

    model.eval()
    y_pred_list = []
    y_true_list = []
    y_predicts_scores = []
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm(testloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            
            #print(y_test_pred)
            for i in range(len(y_test_pred)):
                y_predicts_scores.append(y_test_pred[i])
            
            _, y_pred_ = torch.max(y_test_pred, dim = 1) 
            y_pred_list += list(y_pred_.cpu().numpy())
            y_true_list += list(y_batch.cpu().numpy())

    correct_test_results = (np.array(np.round(y_pred_list)) == np.array(y_true_list)).sum()
    acc = correct_test_results/len(y_true_list)
    #acc = acc * 100
    cls_report = classification_report(np.array(y_true_list), np.array(y_pred_list))
    
    rocauc_score = roc_auc_score(np.array(y_true_list), np.array(y_predicts_scores), multi_class="ovo", average="macro")
    f1score = f1_score(np.array(y_true_list), np.array(y_pred_list), average='weighted')
    
    print(f'ROC AUC SCORE {rocauc_score}')
    print(acc, f1score)
    #print(cls_report)
    return acc, f1score



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
        #print(outputs.shape, y_batch.shape)
        #print(y_batch, outputs)
        loss = criterion(outputs, y_batch)

        acc = multiclass_acc(outputs, y_batch)

        running_loss.append(loss.item() * x_batch.size(0))
        running_acc.append(acc.item())

        loss.backward()
        optimizer.step()
        
    return np.mean(running_loss), np.mean(running_acc)

def validation(model, validloader, batch_size, criterion, device):
  

    model.eval()
    valid_loss = []
    valid_acc = []
    
    #prog_bar = tqdm(enumerate(validloader), total=int(len(validloader)/validloader.batch_size))
    
    with torch.no_grad():
        #for idx, data in prog_bar:
        for x_batch, y_batch in tqdm(validloader):
            #x_batch, y_batch = data
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred = model(x_batch)
            
            loss = criterion(y_pred, y_batch)
            acc = multiclass_acc(y_pred, y_batch)
            valid_loss.append(loss.item() * x_batch.size(0))
            valid_acc.append(acc.item())
            
    return np.mean(valid_loss), np.mean(valid_acc)


            
def training_early_stopping(model, save_path, X_train, y_train, X_valid, y_valid, 
                            patience, batch_size, device, optimizer, num_epoch,             
                            criterion, replay_type, current_task, exp, earlystopping=True):
 
    
    trainloader = get_dataloader(X_train, y_train, batch_size, train_data=True)
    validloader = get_dataloader(X_valid, y_valid, batch_size, train_data=False)
    
    
    train_loss, train_acc = [], []
    valid_loss, valid_acc = [], []
    
    early_stopping = EarlyStopping(model, patience=patience, verbose=True)
    
    
    start_train = time.time()
    for epoch in range(1,num_epoch+1):
        print(f"Epoch {epoch} of {num_epoch}")
        epoch_train_loss, epoch_train_acc = epoch_training(model, trainloader, batch_size, criterion, optimizer, device)
        epoch_valid_loss, epoch_valid_acc = validation(model, validloader, batch_size, criterion, device)
        
        train_loss.append(epoch_train_loss)
        train_acc.append(epoch_train_acc)
        valid_loss.append(epoch_valid_loss)
        valid_acc.append(epoch_valid_acc)
        
        print(f"Train Loss: {epoch_train_loss:.4f}, Train Acc: {epoch_train_acc:.2f}")
        print(f'Val Loss: {epoch_valid_loss:.4f}, Val Acc: {epoch_valid_acc:.2f}')

        
        #lr_scheduler(epoch_valid_loss)
        if earlystopping:
            early_stopping(save_path, epoch, epoch_valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break
                
    end = time.time()
    print(f"Training time: {(end-start_train)/60:.3f} minutes")
    
    #best_model = os.listdir(save_path)
    #best_epoch = int(best_model[0].split('_')[3].split('.')[0])
    return (end-start_train)/60, epoch, train_loss, valid_loss 



def multiclass_acc(y_pred, y_test):
    
    _, y_pred_ = torch.max(y_pred, dim = 1)    
    
    #correct = np.squeeze(y_pred_.eq(y_test.data.view_as(y_pred_)))

    correct_pred = (y_pred_ == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    acc = torch.round(acc * 100)
    
    return acc


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, model, patience=7, verbose=False, delta=0, trace_func=print):
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
    def __call__(self, path, epoch, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(path, epoch, val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(path, epoch, val_loss, model)
            self.counter = 0

    def save_checkpoint(self, path, epoch, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.delete_previous_saved_model(path)
        path = path + 'best_model_epoch_' + str(epoch) + '.pt'
        print(f'{path}')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss
        
    def delete_previous_saved_model(self, path):
        saved_models = os.listdir(path)
        for prev_model in saved_models:
            prev_model = path + prev_model
            #print(prev_model)
            if os.path.isfile(prev_model):
                os.remove(prev_model)
            else: pass