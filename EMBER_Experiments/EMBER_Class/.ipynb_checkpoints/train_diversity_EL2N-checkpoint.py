import torch
from torch import optim
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
import os
import copy

from data import malwareSubDatasetExemplars as SubDataset
from data import ExemplarDataset
from continual_learner import ContinualLearner
import evaluate
import seaborn as sns
from sklearn import metrics, decomposition, manifold
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import random
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, Dataset
from sklearn.ensemble import IsolationForest

from EL2N import *
import utils as utils


def get_dataloader_weights(X, y, batch_size):
    
    X_ = torch.from_numpy(np.array(X)).type(torch.FloatTensor)
    y_ = torch.from_numpy(y).type(torch.FloatTensor)
    
    data_tensored = torch.utils.data.TensorDataset(X_,y_)    
    
    data_loader = torch.utils.data.DataLoader(data_tensored, batch_size = batch_size,
                                              drop_last=False)
    return data_loader



def get_weights(model, layer, X_, Y_, batch_size, device):
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    if layer == 'fc2':
        model.fcE.fc2.register_forward_hook(get_activation(layer))
    elif layer == 'fc3':
        model.fcE.fc3.register_forward_hook(get_activation(layer))
    elif layer == 'fc4':
        model.fcE.fc4.register_forward_hook(get_activation(layer))
    elif layer == 'fc2_bn':
        model.fcE.fc2_bn.register_forward_hook(get_activation(layer))
    elif layer == 'fc3_bn':
        model.fcE.fc3_bn.register_forward_hook(get_activation(layer))
    elif layer == 'fc4_bn':
        model.fcE.fc4_bn.register_forward_hook(get_activation(layer))
    elif layer == 'act2':
        model.fcE.act2.register_forward_hook(get_activation(layer))
    elif layer == 'act3':
        model.fcE.act3.register_forward_hook(get_activation(layer))
    elif layer == 'act4':
        model.fcE.act4.register_forward_hook(get_activation(layer)) 
    
    dataloader = get_dataloader_weights(X_, Y_, batch_size)   
    
    model.eval()
    
    features = []
    
    with torch.no_grad():
        for x_batch, y_batch in tqdm.tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            feats_batch = activation[layer].cpu().numpy()
            
            for f in feats_batch:
                features.append(f)
 
            
    assert len(features) == len(X_)      
    return np.array(features)



def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def custom_validate(model, dataset, batch_size=256, test_size=1024, verbose=True, allowed_classes=None,
             with_exemplars=False, no_task_mask=False, task=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Set model to eval()-mode
    model.eval()

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda())
    total_tested = total_correct = 0
    
    correct_labels = []
    predicted_labels = []
    y_predicts_scores = []
    normalized_scores = []
    
    all_scores = []
    all_labels = []
    
    for data, labels in data_loader:
        # -break on [test_size] (if "None", full dataset is used)
        if test_size:
            if total_tested >= test_size:
                break
        # -evaluate model (if requested, only on [allowed_classes])
        data, labels = data.to(model._device()), labels.to(model._device())
        labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
        #print(labels)
        with torch.no_grad():
            scores = model(data) if (allowed_classes is None) else model(data)[:, allowed_classes]
            
            for sc in scores:
                all_scores.append(sc.cpu().numpy())
            #print(scores)
            #_, predicted = torch.max(scores, 1)

            #y_predicts_scores += list(predicted.detach().cpu().numpy())
                      
                
        # -update statistics
        #total_correct += (predicted == labels).sum().item()
        #total_tested += len(data)
        
        #correct_labels += list(labels.cpu().numpy())
        #predicted_labels += list(predicted.cpu().numpy())
        all_labels += list(labels.cpu().numpy())
    #precision = total_correct / total_tested
    #correct_labels = np.array(correct_labels)

    #if verbose:
    #     print('=>Precision {:.3f}'.format(precision))
    return all_scores, all_labels

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x




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


def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))
        
        
def get_AWS_samples(family_name, family_data, family_weights,\
                        contamination,\
                        num_samples_per_malware_family):
    
    data_X = np.array(family_data)
    
    if len(data_X) > num_samples_per_malware_family:
        
        # fit the model
        family_weights = np.array(family_weights)
        clf = IsolationForest(max_samples=len(family_weights), contamination=contamination)
        clf.fit(family_weights)
        #scores_prediction = clf.decision_function(data_X)
        y_pred = clf.predict(family_weights)


        anomalous_idx = np.where(y_pred == -1.0)
        similar_idx = np.where(y_pred == 1.0)

        #print(f'{family_name}: all-{len(y_pred)} anomalous-{len(anomalous_idx[0])} similar-{len(similar_idx[0])}')
        assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)
        
        anomalous_samples = data_X[anomalous_idx]

        if len(anomalous_samples) > num_samples_per_malware_family:
            anomalous_samples_pool = list(anomalous_samples)
            remaining_samples_to_pick = int(num_samples_per_malware_family/2)
            anomalous_samples = random.sample(anomalous_samples_pool, remaining_samples_to_pick)

        else:
            remaining_samples_to_pick = num_samples_per_malware_family - len(anomalous_samples)

        if remaining_samples_to_pick == 0:
            remaining_samples_to_pick = num_samples_per_malware_family
            
        if remaining_samples_to_pick > len(similar_idx[0]):
            similar_samples = data_X[similar_idx]
        else:
            similar_samples_pool = list(data_X[similar_idx])
            similar_samples = random.sample(similar_samples_pool, remaining_samples_to_pick)
            
        #print(f'anomalous_samples {len(anomalous_samples)} {np.array(anomalous_samples).shape} similar_samples {len(similar_samples)} {np.array(similar_samples).shape}')
        
        replay_samples = np.concatenate((anomalous_samples, similar_samples))
    else:
        replay_samples = data_X
        
    #print(f'Num replay samples {len(replay_samples)}')
    return replay_samples


def IFS_Samples(model, v, Y, v_choose, get_anomalous=True, contamination=0.1, madarEL2N=False):
    data_X = v
    clf = IsolationForest(max_samples=len(data_X), contamination=contamination)
    clf.fit(data_X)
    y_pred = clf.predict(data_X)
    anomalous_idx = np.where(y_pred == -1.0)
    similar_idx = np.where(y_pred == 1.0)

    assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)
    
    if get_anomalous:
        anomalous_samples_pool = list(data_X[anomalous_idx])
        similar_samples_pool = list(data_X[similar_idx])

        v_choose_split = int(np.ceil(v_choose/2))

        if len(anomalous_samples_pool) > v_choose_split:
            if madarEL2N:
                el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
                    model, anomalous_samples_pool, np.ones(len(anomalous_samples_pool)) * Y, v_choose_split, num_classes = 100, batch_size=256)
                anomalous_samples = el2n_subset
            else:
                anomalous_samples = random.sample(anomalous_samples_pool, v_choose_split)

        else:
            anomalous_samples = anomalous_samples_pool

        if len(anomalous_samples) == v_choose_split:
            #print(similar_samples_pool.shape)
            if madarEL2N:
                #print(f'number of samples in the pool {len(similar_samples_pool)}')
                #print(similar_samples_pool)
                el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
                    model, np.array(similar_samples_pool), np.ones(len(similar_samples_pool)) * Y,\
                    v_choose_split, num_classes = 100, batch_size=256)
                similar_samples = el2n_subset
            else:
                similar_samples = random.sample(similar_samples_pool, v_choose_split)
            
        elif len(anomalous_samples) < v_choose_split:
            v_choose_split += v_choose_split - len(anomalous_samples)
            if len(similar_samples_pool) > v_choose_split:
                #similar_samples = random.sample(similar_samples_pool, v_choose_split)
                
                if madarEL2N:
                    el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
                    model, similar_samples_pool, np.ones(len(similar_samples_pool)) * Y, v_choose_split, num_classes = 100, batch_size=256)
                    similar_samples = el2n_subset
                else: 
                    similar_samples = random.sample(similar_samples_pool, v_choose_split)
            else:
                similar_samples = similar_samples_pool
                
        if len(anomalous_samples) > 0 and len(similar_samples) > 0: 
            anomalous_samples, similar_samples = np.array(anomalous_samples), np.array(similar_samples)
            #print(f'anomalous_samples {anomalous_samples.shape} similar_samples {similar_samples.shape}')
            replay_samples = np.concatenate((anomalous_samples, similar_samples))
        else:
            if len(anomalous_samples) <= 0:
                replay_samples = similar_samples
            if len(similar_samples) <= 0:
                replay_samples = anomalous_samples
    else:
        similar_samples_pool = list(data_X[similar_idx])
        if len(similar_samples_pool) > v_choose:
            if madarEL2N:
                    el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
                    model, similar_samples_pool, np.ones(len(similar_samples_pool)) * Y, v_choose, num_classes = 100, batch_size=256)
                    similar_samples = el2n_subset
            else:
                similar_samples = random.sample(similar_samples_pool, v_choose)
        else:
            similar_samples = similar_samples_pool
            
        replay_samples = np.array(similar_samples)
        
    return replay_samples

def AWS_Samples(v, v_weights, v_choose, get_anomalous=True, contamination=0.1):
    
    v_weights = np.array(v_weights) 
    data_X = v
    
    clf = IsolationForest(max_samples=len(v_weights), contamination=contamination)
    clf.fit(v_weights)
    y_pred = clf.predict(v_weights)
    anomalous_idx = np.where(y_pred == -1.0)
    similar_idx = np.where(y_pred == 1.0)

    assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)
    
    if get_anomalous:
        anomalous_samples_pool = list(data_X[anomalous_idx])
        similar_samples_pool = list(data_X[similar_idx])

        v_choose_split = int(np.ceil(v_choose/2))

        if len(anomalous_samples_pool) > v_choose_split:
            anomalous_samples = random.sample(anomalous_samples_pool, v_choose_split)

        else:
            anomalous_samples = anomalous_samples_pool

        if len(anomalous_samples) == v_choose_split:
            similar_samples = random.sample(similar_samples_pool, v_choose_split)
        elif len(anomalous_samples) < v_choose_split:
            v_choose_split += v_choose_split - len(anomalous_samples)
            if len(similar_samples_pool) > v_choose_split:
                similar_samples = random.sample(similar_samples_pool, v_choose_split)
            else:
                similar_samples = similar_samples_pool
        if len(anomalous_samples) > 0 and len(similar_samples) > 0: 
            anomalous_samples, similar_samples = np.array(anomalous_samples), np.array(similar_samples)
            #print(f'anomalous_samples {anomalous_samples.shape} similar_samples {similar_samples.shape}')
            replay_samples = np.concatenate((anomalous_samples, similar_samples))
        else:
            if len(anomalous_samples) <= 0:
                replay_samples = similar_samples
            if len(similar_samples) <= 0:
                replay_samples = anomalous_samples
    else:
        similar_samples_pool = list(data_X[similar_idx])
        if len(similar_samples_pool) > v_choose:
            similar_samples = random.sample(similar_samples_pool, v_choose)
        else:
            similar_samples = similar_samples_pool
            
        replay_samples = np.array(similar_samples)
        
    return replay_samples


def AWS(prevX, weights_X, prevY, memory_budget, min_samples=1, fs_option='ratio'):
    uniqueY = np.unique(prevY)
    
    print(f'replay #of families {len(uniqueY)}')
    countDict = {}
    numSample = 0
    for tmpY in uniqueY:
        tmp_YSample = len(np.where(tmpY == prevY)[0])
        countDict[tmpY] = tmp_YSample
            
        numSample += tmp_YSample
    
    #print(f'numSample {numSample} \n sum(countDict.values()) {sum(countDict.values())}')
    assert numSample == sum(countDict.values())
    
    malware_budget = memory_budget
    
    X_replay, Y_replay = [], []
    
    for indx, Y in enumerate(uniqueY):
        Y_ind = np.where(Y == prevY)
        X_weights = weights_X[Y_ind]
        X_ = prevX[Y_ind]
        
        if fs_option == 'ratio':
            v_choose = int(np.ceil((countDict[Y] / numSample) * memory_budget))

        if fs_option == 'uniform':
            print(f'IFS option {fs_option}')
            v_choose = int(np.ceil(memory_budget / len(uniqueY)))

        if fs_option == 'mix':
            v_choose = int(np.ceil((countDict[Y] / numSample) * memory_budget))
            if v_choose < min_samples:
                print(f'v_choose {v_choose} min_samples {min_samples}')
                v_choose = min_samples
            else:
                print(f'v_choose {v_choose} **')
        
        if len(X_) <= v_choose:
            print(f'len(X_) <= v_choose  {len(X_)}  <= {v_choose}')
            for sample in X_:
                X_replay.append(sample)
                Y_replay.append(Y)
        else:
            print(f'len(X_) {len(X_)} > v_choose {v_choose}')
            X_ = AWS_Samples(X_, X_weights, v_choose, get_anomalous=True, contamination=0.1)
            print(f'len(X_) {len(X_)}  v_choose {v_choose}')
            for sample in X_:
                X_replay.append(sample)
                Y_replay.append(Y)
    
    X_replay, Y_replay = np.array(X_replay), np.array(Y_replay)
    X_replay, Y_replay = shuffle(X_replay, Y_replay)
    
    return X_replay, Y_replay



def IFS(model, prevX, prevY, memory_budget, min_samples=1, fs_option='ratio'):
    uniqueY = np.unique(prevY)
    
    countDict = {}
    numSample = 0
    for tmpY in uniqueY:
        tmp_YSample = len(np.where(tmpY == prevY)[0])
        countDict[tmpY] = tmp_YSample
            
        numSample += tmp_YSample
    
    #print(f'numSample {numSample} \n sum(countDict.values()) {sum(countDict.values())}')
    assert numSample == sum(countDict.values())
    
    malware_budget = memory_budget
    
    X_replay, Y_replay = [], []
    
    for indx, Y in enumerate(uniqueY):
        Y_ind = np.where(Y == prevY)
        X_ = prevX[Y_ind]
        Y_ = prevY[Y_ind]
        
        if fs_option == 'ratio':
            v_choose = int(np.ceil((countDict[Y] / numSample) * memory_budget))

        if fs_option == 'uniform':
            #print(f'IFS option {fs_option} memory {memory_budget} uniqueY {uniqueY}')
            v_choose = int(np.ceil(memory_budget / len(uniqueY)))

        if fs_option == 'mix':
            v_choose = int(np.ceil((countDict[Y] / numSample) * memory_budget))
            if v_choose < min_samples:
                #print(f'v_choose {v_choose} min_samples {min_samples}')
                v_choose = min_samples
            else:
                print(f'v_choose {v_choose} **')
        
        if len(X_) <= v_choose:
            #print(f'len(X_) <= v_choose  {len(X_)}  <= {v_choose}')
            for sample in X_:
                X_replay.append(sample)
                Y_replay.append(Y)
        else:
            #print(f'len(X_) {len(X_)} > v_choose {v_choose}')
            X_ = IFS_Samples(model, X_, Y, v_choose, get_anomalous=True, contamination=0.1, madarEL2N=True)
            
            # print(f'X_ {X_.shape} Y_{Y_.shape}')
            # el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
            #       model, X_, Y_, v_choose, num_classes = 100, batch_size=256)
            # X_ = el2n_subset
            # print(f'X_ {len(X_)}')
            
            for sample in X_:
                X_replay.append(sample)
                Y_replay.append(Y)
    
    X_replay, Y_replay = np.array(X_replay), np.array(Y_replay)
    X_replay, Y_replay = shuffle(X_replay, Y_replay)
    
    return X_replay, Y_replay

def getGRSSamples(X, Y, memory_budget):
    indx = [i for i in range(len(Y))]
    random.shuffle(indx)

    replay_index = indx[:memory_budget]

    X_train = X[replay_index]
    Y_train = Y[replay_index]
    
    print(f'np.unique(Y) {np.unique(Y)} np.unique(Y_train) {np.unique(Y_train)}')
    #assert np.unique(Y_train).all() == np.unique(Y).all()
    
    return X_train, Y_train


def getReplaySamples(model, PreviousTasksData,
                     PreviousTasksLabels, args):
    #replay_portion = 0.5
    memory_budget = args.memory_budget

    X, Y = PreviousTasksData
    
    if args.replay_config == 'frs':
        print(f'******** FRS *********')
        all_replay_X = []
        all_replay_Y = []
        for previousTask, CurrentTaskLabels in enumerate(PreviousTasksLabels):
            for task_Y in CurrentTaskLabels:
                Y_task_ind = np.where(Y == task_Y)

                task_samples = X[Y_task_ind]
                task_labels = Y[Y_task_ind]
                
                if len(task_samples) > args.num_replay_sample:
                    task_samples = random.sample(list(task_samples), args.num_replay_sample)
                     
                
                for ind, frs_sample in enumerate(task_samples):
                    all_replay_X.append(frs_sample)
                    all_replay_Y.append(task_Y)


        all_replay_X, all_replay_Y = np.array(all_replay_X), np.array(all_replay_Y)
        #unique_labels = np.unique(all_replay_Y)
        
        return all_replay_X, all_replay_Y
    
    elif args.replay_config == 'ifs':
        print(f'i am IFS')
        all_replay_X = []
        all_replay_Y = []
        for previousTask, CurrentTaskLabels in enumerate(PreviousTasksLabels):
            for task_Y in CurrentTaskLabels:
                Y_task_ind = np.where(Y == task_Y)

                task_samples = X[Y_task_ind]
                task_labels = Y[Y_task_ind]
                
                for ind, l in enumerate(task_labels):
                    all_replay_X.append(task_samples[ind])
                    all_replay_Y.append(l)

        all_replay_X, all_replay_Y = np.array(all_replay_X), np.array(all_replay_Y)
        
        all_replay_X, all_replay_Y = IFS(model, all_replay_X, all_replay_Y,\
                                         memory_budget, min_samples = args.min_samples,\
                                         fs_option=args.ifs_option)
        
        #unique_labels = np.unique(all_replay_Y)
        
        return all_replay_X, all_replay_Y
    
    elif args.replay_config == 'aws':
        print(f'i am AWS')
        all_replay_X = []
        #all_weights_X = []
        all_replay_Y = []
        for previousTask, CurrentTaskLabels in enumerate(PreviousTasksLabels):
            for task_Y in CurrentTaskLabels:
                Y_task_ind = np.where(Y == task_Y)

                task_samples = X[Y_task_ind]
                task_labels = Y[Y_task_ind]
                
#                 cuda = torch.cuda.is_available()
#                 device = torch.device("cuda" if cuda else "cpu")
#                 k_weights = get_weights(model, args.layer, task_samples, task_labels, args.batch, device)

                for ind, ifs_sample in enumerate(task_samples):
                    all_replay_X.append(ifs_sample)
#                     all_weights_X.append(k_weights[ind])
                    all_replay_Y.append(task_Y)

        all_replay_X, all_replay_Y = np.array(all_replay_X), np.array(all_replay_Y)            
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        all_weights_X = get_weights(model, args.layer, all_replay_X, all_replay_Y, args.batch, device)
        
        
        assert len(all_weights_X) == len(all_replay_X)
        all_replay_X, all_weights_X, all_replay_Y = np.array(all_replay_X), np.array(all_weights_X), np.array(all_replay_Y)
        
        all_replay_X, all_replay_Y = AWS(all_replay_X, all_weights_X, all_replay_Y,\
                                         memory_budget, min_samples = args.min_samples,\
                                         fs_option=args.ifs_option)
        all_replay_X, all_replay_Y = np.array(all_replay_X), np.array(all_replay_Y)
        #unique_labels = np.unique(all_replay_Y)
        
        return all_replay_X, all_replay_Y      
    else: #grs
        all_replay_X = []
        all_replay_Y = []
        for previousTask, CurrentTaskLabels in enumerate(PreviousTasksLabels):
            for task_Y in CurrentTaskLabels:
                Y_task_ind = np.where(Y == task_Y)

                task_samples = X[Y_task_ind]
                task_labels = Y[Y_task_ind]

                for ind, l in enumerate(task_labels):
                    all_replay_X.append(task_samples[ind])
                    all_replay_Y.append(l)

        all_replay_X, all_replay_Y = np.array(all_replay_X), np.array(all_replay_Y)
        unique_labels = np.unique(all_replay_Y)
        
        
        if args.grs_joint:
            return all_replay_X, all_replay_Y
        else:
            all_replay_X, all_replay_Y = getGRSSamples(all_replay_X, all_replay_Y, memory_budget)
            
            return all_replay_X, all_replay_Y
    
    
def get_rest_task_data(RestTasksData, RestTasksLabels):
    
    X, Y = RestTasksData
    
    all_rest_X = []
    all_rest_Y = []
    for restTask, restCurrentTaskLabels in enumerate(RestTasksLabels):
        for task_Y in restCurrentTaskLabels:
            Y_task_ind = np.where(Y == task_Y)

            task_samples = X[Y_task_ind]
            task_labels = Y[Y_task_ind]

            for ind, l in enumerate(task_labels):
                all_rest_X.append(task_samples[ind])
                all_rest_Y.append(l)


    all_rest_X, all_rest_Y = np.array(all_rest_X), np.array(all_rest_Y)
    #unique_labels = np.unique(all_rest_Y)

    return all_rest_X, all_rest_Y
    
    
def get_current_task_data(CurrentTaskData, CurrentTaskLabels):
    
    X, Y = CurrentTaskData
    
    X_task_samples = []
    Y_task_labels = []
    
    for task_Y in CurrentTaskLabels:
        Y_task_ind = np.where(Y == task_Y)
        
        task_samples = X[Y_task_ind]
        task_labels = Y[Y_task_ind]
        
        for ind, l in enumerate(task_labels):
            X_task_samples.append(task_samples[ind])
            Y_task_labels.append(l)
        
    X_task_samples, Y_task_labels = np.array(X_task_samples),\
                                    np.array(Y_task_labels)
    
    return X_task_samples, Y_task_labels



def get_current_task_rest_data(X, Y, CurrentTaskLabels):
    
    X_task_samples = []
    Y_task_labels = []
    
    for task_Y in CurrentTaskLabels:
        Y_task_ind = np.where(Y == task_Y)
        
        task_samples = X[Y_task_ind]
        task_labels = Y[Y_task_ind]
        
        for ind, l in enumerate(task_labels):
            X_task_samples.append(task_samples[ind])
            Y_task_labels.append(l)
        
    X_task_samples, Y_task_labels = np.array(X_task_samples),\
                                    np.array(Y_task_labels)
    
    return X_task_samples, Y_task_labels  



def get_current_task_test_data(X, Y, CurrentTaskLabels):
    
    X_task_samples = []
    Y_task_labels = []
    
    for task_Y in CurrentTaskLabels:
        Y_task_ind = np.where(Y == task_Y)
        
        task_samples = X[Y_task_ind]
        task_labels = Y[Y_task_ind]
        
        for ind, l in enumerate(task_labels):
            X_task_samples.append(task_samples[ind])
            Y_task_labels.append(l)
        
    X_task_samples, Y_task_labels = np.array(X_task_samples),\
                                    np.array(Y_task_labels)
    
    return X_task_samples, Y_task_labels    




class malwareTrainDataset(Dataset):
    
    def __init__(self, dataset):
        super().__init__()
        self.samples, self.labels = dataset

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        
        sample = self.samples[index]
        target = self.labels[index]
        
        return (sample, target) 
        
def train_cl(model, model_save_path, ember_train, ember_test, train_datasets, test_datasets,\
             args, replay_mode="none",\
             scenario="class",classes_per_task=None,iters=2000,batch_size=32,\
             loss_cbs=list(), eval_cbs=list(),
             use_exemplars=True, add_exemplars=False, metric_cbs=list()):
    '''Train a model (with a "train_a_batch" method) on multiple tasks, with replay-strategy specified by [replay_mode].

    [model]             <nn.Module> main model to optimize across all tasks
    [train_datasets]    <list> with for each task the training <DataSet>
    [replay_mode]       <str>, choice from "generative", "exact", "current", "offline" and "none"
    [scenario]          <str>, choice from "task", "domain" and "class"
    [classes_per_task]  <int>, # of classes per task
    [iters]             <int>, # of optimization-steps (i.e., # of batches) per task
    [generator]         None or <nn.Module>, if a seperate generative model should be trained (for [gen_iters] per task)
    [*_cbs]             <list> of call-back functions to evaluate training-progress'''
    
    init_classes = args.init_classes
    target_classes = args.target_classes
    num_class = target_classes
    
    if scenario == 'class':
        initial_task_num_classes = init_classes
        if initial_task_num_classes > target_classes:
            raise ValueError(f"Initial Number of Classes cannot be more than {target_classes} classes!")

        #print(f'num_tasks {args.tasks}', flush=True)
        left_tasks = args.tasks - 1 
        classes_per_task_except_first_task = int((num_class - initial_task_num_classes) / left_tasks)



        #print(selected_classes)
        first_task = list(range(initial_task_num_classes))

        labels_per_task = [first_task] + [list(initial_task_num_classes +\
                                           np.array(range(classes_per_task_except_first_task)) +\
                                           classes_per_task_except_first_task * task_id)\
                                          for task_id in range(left_tasks)]
        #print(labels_per_task)

        classes_per_task = classes_per_task_except_first_task

    else:
        tasks = args.tasks
        classes_per_task = int(np.floor(num_class / tasks))

        labels_per_task = [list(np.array(range(classes_per_task)) +\
                            classes_per_task * task_id) for task_id in range(tasks)] 
        
        #print(labels_per_task)
    
    #print(model)
    #model_save_path = './ember_saved_models/1_first_class.pt'
    #model.load_state_dict(torch.load(model_save_path))
    
    
    standardization = StandardScaler()
    
    # Set model in training-mode
    model.train()

    # Use cuda?
    cuda = model._is_on_cuda()
    device = model._device()

    # Initiate possible sources for replay (no replay for 1st task)
    Exact = Generative = Current = False
    previous_model = None

    all_task_average_f1scores = []
    all_task_scores = []
    all_task_labels = []
    all_task_accuracies = []
    
    all_replay_X, all_replay_Y = [], []
    
    
    # Loop over all tasks.
    for task, taskLabels in enumerate(labels_per_task, 1):
        print(f'task {task} =>> taskLabels {taskLabels}', flush=True)
        if scenario=="class":
            #print(f'replay_mode=="offline" and scenario=="class"')
            if task != 1:
                    if args.replay_config == 'aws':
                        from aws_encoder import Classifier as AWS_Classifier
                        model_new = AWS_Classifier(image_size=49, image_channels=1, classes=100).to(device)
                        #load the best model for this task
                        print(f'model_new {model_new}')
                        model_new.load_state_dict(torch.load(model_path)) 
                        prev_replay_X, prev_replay_Y = getReplaySamples(model_new,\
                                                     ember_train, labels_per_task[:task-1], args)
                        
                    else:
                        prev_replay_X, prev_replay_Y = getReplaySamples(model, ember_train, labels_per_task[:task-1], args) 

                    all_replay_X, all_replay_Y = prev_replay_X, prev_replay_Y
                    all_current_X, all_current_Y = get_current_task_data(ember_train, labels_per_task[task-1])
                    
                    print(f'all_current_X {all_current_X.shape} all_replay_X {all_replay_X.shape}')
                    all_X = np.concatenate((all_current_X, all_replay_X))
                    all_Y = np.concatenate((all_current_Y, all_replay_Y))
            else:
                all_X, all_Y = get_current_task_data(ember_train, labels_per_task[task-1])
                
            print()
            print(f'\n all_X {all_X.shape} all_Y {all_Y.shape}\n', flush=True)
            #print(f'all_Y {np.unique(all_Y)}', flush=True)
            standard_scaler = standardization.partial_fit(all_X)

            all_X = standard_scaler.transform(all_X)

            x_test, y_test = ember_test
            x_test = standard_scaler.transform(x_test)

            train_dataset = malwareTrainDataset((all_X, all_Y))
            training_dataset = train_dataset
            
            
            num_training_samples = len(all_Y)
            
            test_datasets = []
            for labels in labels_per_task:
                X_test_task, y_test_task = get_current_task_test_data(x_test, y_test, labels)
                #print(f'y_test_task {np.unique(y_test_task)}')
                test_datasets.append(malwareTrainDataset((X_test_task, y_test_task)))
                
        if replay_mode=="offline" and scenario == "task":
            if args.replay_portion == 1.0 and args.replay_config == 'grs':
                x_test, y_test = ember_test
                train_datasets = [None]*tasks
                
                num_training_samples = 0
                
                for ct, labels in enumerate(labels_per_task):
                    current_X, current_Y = get_current_task_data(ember_train, labels)
                    
                    if ct == task-1:
                        standard_scaler = standardization.partial_fit(current_X)
                        current_X = standard_scaler.transform(current_X)
                        train_dataset = malwareTrainDataset((current_X, current_Y))
                        #print(f'current task {task} labels {np.unique(current_Y)}')
                        training_dataset = train_dataset
                        num_training_samples += len(current_Y)
                    else:
                        current_X = standard_scaler.transform(current_X)
                        train_dataset = malwareTrainDataset((current_X, current_Y))
                        num_training_samples += len(current_Y)
                        
                    train_datasets[ct] = train_dataset
                
                x_test = standard_scaler.transform(x_test)
                
                test_datasets = []
                for labels in labels_per_task:
                    X_test_task, y_test_task = get_current_task_test_data(x_test, y_test, labels)
                    #print(f'y_test_task {np.unique(y_test_task)}')
                    test_datasets.append(malwareTrainDataset((X_test_task, y_test_task)))
                    
            else:
                
                if task == 1:
                    current_X, current_Y = get_current_task_data(ember_train, labels_per_task[task-1])

                    #print(f'current task labels {labels_per_task[task-1]}')
                    #print(f'rest task labels {labels_per_task[task:]}')
                    rest_X, rest_Y = get_rest_task_data(ember_train, labels_per_task[task:])
                    all_rest_X = rest_X
                    all_rest_Y = rest_Y

                else:
                    current_X, current_Y = get_current_task_data(ember_train, labels_per_task[task-1])
                    if args.replay_config == 'aws':
                        from aws_encoder import Classifier as AWS_Classifier
                        model_new = AWS_Classifier(image_size=49, image_channels=1, classes=100).to(device)
                        #load the best model for this task
                        #print(f'model_new {model_new}')
                        model_new.load_state_dict(torch.load(model_path)) 
                        
                        prev_replay_X, prev_replay_Y = getReplaySamples(model_new,\
                                                         ember_train, labels_per_task[:task-1], args)
                        
                    else:
                        prev_replay_X, prev_replay_Y = getReplaySamples(model, ember_train, labels_per_task[task-2:task-1], args)
                    
                    print(f'prev_replay_Y {prev_replay_Y.shape} {np.unique(prev_replay_Y)}')

                    if task > 2:
                        all_replay_X, all_replay_Y = np.concatenate((all_replay_X, prev_replay_X)),\
                                                             np.concatenate((all_replay_Y, prev_replay_Y))
                    else:
                        all_replay_X, all_replay_Y = prev_replay_X, prev_replay_Y

                    #print(f'all_replay_Y {len(all_replay_Y)} {np.unique(all_replay_Y)}')
                    

                    if task != tasks:
                        rest_X, rest_Y = get_rest_task_data(ember_train, labels_per_task[task:])
                        #print(f'rest_Y {np.unique(rest_Y)}')
                        #print(f'rest_Y {len(rest_Y)} {np.unique(rest_Y)}')

                        all_rest_X = np.concatenate((all_replay_X, rest_X))
                        all_rest_Y = np.concatenate((all_replay_Y, rest_Y))

                        #print(f'1 all_rest_Y {np.unique(all_rest_Y)}')

                    else:
                        all_rest_X = all_replay_X
                        all_rest_Y = all_replay_Y

                print()
                #print(f'\n current_Y {len(current_Y)} {np.unique(current_Y)}')
                #print(f'\n all_rest_Y {np.unique(np.array(all_rest_Y))}\n')
                standard_scaler = standardization.partial_fit(current_X)

                current_X = standard_scaler.transform(current_X)
                all_rest_X = standard_scaler.transform(all_rest_X)

                x_test, y_test = ember_test
                x_test = standard_scaler.transform(x_test)
                
                train_datasets = [None]*tasks
                for ct, labels in enumerate(labels_per_task):
                    #print(f'ct {ct}  labels {labels}')
                    if ct == task-1:
                        #print(f'ct {ct} current_task_Y {np.unique(current_Y)}')
                        train_dataset = malwareTrainDataset((current_X, current_Y))
                        #print(f'current task data {len(train_dataset)}')
                        training_dataset = train_dataset
                        train_datasets[ct] = train_dataset
                    else:
                        rest_task_X, rest_task_Y = get_current_task_rest_data(all_rest_X, all_rest_Y, labels)
                        #print(f'ct {ct} rest_task_Y {np.unique(rest_task_Y)}')
                        train_datasets[ct] = malwareTrainDataset((rest_task_X, rest_task_Y))

                num_training_samples = len(current_Y) + len(rest_task_Y)
                
                test_datasets = []
                for labels in labels_per_task:
                    X_test_task, y_test_task = get_current_task_test_data(x_test, y_test, labels)
                    test_datasets.append(malwareTrainDataset((X_test_task, y_test_task)))        
        
        
        if replay_mode=="none" and scenario == "task":
            if args.replay_portion == 0.0 and args.replay_config == 'grs':
                #print(f'i am here')
                x_test, y_test = ember_test

                current_X, current_Y = get_current_task_data(ember_train, taskLabels)
                standard_scaler = standardization.partial_fit(current_X)
                current_X = standard_scaler.transform(current_X)
                train_dataset = malwareTrainDataset((current_X, current_Y))
                #print(f'current task {task} labels {np.unique(current_Y)}')
                training_dataset = train_dataset
                
                x_test = standard_scaler.transform(x_test)
                
                test_datasets = []
                for labels in labels_per_task:
                    X_test_task, y_test_task = get_current_task_test_data(x_test, y_test, labels)
                    #print(f'y_test_task {np.unique(y_test_task)}')
                    test_datasets.append(malwareTrainDataset((X_test_task, y_test_task)))
                            
        
        # -but if "offline"+"task"-scenario: all tasks so far included in 'exact replay' & no current batch
        if replay_mode=="offline" and scenario == "task":
            Exact = True
            previous_datasets = train_datasets

        # Add exemplars (if available) to current dataset (if requested)
        if add_exemplars and task>1:
            target_transform = None
            exemplar_dataset = ExemplarDataset(model.exemplar_sets, target_transform=target_transform)
            training_dataset = ConcatDataset([train_dataset, exemplar_dataset])
        else:
            training_dataset = train_dataset
        

        # Find [active_classes]
        if scenario == "task":
            # -for Task-IL scenario, create <list> with for all tasks so far a <list> with the active classes
            active_classes = [list(range(classes_per_task * i, classes_per_task * (i + 1))) for i in range(task)]
        elif scenario == "class":
            # -for Class-IL scenario, create one <list> with active classes of all tasks so far
            init_classes = args.init_classes
            if task == 1:
                active_classes = list(range(init_classes))
            else:
                active_classes = list(range(init_classes + classes_per_task * (task -1)))
               
        #print(f'task {task} - active_classes {active_classes}', flush=True)
        
        
        # Reset state of optimizer(s) for every task (if requested)
        if model.optim_type=="adam_reset":
            model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))

        # Initialize # iters left on current data-loader(s)
        iters_left = iters_left_previous = 1
        if scenario=="task":
            up_to_task = task if replay_mode=="offline" else task-1
            iters_left_previous = [1]*up_to_task
            data_loader_previous = [None]*up_to_task
        
        import time
        task_start_training = time.time()
        
        # Define tqdm progress bar(s)
        progress = tqdm(range(1, iters+1))

        # Loop over all iterations
        #print(f'training_dataset {len(train_dataset)}')
        
        for batch_index in range(1, iters+1):

            # Update # iters left on current data-loader(s) and, if needed, create new one(s)
            iters_left -= 1
            if iters_left==0:
                #print(f'i am here creating data loader')
                # msr --> problem in data loader
                data_loader = iter(utils.get_data_loader(training_dataset, batch_size, cuda=cuda, drop_last=True))
                # NOTE:  [train_dataset]  is training-set of current task
                #      [training_dataset] is training-set of current task with stored exemplars added (if requested)
                iters_left = len(data_loader)

            if Exact:
                if scenario=="task":
                    
                    #print(f'previous_datasets {previous_datasets}')
                    
                    up_to_task = task if replay_mode=="offline" else task-1
                    batch_size_replay = int(np.ceil(batch_size/up_to_task)) if (up_to_task>1) else batch_size
                    # -in Task-IL scenario, need separate replay for each task
                    for task_id in range(up_to_task):
                        batch_size_to_use = min(batch_size_replay, len(previous_datasets[task_id]))
                        iters_left_previous[task_id] -= 1
                        if iters_left_previous[task_id]==0:
                            data_loader_previous[task_id] = iter(utils.get_data_loader(
                                previous_datasets[task_id], batch_size_to_use, cuda=cuda, drop_last=True
                            ))
                            iters_left_previous[task_id] = len(data_loader_previous[task_id])

                else:
                    iters_left_previous -= 1
                    if iters_left_previous==0:
                        batch_size_to_use = min(batch_size, len(ConcatDataset(previous_datasets)))
                        data_loader_previous = iter(utils.get_data_loader(ConcatDataset(previous_datasets),
                                                                          batch_size_to_use, cuda=cuda, drop_last=True))
                        iters_left_previous = len(data_loader_previous)


            # -----------------Collect data------------------#

            #####-----CURRENT BATCH-----#####
            if replay_mode=="offline" and scenario=="task":
                x = y = scores = None
            else:
                #print(f'data_loader {data_loader}')
                x, y = next(data_loader)                                    #--> sample training data of current task
                #x = x.double()
                #print(f'i am in in line 447')
                y = y-classes_per_task*(task-1) if scenario=="task" else y  #--> ITL: adjust y-targets to 'active range'
                x, y = x.to(device), y.to(device)                           #--> transfer them to correct device
                # If --bce, --bce-distill & scenario=="class", calculate scores of current batch with previous model
                binary_distillation = hasattr(model, "binaryCE") and model.binaryCE and model.binaryCE_distill
                if binary_distillation and scenario=="class" and (previous_model is not None):
                    with torch.no_grad():
                        scores = previous_model(x)[:, :(classes_per_task * (task - 1))]
                else:
                    scores = None


            #####-----REPLAYED BATCH-----#####
            if not Exact and not Generative and not Current:
                x_ = y_ = scores_ = None   #-> if no replay

            ##-->> Exact Replay <<--##
            if Exact:
                #print(f'i am here EXACT')
                scores_ = None
                if scenario in ("domain", "class"):
                    # Sample replayed training data, move to correct device
                    x_, y_ = next(data_loader_previous)
                    x_ = x_.to(device)
                    y_ = y_.to(device) if (model.replay_targets=="hard") else None
                    # If required, get target scores (i.e, [scores_]         -- using previous model, with no_grad()
                    if (model.replay_targets=="soft"):
                        with torch.no_grad():
                            scores_ = previous_model(x_)
                        scores_ = scores_[:, :(classes_per_task*(task-1))] if scenario=="class" else scores_
                        #-> when scenario=="class", zero probabilities will be added in the [utils.loss_fn_kd]-function
                elif scenario=="task":
                    # Sample replayed training data, wrap in (cuda-)Variables and store in lists
                    x_ = list()
                    y_ = list()
                    up_to_task = task if replay_mode=="offline" else task-1
                    
                    for task_id in range(up_to_task):
                        x_temp, y_temp = next(data_loader_previous[task_id])
                        x_.append(x_temp.to(device))
                        # -only keep [y_] if required (as otherwise unnecessary computations will be done)
                        if model.replay_targets=="hard":
                            y_temp = y_temp - (classes_per_task*task_id) #-> adjust y-targets to 'active range'
                            y_.append(y_temp.to(device))
                        else:
                            y_.append(None)
                            
                    # If required, get target scores (i.e, [scores_]         -- using previous model
                    if (model.replay_targets=="soft") and (previous_model is not None):
                        scores_ = list()
                        for task_id in range(up_to_task):
                            with torch.no_grad():
                                scores_temp = previous_model(x_[task_id])
                            scores_temp = scores_temp[:, (classes_per_task*task_id):(classes_per_task*(task_id+1))]
                            scores_.append(scores_temp)

            #---> Train MAIN MODEL
            if batch_index <= iters:
                loss_dict = model.train_a_batch(x, y, x_=x_, y_=y_, scores=scores, scores_=scores_,
                                                active_classes=active_classes, task=task, rnt = 1./task)


                # Fire callbacks (for visualization of training-progress / evaluating performance after each task)
                for loss_cb in loss_cbs:
                    if loss_cb is not None:
                        loss_cb(progress, batch_index, loss_dict, task=task)
                        
                        
                for eval_cb in eval_cbs:
                    if eval_cb is not None:
                        eval_cb(model, batch_index, task=task)
                        
                
        ##----------> UPON FINISHING EACH TASK...
        # Close progres-bar(s)
        progress.close()
        
        task_end_training = time.time()
        task_training_time = (task_end_training - task_start_training)/60
        
        path = '../../ember_saved_model/' + args.scenario + '/'
        create_parent_folder(path)
        
        model_path = path + str(args.replay_config) + '_' + str(args.num_replay_sample) + '.pt'
        #print(f'{path}')
        torch.save(model.state_dict(), model_path)  


        # Calculate statistics required for metrics
        for metric_cb in metric_cbs:
            if metric_cb is not None:
                metric_cb(args, model, iters, task=task)
        
        #metrics_avg_callback = metric_cbs['average']
        #print(f'metric_cbs average {metrics_avg_callback} Mean {np.mean(metrics_avg_callback)}')
        
        # REPLAY: update source for replay
        previous_model = copy.deepcopy(model).eval()
        if replay_mode == 'current':
            Current = True
        elif replay_mode in ('exemplars', 'exact'):
            Exact = True
            if replay_mode == "exact":
                previous_datasets = train_datasets[:task]
            else:
                if scenario == "task":
                    previous_datasets = []
                    for task_id in range(task):
                        previous_datasets.append(
                            ExemplarDataset(
                                model.exemplar_sets[
                                (classes_per_task * task_id):(classes_per_task * (task_id + 1))],
                                target_transform=lambda y, x=classes_per_task * task_id: y + x)
                        )
                else:
                    #print(f'i am here ')
                    target_transform = (lambda y, x=classes_per_task: y % x) if scenario == "domain" else None
                    previous_datasets = [
                        ExemplarDataset(model.exemplar_sets, target_transform=target_transform)]
        
        

        accs = [evaluate.validate(
                model, test_datasets[i], verbose=True, test_size=None, task=i+1, with_exemplars=False,
                allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
            ) for i in range(task)] 

        # accs = [evaluate.validate(
        #         model, test_datasets[i], verbose=True, test_size=None, task=i+1, with_exemplars=False,
        #         allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
        #     ) for i in range(task)]
        
#         precisions = [evaluate.validate(
#                 model, test_datasets[i], whichMetric='precision', verbose=True, test_size=None, task=i+1, with_exemplars=False,
#                 allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
#             ) for i in range(task)]
        
#         recalls = [evaluate.validate(
#                 model, test_datasets[i], whichMetric='recall', verbose=True, test_size=None, task=i+1, with_exemplars=False,
#                 allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
#             ) for i in range(task)]
        
#         f1scores = [evaluate.validate(
#                 model, test_datasets[i], whichMetric='f1score', verbose=True, test_size=None, task=i+1, with_exemplars=False,
#                 allowed_classes=list(range(classes_per_task*i, classes_per_task*(i+1))) if scenario=="task" else None
#             ) for i in range(task)]
        
        
#         print(f'\n\naccs {accs}\n precisions {precisions}\n recalls {recalls}\n f1scores {f1scores}', flush=True)
        
        #print(f'\n\nacc current task {accs[task-1]}\n')
        # if task-1 != args.tasks:
        #     #print(f'\n\nacc next tasks {accs[task:]}\n')
        #     results_f = open(args.r_dir, 'a') 
        #     result_string = '{}\t{:.5f}\t{}\t{:.5f}\t\n'.format(task, np.mean(accs), num_training_samples, task_training_time)
        #     results_f.write(result_string)
        #     results_f.flush()
        #     results_f.close() 
    
              
        
        
        results_f = open(args.r_dir, 'a') 
        result_string = '{}\t{:.5f}\t{}\n'.format(task, np.mean(accs), num_training_samples)
        results_f.write(result_string)
        results_f.flush()
        results_f.close() 
              
        
        
    
        
        