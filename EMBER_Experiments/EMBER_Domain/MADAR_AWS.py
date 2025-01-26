import numpy as np
import random
import datetime
import time
import os
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from tqdm import tqdm
import ast
import pandas as pd
import seaborn as sns
from sklearn.utils import shuffle
import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
plt.rcParams['font.size'] = 18
#plt.rcParams['font.family'] = "serif"
tdir = 'in'
major = 5.0
minor = 3.0
plt.rcParams['xtick.direction'] = tdir
plt.rcParams['ytick.direction'] = tdir
plt.rcParams['xtick.major.size'] = major
plt.rcParams['xtick.minor.size'] = minor
plt.rcParams['ytick.major.size'] = major
plt.rcParams['ytick.minor.size'] = minor


from ember_utils import *
from ember_model import *
from ember_pjr_utils import *



class Ember_MLP_Net_V2(nn.Module):
    def __init__(self):
        super(Ember_MLP_Net_V2, self).__init__()
        
        input_features = 2381
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
        #x = self.fc1_bn(x)
        x = self.act1(x) 
        x = self.fc1_drop(x)

        x = self.fc2(x)
        #x = self.fc2_bn(x)
        x = self.act2(x) 
        x = self.fc2_drop(x)
        
        x = self.fc3(x)
        #x = self.fc3_bn(x)
        x = self.act3(x) 
        x = self.fc3_drop(x)
        
        x = self.fc4(x)
        #x = self.fc4_bn(x)
        x = self.act4(x)
        x = self.fc4_drop(x)
        
        x = self.fc_last(x)
        x = self.out(x)
        return x


def get_dataloader_weights(X, y, batch_size):
    
    X_ = torch.from_numpy(np.array(X)).type(torch.FloatTensor)
    y_ = torch.from_numpy(y).type(torch.FloatTensor)
    
    data_tensored = torch.utils.data.TensorDataset(X_,y_)    
    
    data_loader = torch.utils.data.DataLoader(data_tensored, batch_size = batch_size,
                                              num_workers=1, drop_last=False)
    return data_loader



def get_weights(model, layer, X_, Y_, batch_size, device):
    
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach()
        return hook
    
    if layer == 'fc2':
        model.fc2.register_forward_hook(get_activation(layer))
    elif layer == 'fc3':
        model.fc3.register_forward_hook(get_activation(layer))
    elif layer == 'fc4':
        model.fc4.register_forward_hook(get_activation(layer))
    elif layer == 'fc2_bn':
        model.fc2_bn.register_forward_hook(get_activation(layer))
    elif layer == 'fc3_bn':
        model.fc3_bn.register_forward_hook(get_activation(layer))
    elif layer == 'fc4_bn':
        model.fc4_bn.register_forward_hook(get_activation(layer))
    elif layer == 'act2':
        model.act2.register_forward_hook(get_activation(layer))
    elif layer == 'act3':
        model.act3.register_forward_hook(get_activation(layer))
    elif layer == 'act4':
        model.act4.register_forward_hook(get_activation(layer)) 
    
    dataloader = get_dataloader_weights(X_, Y_, batch_size)   
    
    model.eval()
    
    features = []

    with torch.no_grad():
        for x_batch, y_batch in tqdm(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_test_pred = model(x_batch)
            feats_batch = activation[layer].cpu().numpy()
            
            for f in feats_batch:
                features.append(f)
 
            
    assert len(features) == len(X_)      
    return np.array(features)

    
def GetFamilyDict(X_train, Y_train, Y_train_family,\
                  task_month, mal_cnt, global_family_dict):
    count = 0
    for x_ind, x_sample in enumerate(X_train):
        count += 1
        #print(x_ind, Y_train[x_ind])

        if Y_train[x_ind] == 0:
            global_family_dict["goodware"].append(x_sample)
        if Y_train[x_ind] == 1:
            mal_cnt += 1
            
            if Y_train_family[x_ind] == '':
                global_family_dict["others_family"].append(x_sample)
            else:
                global_family_dict[Y_train_family[x_ind]].append(x_sample)
    
    #print(f'Task {task_month} and #-of new samples stored {count}')
    
    return global_family_dict, mal_cnt



def get_anomalyScoresSamples(fam, weight_data,\
                             raw_data, samples_in_family,\
                             chooseSample = True):
    clf = IsolationForest(max_samples=len(weight_data))
    clf.fit(weight_data)
    #scores_prediction = clf.decision_function(weight_data)
    y_pred = clf.predict(weight_data)


    anomalous_idx = np.where(y_pred == -1.0)
    similar_idx = np.where(y_pred == 1.0)

    #print(f'family {fam} anomaly-similar: {len(anomalous_idx[0])} - {len(similar_idx[0])}')
    assert len(anomalous_idx[0]) + len(similar_idx[0]) == len(y_pred)
   
    
    if chooseSample:
        raw_data = np.array(raw_data)
        anomalous_samples = raw_data[anomalous_idx]
        
        if len(anomalous_samples) >= num_samples_per_malware_family:
            anomalous_samples_pool = list(anomalous_samples)
            remaining_samples_to_pick = int(num_samples_per_malware_family/2)
            anomalous_samples = random.sample(anomalous_samples_pool, remaining_samples_to_pick)

        else:
            remaining_samples_to_pick = num_samples_per_malware_family - len(anomalous_samples)        
        
        if remaining_samples_to_pick >= len(similar_idx):
            similar_samples = raw_data[similar_idx]
        else:
            similar_samples_pool = list(raw_data[similar_idx])
            similar_samples = random.sample(similar_samples_pool, remaining_samples_to_pick)
                
        try:
            mal_replay_samples = np.concatenate((np.array(anomalous_samples), np.array(similar_samples)))
        except:
            print(np.array(anomalous_samples).shape, np.array(similar_samples).shape)

        return mal_replay_samples
    else:
        return scores_prediction    

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
        data_X = np.array(data_X)
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


def AWS(GFamilyDict, memory_budget, model, layer, batch_size, device,\
        goodware_ifs=False, min_samples=1, fs_option='ratio'):
    #fs_option = 'uniform'
    #memory_budget = 1000
    goodware_budget = malware_budget = int(np.ceil(memory_budget/2))
    
    num_families = len(GFamilyDict.keys()) - 1 
    pre_malSamples = []
    #cnt = 0
    #fam_cnt = 0
    for k, v in GFamilyDict.items():
        if k != 'goodware':
            if malware_count > malware_budget:
                if fs_option != 'gifs':
                    #fam_cnt += 1
                    v = np.array(v)
                    #print(f'{k} - {len(v)}')
                    #cnt += len(v)

                    if fs_option == 'ratio':
                        v_choose = int(np.ceil((len(v) / malware_count) * malware_budget))

                    if fs_option == 'uniform':
                        v_choose = int(np.ceil(malware_budget / num_families))

                    if fs_option == 'mix':
                        v_choose = int(np.ceil((len(v) / malware_count) * malware_budget))
                        if v_choose < min_samples:
                            #print(f'v_choose {v_choose} min_samples {min_samples}')
                            v_choose = min_samples
                        #else: print(f'v_choose {v_choose} **')                

                    if len(v) <= v_choose:
                        for i in v:
                            pre_malSamples.append(i)
                    else:
                        v_Y = np.ones(len(v))
                        v_weights = get_weights(model, layer, v, v_Y, batch_size, device)
                        v = AWS_Samples(v, v_weights, v_choose, get_anomalous=True, contamination=0.1)

                        for i in v:
                            pre_malSamples.append(i)
                else:
                    for i in v:
                        pre_malSamples.append(i)
            else:
                for i in v:
                    pre_malSamples.append(i)
    
    if fs_option == 'gifs':
        if malware_budget < len(pre_malSamples):
            pre_malSamples = random.sample(list(pre_malSamples), malware_budget)
    
    
    all_Goodware = GFamilyDict['goodware']
    if goodware_ifs:
        #print(f'I am here NOW.')
        v = all_Goodware
        pre_GoodSamples = []     
        if len(all_Goodware) > goodware_budget:
            
            v_Y = np.ones(len(v))
            v_weights = get_weights(model, layer, v, v_Y, batch_size, device)
            v_choose = goodware_budget
            v = AWS_Samples(v, v_weights, v_choose, get_anomalous=True, contamination=0.1)
            for i in v:
                pre_GoodSamples.append(i)
        else:
            for i in v:
                pre_GoodSamples.append(i)
    else:
        if goodware_budget > len(all_Goodware):
            pre_GoodSamples = all_Goodware
        else:
            pre_GoodSamples = random.sample(list(all_Goodware), goodware_budget)
    
    print(f'pre_GoodSamples {np.array(pre_GoodSamples).shape} pre_malSamples {np.array(pre_malSamples).shape}')
    samples_to_replay = np.concatenate((np.array(pre_GoodSamples), np.array(pre_malSamples)))
    labels_to_replay = np.concatenate((np.zeros(len(pre_GoodSamples)), np.ones(len(pre_malSamples))))
    
    X_replay, Y_replay = shuffle(samples_to_replay, labels_to_replay)
    
    return X_replay, Y_replay

    

def get_anomalySamples(family_dict,\
                       samples_in_family,\
                       model, layer, device):
    
    pre_malware_samples = []
    for k, v in family_dict.items():
        #print(f'family {k} ....')
        if k != 'goodware':
            if len(v) > samples_in_family:
                #print(f'family {k} to IF.')
                
                k_samples = v
                k_Y = np.ones(len(k_samples))
                k_weights = get_weights(model, layer, k_samples, k_Y, batch_size, device)
                
                #print(f'k_samples {len(k_samples)}  k_weights {len(k_weights)}')
                
                k_selected_samples = AWS_Samples(k, k_weights,\
                                                     k_samples, samples_in_family,\
                                                     chooseSample = True)
                for sample in k_selected_samples:
                    pre_malware_samples.append(sample)
                
                family_dict[k] = list(k_selected_samples)
            else:
                for sample in v:
                    pre_malware_samples.append(sample)
    

    if len(family_dict['goodware']) < len(pre_malware_samples):
        pre_goodware_samples = random.sample(family_dict['goodware'], len(family_dict['goodware']))
    else:
        pre_goodware_samples = random.sample(family_dict['goodware'], len(pre_malware_samples))
    
    family_dict['goodware'] = list(pre_goodware_samples)
    
    replay_samples = np.concatenate((list(pre_goodware_samples),\
                                     list(pre_malware_samples)))
    labels_to_replay = np.concatenate((list(np.zeros(len(pre_goodware_samples))),\
                                       (np.ones(len(pre_malware_samples)))))
    
    #random.shuffle(replay_samples, labels_to_replay)
    from sklearn.utils import shuffle
    X_, Y_ = shuffle(replay_samples, labels_to_replay)
    
    return X_, Y_, family_dict


parser = argparse.ArgumentParser()
parser.add_argument('--num_exps', type=int, default=1, required=False, help='Number of Experiments to Run.')
parser.add_argument('--contamination', type=float, default=0.1, required=False)
parser.add_argument('--num_epoch', type=int, default=500, required=False)
parser.add_argument('--batch_size', type=int, default=2000, required=False)
parser.add_argument('--memory_budget', type=int, required=True)
parser.add_argument('--min_samples', type=int, default=1, required=False)
parser.add_argument('--batch_norm', action="store_true", required=False,\
                    help = 'Get weights from the model with or withour batch normalization layer.')
parser.add_argument('--layer', default='act4', choices=["act4", "fc4", "fc4_bn"], required=False)

parser.add_argument('--ifs_option', type=str,\
                    required=True, choices=['ratio', 'uniform', 'gifs', 'mix'])
parser.add_argument('--goodware_ifs', action="store_true", required=False)
parser.add_argument('--data_dir', type=str,\
                    default='../../../month_based_processing_with_family_labels/', required=False)

args = parser.parse_args()


all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']


patience = 5
replay_type = ifs_option = args.ifs_option
data_dir = args.data_dir
num_exps = args.num_exps
num_epoch = args.num_epoch
batch_size = args.batch_size
memory_budget = args.memory_budget
min_samples = args.min_samples
layer = args.layer #'act4'
batch_norm = args.batch_norm
contamination = args.contamination #0.1 #[0.2, 0.3, 0.4, 0.5]

replay_type = 'aws'

exp_type = 'weights'

exp_seeds = [random.randint(1, 99999) for i in range(num_exps)]


expSaveDir = '../../AWS_Final_' + str(contamination) + '_'
resSaveDir = '../../AWS_Results'
expSaveFile = '/AWS_'  + str(contamination) + '_'


raw_anomalyScores_Dict = {}
weight_anomalyScores_Dict = {}

restartMonth = 100
cnt =  1    
for exp in exp_seeds:
    start_time = time.time()
    use_cuda = True
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(exp)

    model = Ember_MLP_Net()
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.000001)
       
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)
    print(f'Model has {count_parameters(model)/1000000}m parameters')    
    criterion = nn.BCELoss()    

    GFamilyDict = defaultdict(list)
    malware_count = 0
    
    standardization = StandardScaler()
    standard_scaler = None
    for task_month in range(len(all_task_months)):
                
        print(f'\n{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Round {cnt} ...')
        task_start = time.time()
        
        #task_month = task_month
        current_task = all_task_months[task_month]
        task_months = all_task_months[:task_month+1]
        print(f'Current Task {current_task} w/ memory_budget {memory_budget}')


        model_save_dir = '../../AWS_SavedModel' + '/AWSModel_' + str(memory_budget) + '/' + str(current_task) + '/'
        create_parent_folder(model_save_dir)
        
        opt_save_path = '../../AWS_SavedModel' + '/AWSOpt_' + str(memory_budget) + '/' + str(current_task) + '/'
        create_parent_folder(opt_save_path)
        
        results_save_dir =  '../../AWS_SavedResults_' +'/AWS_' + str(memory_budget) + '/' 
        create_parent_folder(results_save_dir)
        
        
        if task_month == restartMonth -1:
            tmpcurrent_task = all_task_months[task_month - 1]
            tmp_model_save_dir = str(expSaveDir) + 'model_' +\
                    str(exp_type) + str(expSaveFile) +\
                    str(memory_budget) + '/' + str(tmpcurrent_task) + '/'
            
            dict_save_file = tmp_model_save_dir + 'global_family_dict_' + str(task_month) + '.npz'
            dictFile = np.load(dict_save_file)
            GFamilyDict = dictFile['dictfile']
        else:
            dict_save_file = model_save_dir + 'global_family_dict_' + str(task_month) + '.npz'
            np.savez_compressed(dict_save_file, dictfile = GFamilyDict)
        
        
        
        X_train, Y_train, Y_train_family = get_family_labeled_month_data(data_dir, current_task)
        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_months, mlp_net=True)
        
        
        if current_task == all_task_months[0]:
            GFamilyDict, malware_count = GetFamilyDict(X_train, Y_train, Y_train_family,\
                                                   current_task, malware_count, GFamilyDict)
            num_Y_replay = 0
        else:
            if batch_norm == 'yes':
                if current_task != all_task_months[0]:
                    X_replay, Y_replay = AWS(GFamilyDict, memory_budget, model,\
                                          layer, batch_size, device,\
                                          goodware_ifs=args.goodware_ifs, min_samples=min_samples,
                                          fs_option=ifs_option)



                results_f = open(os.path.join(results_save_dir + layer + '_aws_bn_' + str(memory_budget) + '_results.txt'), 'a')

            else:
                if current_task != all_task_months[0]:
                    model_new = Ember_MLP_Net_V2()
                    model_new = model_new.to(device)
                    #load the best model for this task
                    #best_model_path = model_save_dir + os.listdir(model_save_dir)[0]
                    print(f'loading best model {best_model_path}')
                    model_new.load_state_dict(torch.load(best_model_path))

                    X_replay, Y_replay = AWS(GFamilyDict, memory_budget, model_new,\
                                          layer, batch_size, device,\
                                          goodware_ifs=args.goodware_ifs, min_samples=min_samples,
                                          fs_option=ifs_option)

                results_f = open(os.path.join(results_save_dir + layer + '_aws_wobn_' + str(memory_budget) + '_results.txt'), 'a')
            
            num_Y_replay = len(Y_replay)    
            GFamilyDict, malware_count = GetFamilyDict(X_train, Y_train, Y_train_family,\
                                                   current_task, malware_count, GFamilyDict)
            print(f'Replay Samples {len(Y_replay)}')
            X_train, Y_train = np.concatenate((X_train, X_replay)), np.concatenate((Y_train, Y_replay))
            
            
        
        X_train, Y_train = shuffle(X_train, Y_train)
       
        print()
        print(f'X_train {X_train.shape} Y_train {Y_train.shape}')
        print()
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Standardizing ...')
        standard_scaler = standardization.partial_fit(X_train)

        X_train = standard_scaler.transform(X_train)
        X_test = standard_scaler.transform(X_test)

        X_train, Y_train = np.array(X_train, np.float32), np.array(Y_train, np.int32)
        X_test, Y_test = np.array(X_test, np.float32), np.array(Y_test, np.int32)
    
        
        print(f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Training ...')
        task_training_time, epoch_ran, training_loss, validation_loss  =\
                                training_early_stopping(model, model_save_dir, opt_save_path,\
                                X_train, Y_train, X_test, Y_test, patience,\
                                batch_size, device, optimizer, num_epoch,\
                                 criterion, replay_type, current_task, exp, earlystopping=True)
        
        
        
        model = Ember_MLP_Net()
        model = model.to(device)
        #load the best model for this task
        best_model_path = model_save_dir + os.listdir(model_save_dir)[0]
        print(f'loading best model {best_model_path}')
        model.load_state_dict(torch.load(best_model_path))
        
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.000001)
        best_optimizer = opt_save_path + os.listdir(opt_save_path)[0]
        print(f'loading best optimizer {best_optimizer}')
        optimizer.load_state_dict(torch.load(best_optimizer))
        
        acc, precision, recall, f1score = testing_aucscore(model, X_test, Y_test, batch_size, device)
        
        
        end_time = time.time()

        print(f'Elapsed time {(end_time - start_time)/60} mins.')    
        

        task_end = time.time()
        task_run_time = (task_end - task_start)/60
        
        
        num_replay_samples = len(X_train)
        
        if args.goodware_ifs:
            if ifs_option != 'mix':
                results_f = open(os.path.join('./Results_Domain/' + 'aws_good_' + str(ifs_option) + '_' + str(memory_budget) + '_results.txt'), 'a')
            else:
                results_f = open(os.path.join('./Results_Domain/' + 'aws_good_' + str(ifs_option) + '_' + str(min_samples) + '_' + str(memory_budget) + '_results.txt'), 'a')
        else:
            if ifs_option != 'mix':
                results_f = open(os.path.join('./Results_Domain/' + 'aws_' + str(ifs_option) + '_' + str(memory_budget) + '_results.txt'), 'a')
            else:
                results_f = open(os.path.join('./Results_Domain/' + 'aws_' + str(ifs_option) + '_' + str(min_samples) + '_' + str(memory_budget) + '_results.txt'), 'a')


        result_string = '{}\t{}\t{}\t{}\t{}\t{}\t\n'.format(current_task, acc, precision, recall, f1score, num_Y_replay)
        
        results_f.write(result_string)
        results_f.flush()
        results_f.close()
        
    
    end_time = time.time()
    cnt += 1
    print(f'Elapsed time {(end_time - start_time)/60} mins.')
    
    del model_save_dir
    del opt_save_path
    del results_save_dir
