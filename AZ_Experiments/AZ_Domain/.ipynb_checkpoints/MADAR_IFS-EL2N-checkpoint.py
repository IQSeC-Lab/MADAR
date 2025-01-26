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
#import seaborn as sns
from sklearn.utils import shuffle

import argparse
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader

from az_utils import *
from ember_model import *


def GetFamilyDict(X_train, Y_train, Y_train_family,\
                  task_year, mal_cnt, global_family_dict):
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
    
    #print(f'Task {task_year} and #-of new samples stored {count}')
    
    return global_family_dict, mal_cnt

def IFS_Samples(v, v_choose, get_anomalous=True, contamination=0.1):
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
            #print(f'anomalous_samples_pool {len(anomalous_samples_pool)} > v_choose_split {v_choose_split}')
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


def MixSampleCount(GBudget, MinBudget, GFamilyDict):
    
    import copy 
    tmpBudget = copy.deepcopy(GBudget)
    print(f'budget unallocated {GBudget}')
    
    GfamStat = {}
    
    for fam, S in GFamilyDict.items():
        if fam != 'goodware':
            GfamStat[fam] = len(S)
    
    assert len(GfamStat.keys()) == len(GFamilyDict.keys()) - 1
    
    GfamChoose = {}
    GfamTemp = {}
    
    allocated = 0
    for fam, numSample in GfamStat.items():
        if numSample > MinBudget:
            GfamChoose[fam] = MinBudget
            
            #print(f'numSample {numSample} MinBudget {MinBudget}')
            
            GfamTemp[fam] = numSample - MinBudget
            GBudget -= MinBudget
            allocated += MinBudget
        else:
            #print(f'fam {fam} numSample {numSample}')
            GfamChoose[fam] = numSample
            GfamTemp[fam] = 0
            GBudget -= numSample
            allocated += numSample

    UnallocatedSamples = int(sum(GfamTemp.values()))
    
    if allocated > tmpBudget:
        print(f'GBudget {tmpBudget} allocated {allocated}')
        print(f'reduce minimum samples, budget is lower than required allocation')
        
        
    #print(f'allocated {allocated} unallocated {GBudget} Sample remainin {UnallocatedSamples}')
    for fam, numSample in GfamTemp.items():
        if numSample != 0:
            #print(f'here ')
            allocate = int(np.round((numSample/UnallocatedSamples) * GBudget))
            #print(f'GBudget {GBudget} {np.round(numSample/UnallocatedSamples)} allocate {allocate}')
            GfamChoose[fam] += allocate
    
    return GfamChoose

def IFS(GFamilyDict, memory_budget,\
        goodware_ifs=False, min_samples=1, fs_option='ratio'):
    #fs_option = 'uniform'
    #memory_budget = 1000
    
    goodware_budget = int(np.round(memory_budget * 0.9))
    malware_budget = memory_budget - goodware_budget
    
    num_families = len(GFamilyDict.keys()) - 1 
    pre_malSamples = []
    #cnt = 0
    #fam_cnt = 0
    
    if malware_count > malware_budget:
        if fs_option == 'mix':
            GfamChoose = MixSampleCount(malware_budget, min_samples, GFamilyDict)
    
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
                        #print(f'malware_count {malware_count} > malware_budget {malware_budget}')
                        v_choose = GfamChoose[k]
                        #print(f'v_choose {v_choose} **')
                        
#                         v_choose = int(np.ceil((len(v) / malware_count) * malware_budget))
#                         if v_choose < min_samples:
#                             #print(f'v_choose {v_choose} min_samples {min_samples}')
#                             v_choose = min_samples
#                         #else: print(f'v_choose {v_choose} **')                

                    if len(v) <= v_choose:
                        for i in v:
                            pre_malSamples.append(i)
                    else:
                        v = IFS_Samples(v, v_choose, get_anomalous=True, contamination=0.1)
                        for i in v:
                            pre_malSamples.append(i)
                else:
                    for i in v:
                        pre_malSamples.append(i)
            else:
                #print(f'malware_count {malware_count} <= malware_budget {malware_budget}')
                for i in v:
                    pre_malSamples.append(i)
    
    if fs_option == 'gifs':
        if malware_budget < len(pre_malSamples):
            pre_malSamples = random.sample(list(pre_malSamples), malware_budget)
    
    
    all_Goodware = GFamilyDict['goodware']
    if goodware_ifs:
        #print(f'I am here NOW.')
        pre_GoodSamples = []
        v = np.array(all_Goodware)
        v_choose = goodware_budget
        v = IFS_Samples(v, v_choose, get_anomalous=True, contamination=0.1)
        for i in v:
            pre_GoodSamples.append(i)
    else:
        if goodware_budget > len(all_Goodware):
            pre_GoodSamples = all_Goodware
        else:
            pre_GoodSamples = random.sample(list(all_Goodware), goodware_budget)
    
    #print(f'\n\nReplay Goodware {len(pre_GoodSamples)} Replay Malware {len(pre_malSamples)}\n\n')
    samples_to_replay = np.concatenate((np.array(pre_GoodSamples), np.array(pre_malSamples)))
    labels_to_replay = np.concatenate((np.zeros(len(pre_GoodSamples)), np.ones(len(pre_malSamples))))
    
    X_replay, Y_replay = shuffle(samples_to_replay, labels_to_replay)
    
    return X_replay, Y_replay



parser = argparse.ArgumentParser()
parser.add_argument('--num_exps', type=int, default=3, required=False, help='Number of Experiments to Run.')
parser.add_argument('--contamination', type=float, default=0.1, required=False)
parser.add_argument('--num_epoch', type=int, default=200, required=False)
parser.add_argument('--batch_size', type=int, default=2048, required=False)
parser.add_argument('--memory_budget', type=int, required=True)
parser.add_argument('--min_samples', type=int, default=1, required=False)
parser.add_argument('--ifs_option', type=str,\
                    required=True, choices=['ratio', 'uniform', 'gifs', 'mix'])
parser.add_argument('--goodware_ifs', action="store_true", required=False)
parser.add_argument('--data_dir', type=str,\
                    default='/home/msrahman3/IQSeC-Datasets/AZ-Datasets/Domain_Transformed', required=False)

args = parser.parse_args()


all_task_years = ['2008', '2009', '2010', '2011', '2012', '2013', '2014', '2015', '2016']

madarEL2N = True

patience = 5
replay_type = ifs_option = args.ifs_option
data_dir = args.data_dir
num_exps = args.num_exps
num_epoch = args.num_epoch
batch_size = args.batch_size
memory_budget = args.memory_budget
min_samples = args.min_samples

contamination = args.contamination #0.1 #[0.2, 0.3, 0.4, 0.5]

exp_seeds = [random.randint(1, 99999) for i in range(num_exps)]


input_features = 1789


cnt =  1    
for exp in exp_seeds:
    start_time = time.time()
    use_cuda = True
    print('Torch', torch.__version__, 'CUDA', torch.version.cuda)
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.manual_seed(exp)

    model = Ember_MLP_Net(input_features)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.000001)
       
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
    for task_year in range(len(all_task_years)):
                
        print(f'\n{datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} Round {cnt} ...')
        task_start = time.time()
        
        current_task = all_task_years[task_year]
        task_years = all_task_years[:task_year+1]
        print(f'Current Task {current_task} with Budget {memory_budget}')
        
        model_save_dir = '../az_model/IFS_SavedModel' + '/IFSModel_' + str(memory_budget) + '/' + str(current_task) + '/'
        create_parent_folder(model_save_dir)
        
        opt_save_path = '../az_model/IFSS_SavedModel' + '/IFSOpt_' + str(memory_budget) + '/' + str(current_task) + '/'
        create_parent_folder(opt_save_path)
        
        results_save_dir =  '../az_model/IFS_SavedResults_' +'/IFS_' + str(memory_budget) + '/' 
        create_parent_folder(results_save_dir)
        
        
        X_train, Y_train, Y_train_family = get_family_labeled_year_data(data_dir, current_task)
        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_years, mlp_net=True)
        

        if current_task == all_task_years[0]:
            GFamilyDict, malware_count = GetFamilyDict(X_train, Y_train, Y_train_family,\
                                                   current_task, malware_count, GFamilyDict)
            num_Y_replay = 0
        else:
            X_replay, Y_replay = IFS(GFamilyDict, memory_budget, goodware_ifs=args.goodware_ifs,\
                                     min_samples=min_samples, fs_option=ifs_option)
            num_Y_replay = len(Y_replay)
            
            GFamilyDict, malware_count = GetFamilyDict(X_train, Y_train, Y_train_family,\
                                           current_task, malware_count, GFamilyDict)
            
            
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
        
        
        
        best_model_path = model_save_dir + os.listdir(model_save_dir)[0]
        print(f'loading best model {best_model_path}')
        model.load_state_dict(torch.load(best_model_path, weights_only=True))

        #optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.000001)
        best_optimizer = opt_save_path + os.listdir(opt_save_path)[0]
        print(f'loading best optimizer {best_optimizer}')
        optimizer.load_state_dict(torch.load(best_optimizer, weights_only=True))


        acc, rocauc, precision, recall, f1score = testing_aucscore(model, X_test, Y_test, batch_size, device)
        end_time = time.time()
        print(f'Elapsed time {(end_time - start_time)/60} mins.') 
        
        task_end = time.time()
        task_run_time = (task_end - task_start)/60
        
        if madarEL2N:
            results_save_dir = './Results_Domain_EL2N/'
            create_parent_folder(results_save_dir)
        else:
            results_save_dir = './Results_Domain/'
            create_parent_folder(results_save_dir)
        
        if args.goodware_ifs:
            if ifs_option != 'mix':
                results_f = open(os.path.join(results_save_dir + 'ifs_good_' + str(ifs_option) + '_' + str(memory_budget) + '_results.txt'), 'a')
            else:
                results_f = open(os.path.join(results_save_dir + 'ifs_good_' + str(ifs_option) + '_' + str(min_samples) + '_' + str(memory_budget) + '_results.txt'), 'a')
        else:
            if ifs_option != 'mix':
                results_f = open(os.path.join(results_save_dir + 'ifs_' + str(ifs_option) + '_' + str(memory_budget) + '_results.txt'), 'a')
            else:
                results_f = open(os.path.join(results_save_dir + 'ifs_' + str(ifs_option) + '_' + str(min_samples) + '_' + str(memory_budget) + '_results.txt'), 'a')

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