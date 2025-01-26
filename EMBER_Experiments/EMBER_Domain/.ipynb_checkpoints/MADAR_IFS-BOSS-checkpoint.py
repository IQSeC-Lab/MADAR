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
from torch.utils.data import Dataset, DataLoader, TensorDataset

from ember_utils import *
from ember_model import *
from ember_pjr_utils import *
from EL2N import *

# def compute_el2n_scores_v1(model, dataloader, device, num_classes=2):
#     el2n_scores = []

#     # Dynamically choose loss function based on the number of classes
#     if num_classes == 2:
#         criterion = nn.BCEWithLogitsLoss(reduction='none')
#     else:
#         criterion = nn.CrossEntropyLoss(reduction='none')  # Multi-class

#     with torch.no_grad():
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
            
#             if num_classes == 2:
#                 # Binary classification
#                 outputs = outputs.squeeze(-1)  # Shape [batch_size]
#                 targets = targets.float()  # BCELoss expects float targets
#             else:
#                 # Multi-class classification
#                 targets = targets.long()  # CrossEntropyLoss expects int targets

#             try:
#                 loss = criterion(outputs, targets)
#                 el2n_scores.extend(loss.cpu().numpy())
#             except RuntimeError as e:
#                 print(f"Error in batch. Outputs: {outputs.shape}, Targets: {targets.shape}")
#                 raise e
    
#     return np.array(el2n_scores)

# def compute_el2n_scores(model, dataloader, device, num_classes=2):
#     el2n_scores = []
#     labels = []

#     # Dynamically choose loss function based on the number of classes
#     if num_classes == 2:
#         criterion = nn.BCEWithLogitsLoss(reduction='none')
#     else:
#         criterion = nn.CrossEntropyLoss(reduction='none')  # Multi-class

#     with torch.no_grad():
#         for inputs, targets in dataloader:
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
            
#             if num_classes == 2:
#                 # Binary classification
#                 outputs = outputs.squeeze(-1)  # Shape [batch_size]
#                 targets = targets.float()  # BCELoss expects float targets
#             else:
#                 # Multi-class classification
#                 targets = targets.long()  # CrossEntropyLoss expects int targets

#             try:
#                 loss = criterion(outputs, targets)
#                 el2n_scores.extend(loss.cpu().numpy())
#                 labels.extend(targets.cpu().numpy())  # Store labels
#             except RuntimeError as e:
#                 print(f"Error in batch. Outputs: {outputs.shape}, Targets: {targets.shape}")
#                 raise e
    
#     return np.array(el2n_scores), np.array(labels)


# def select_top_samples_v1(data, labels, el2n_scores, budget):
#     # Sort indices by EL2N scores in descending order
#     top_indices = np.argsort(el2n_scores)[-budget:]

#     # Extract data and labels corresponding to the selected indices
#     selected_data = data[top_indices]
#     selected_labels = labels[top_indices]

#     return selected_data, top_indices, selected_labels

# def select_top_samples(data, labels, el2n_scores, budget):
#     # Detect the number of unique classes
#     unique_classes = np.unique(labels)
#     num_classes = len(unique_classes)

#     # Sort indices by EL2N scores in descending order
#     top_indices = np.argsort(el2n_scores)[-budget:]

#     # Handle one-class scenario
#     if num_classes == 1:
#         # Direct selection for one class
#         selected_data = data[top_indices]
#         selected_labels = labels[top_indices]
#         return selected_data, top_indices, selected_labels

#     # Multi-class scenario
#     selected_data = data[top_indices]
#     selected_labels = labels[top_indices]

#     return selected_data, top_indices, selected_labels    

# # Main function for subset selection based on EL2N

# def el2n_subset_selection_v1(model, data, labels, budget, batch_size=32):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     # Prepare dataset and dataloader
#     dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), 
#                             torch.tensor(labels, dtype=torch.long))
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     # Compute EL2N scores
#     el2n_scores = compute_el2n_scores(model, dataloader, device)

#     # Select top Beta samples
#     selected_data, selected_indices, selected_labels = select_top_samples(data, labels, el2n_scores, budget)

#     print(f"Selected {len(selected_indices)} samples using EL2N.")
#     return selected_data, selected_indices, selected_labels

# def el2n_subset_selection(model, data, labels, budget, num_classes=2, batch_size=32):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model.to(device)

#     # Prepare dataset and dataloader
#     dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), 
#                             torch.tensor(labels, dtype=torch.long))
#     dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

#     # Compute EL2N scores
#     el2n_scores = compute_el2n_scores(model, dataloader, device, num_classes=num_classes)

#     # Select top samples
#     selected_subset, selected_indices, selected_labels = select_top_samples(
#         data, labels, el2n_scores, budget
#     )

#     print(f"Selected {len(selected_subset)} samples using EL2N.")
#     print(f"Number of Classes Detected: {len(np.unique(labels))}")
    
#     return selected_subset, selected_indices, selected_labels




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



# def calculate_importance_scores(model, data, labels, method="gradient"):
#     """
#     Calculates importance scores for 1D data using an MLP model.

#     Args:
#         model: The trained MLP model (PyTorch).
#         data: The 1D input data (PyTorch tensor or NumPy array).
#         labels: The corresponding labels for the data (PyTorch tensor).
#         method: The method to use for calculating importance scores.
#                 Currently supports "gradient" and "perturbation".

#     Returns:
#         A NumPy array of importance scores, one score per data point.
#     """
    
#     device = torch.device("cuda")

#     # Ensure data and labels are tensors and on the same device
#     if isinstance(data, np.ndarray):
#         data = torch.tensor(data, dtype=torch.float32).to(device)
#     else:
#         data = data.to(device).float()  # Ensure data is float32

#     labels = labels.to(device).long()  # Ensure labels are int64 (long)
    
#     if method == "gradient":
#         data.requires_grad = True
#         #print(f'data {data}')
#         outputs = model(data).float()  # Ensure model outputs are float32
#         loss = torch.nn.functional.cross_entropy(outputs, labels)
#         gradients = torch.autograd.grad(loss, data)[0]
#         importance_scores = torch.abs(gradients).mean(dim=1).detach().cpu().numpy()

#     elif method == "perturbation":
#         baseline_predictions = torch.argmax(model(data).float(), dim=1)
#         importance_scores = np.zeros(data.shape[0])

#         for i in range(data.shape[1]):
#             perturbed_data = data.clone().detach()
#             perturbed_data[:, i] += 0.01
#             perturbed_predictions = torch.argmax(model(perturbed_data).float(), dim=1)
#             importance_scores += np.abs(perturbed_predictions.cpu().numpy() - baseline_predictions.cpu().numpy()).mean(axis=0)
#     else:
#         raise ValueError("Invalid method. Choose 'gradient' or 'perturbation'")

#     return importance_scores

def calculate_importance_scores(model, data, labels, method="gradient"):
    """
    Calculates importance scores for 1D data using an MLP model.

    Args:
        model: The trained MLP model (PyTorch).
        data: The 1D input data (PyTorch tensor or NumPy array).
        labels: The corresponding labels for the data (PyTorch tensor).
        method: The method to use for calculating importance scores.
                Supports "gradient" and "perturbation".

    Returns:
        A NumPy array of importance scores, one score per data point.
    """
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Ensure data and labels are tensors and on the same device
    if isinstance(data, np.ndarray):
        data = torch.tensor(data, dtype=torch.float32).to(device)
    else:
        data = data.to(device).float()

    labels = labels.to(device).float()  # Labels should be float for BCE
    
    if method == "gradient":
        data.requires_grad = True
        outputs = model(data).squeeze(1)  # Ensure outputs shape [batch_size]
        loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs, labels)
        gradients = torch.autograd.grad(loss, data)[0]
        importance_scores = torch.abs(gradients).mean(dim=1).detach().cpu().numpy()

    elif method == "perturbation":
        outputs = torch.sigmoid(model(data)).squeeze(1)  # Get probabilities
        baseline_predictions = (outputs > 0.5).float()
        importance_scores = np.zeros(data.shape[0])

        for i in range(data.shape[1]):
            perturbed_data = data.clone().detach()
            perturbed_data[:, i] += 0.01
            perturbed_outputs = torch.sigmoid(model(perturbed_data)).squeeze(1)
            perturbed_predictions = (perturbed_outputs > 0.5).float()
            
            importance_scores += np.abs(perturbed_predictions.cpu().numpy() - baseline_predictions.cpu().numpy()).mean(axis=0)
    else:
        raise ValueError("Invalid method. Choose 'gradient' or 'perturbation'")

    return importance_scores

    
def boss_selection(model, data, labels, budget_per_class):
    """
    Selects the most important samples using the BOSS method.

    Args:
        model: The trained MLP model (PyTorch).
        data: The 1D input data (PyTorch tensor).
        labels: The corresponding labels for the data (PyTorch tensor).
        budget_per_class: The number of samples to select per class.

    Returns:
        A list of indices representing the selected samples.
    """
    labels = labels.unsqueeze(0) if labels.ndim == 0 else labels
    num_classes = len(torch.unique(labels))

    #print(f'labels {labels}\n\n num_classes {num_classes}')

    #print(f'boss_selection >> labels {labels.shape} data {np.array(data).shape}')

    
    # data = data.to(device)
    # labels = labels.to(device)
    # model = model.to(device)

    
    selected_indices = []

    for class_id in range(num_classes):
        class_id = torch.unique(labels)
        class_indices = torch.where(labels == class_id)[0]

        
        #print(f'class_indices {class_indices}')

        data_ = np.array(data)
        class_data = data_[class_indices]

        #class_data = torch.index_select(data, 0, class_indices)
        
        class_labels = labels[class_indices]

        ### msr, jan 6, problem here.
        #print(class_data, class_labels)
        importance_scores = calculate_importance_scores(model, class_data, class_labels, method="gradient")

        importance_scores = torch.tensor(importance_scores)
        sorted_indices = torch.argsort(importance_scores, descending=True)

        selected_class_indices = sorted_indices[:budget_per_class]
        selected_indices.extend(class_indices[selected_class_indices].tolist())

    return selected_indices




def IFS_Samples(model, v, v_choose, get_anomalous=True, contamination=0.1, madarEL2N=False):
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
            
            if madarEL2N:
                #print(type(similar_samples_pool))
                selected_indices = boss_selection(
                            model,
                            anomalous_samples_pool,
                            torch.tensor(np.ones(len(anomalous_samples_pool))),
                            budget_per_class=v_choose_split
                        )


                anomalous_samples_pool = np.array(anomalous_samples_pool)
                anomalous_samples = anomalous_samples_pool[selected_indices] #.cpu().numpy()

                
                
                # el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
                # model, anomalous_samples_pool, np.ones(len(anomalous_samples_pool)), v_choose_split, num_classes = 2, batch_size=32)
                # anomalous_samples = el2n_subset
            # else:
            #     anomalous_samples = random.sample(anomalous_samples_pool, v_choose_split)

        else:
            anomalous_samples = anomalous_samples_pool

        if len(anomalous_samples) == v_choose_split:
            if madarEL2N:
                # el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
                # model, similar_samples_pool, np.ones(len(similar_samples_pool)), v_choose_split, num_classes = 2, batch_size=32)

                #similar_budget = v_choose - len(anomalous_samples) 
                selected_similar_indices = boss_selection(
                            model,
                            similar_samples_pool,
                            torch.tensor(np.ones(len(similar_samples_pool))),
                            budget_per_class=v_choose_split
                        )
                
                #similar_samples = el2n_subset
                similar_samples_pool = np.array(similar_samples_pool)
                similar_samples = similar_samples_pool[selected_similar_indices]
            # else:
            #     similar_samples = random.sample(similar_samples_pool, v_choose_split)
            
        elif len(anomalous_samples) < v_choose_split:
            v_choose_split += v_choose_split - len(anomalous_samples)
            if len(similar_samples_pool) > v_choose_split:
                if madarEL2N:
                    # el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
                    # model, similar_samples_pool, np.ones(len(similar_samples_pool)), v_choose_split, num_classes = 2, batch_size=32)
                    selected_similar_indices = boss_selection(
                            model,
                            similar_samples_pool,
                            torch.tensor(np.ones(len(similar_samples_pool))),
                            budget_per_class=v_choose_split
                        )
                    
                    
                    # similar_samples = el2n_subset
                    similar_samples_pool = np.array(similar_samples_pool)
                    similar_samples = similar_samples_pool[selected_similar_indices]
                # else:
                #     similar_samples = random.sample(similar_samples_pool, v_choose_split)
                
            else:
                similar_samples = similar_samples_pool
                
        if len(anomalous_samples) > 0 and len(similar_samples) > 0: 
            anomalous_samples, similar_samples = np.array(anomalous_samples), np.array(similar_samples)
            
            #print(f'type(anomalous_samples)  {type(anomalous_samples)}')

            if anomalous_samples.ndim != similar_samples.ndim:
                #print(f'anomalous {anomalous_samples.ndim}, {anomalous_samples}')
    
                #print(f'similar {similar_samples.ndim}, {similar_samples}')

                if anomalous_samples.ndim > similar_samples.ndim:
                    anomalous_samples = np.squeeze(anomalous_samples)
                elif anomalous_samples.ndim < similar_samples.ndim:
                    similar_samples = np.squeeze(similar_samples)
                
            
            # Now concatenate safely
            replay_samples = np.concatenate((anomalous_samples, similar_samples), axis=0)
                        
            #print(f'anomalous_samples {anomalous_samples.shape} similar_samples {similar_samples.shape}')
            replay_samples = np.concatenate((anomalous_samples, similar_samples))
        else:
            if len(anomalous_samples) <= 0:
                replay_samples = similar_samples
            if len(similar_samples) <= 0:
                replay_samples = anomalous_samples
    # else:
    #     similar_samples_pool = list(data_X[similar_idx])
    #     if len(similar_samples_pool) > v_choose:
    #         if madarEL2N:
    #             el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
    #                 model, similar_samples_pool, np.ones(len(similar_samples_pool)), v_choose, num_classes = 2, batch_size=32)
    #             similar_samples = el2n_subset
    #         else:
    #             similar_samples = random.sample(similar_samples_pool, v_choose)
    #     else:
    #         similar_samples = similar_samples_pool
            
    #     replay_samples = np.array(similar_samples)
        
    return replay_samples


def IFS(model, GFamilyDict, memory_budget,\
        goodware_ifs=False, min_samples=1, fs_option='ratio', madarEL2N=False):
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # # Prepare dataset and dataloader
    # dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), 
    #                         torch.tensor(labels, dtype=torch.long))
    # dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
    
    # num_classes = len(np.unique(labels))
    # el2n_scores, el2n_labels = compute_el2n_scores(model, dataloader, device, num_classes=num_classes)

    # for score, label in zip(el2n_scores, el2n_labels):
    #     print(f"Score: {score:.4f}, Label: {label}")
    
        
    
    #fs_option = 'uniform'
    #memory_budget = 1000
    goodware_budget = malware_budget = int(np.ceil(memory_budget/2))
    
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
                        if madarEL2N:
                            v = IFS_Samples(model, v, v_choose, get_anomalous=True, contamination=0.1, madarEL2N=True)
                        else:
                            v = IFS_Samples(model, v, v_choose, get_anomalous=True, contamination=0.1)
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
            print(f'goodware_budget {goodware_budget}')
            if madarEL2N:
                # el2n_subset, selected_indices, selected_labels, el2n_scores = el2n_subset_selection(
                #     model, all_Goodware, np.zeros(len(all_Goodware)), goodware_budget, batch_size=32)
                # pre_GoodSamples = el2n_subset
                selected_goodware_indices = boss_selection(
                            model,
                            all_Goodware,
                            torch.tensor(np.zeros(len(all_Goodware))),
                            budget_per_class=goodware_budget
                        )
                all_Goodware = np.array(all_Goodware)
                pre_GoodSamples = all_Goodware[selected_goodware_indices]
            else:
                pre_GoodSamples = random.sample(list(all_Goodware), goodware_budget)
    
    print(f'\n\nReplay Goodware {len(pre_GoodSamples)} Replay Malware {len(pre_malSamples)}\n\n')


    pre_GoodSamples, pre_malSamples = np.array(pre_GoodSamples), np.array(pre_malSamples)
    
    if pre_GoodSamples.ndim != pre_malSamples.ndim:

        #print(f'pre_malSamples {pre_malSamples.shape} {pre_malSamples}')
        #print(f'pre_GoodSamples {pre_GoodSamples.shape} {pre_GoodSamples}')
        if pre_GoodSamples.ndim > pre_malSamples.ndim:
            pre_GoodSamples = np.squeeze(pre_GoodSamples)
        elif pre_GoodSamples.ndim < pre_malSamples.ndim:
            pre_malSamples = np.squeeze(pre_malSamples)
            
    samples_to_replay = np.concatenate((pre_GoodSamples, pre_malSamples))
    labels_to_replay = np.concatenate((np.zeros(len(pre_GoodSamples)), np.ones(len(pre_malSamples))))
    
    X_replay, Y_replay = shuffle(samples_to_replay, labels_to_replay)
    
    return X_replay, Y_replay



parser = argparse.ArgumentParser()
parser.add_argument('--num_exps', type=int, default=2, required=False, help='Number of Experiments to Run.')
parser.add_argument('--contamination', type=float, default=0.1, required=False)
parser.add_argument('--num_epoch', type=int, default=200, required=False)
parser.add_argument('--batch_size', type=int, default=2000, required=False)
parser.add_argument('--memory_budget', type=int, required=True)
parser.add_argument('--min_samples', type=int, default=1, required=False)
parser.add_argument('--ifs_option', type=str,\
                    required=True, choices=['ratio', 'uniform', 'gifs', 'mix'])
parser.add_argument('--goodware_ifs', action="store_true", required=False)
parser.add_argument('--data_dir', type=str,\
                    default='/home/msrahman3/IQSeC-Datasets/EMBER-Domain-Family-labels/', required=False)

args = parser.parse_args()


all_task_months = ['2018-01', '2018-02', '2018-03', '2018-04', '2018-05', '2018-06',
                   '2018-07', '2018-08', '2018-09', '2018-10', '2018-11', '2018-12']

# data_dir = '../../month_based_processing_with_family_labels/'

patience = 5
replay_type = ifs_option = args.ifs_option
data_dir = args.data_dir
num_exps = args.num_exps
num_epoch = args.num_epoch
batch_size = args.batch_size
memory_budget = args.memory_budget
min_samples = args.min_samples


madarEL2N = True

contamination = args.contamination #0.1 #[0.2, 0.3, 0.4, 0.5]

exp_seeds = [random.randint(1, 99999) for i in range(num_exps)]


expSaveDir = '../../IFS_Final_' + str(contamination) + '_'
resSaveDir = '../../IFS_Results'
expSaveFile = '/IFS_'  + str(contamination) + '_'

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


        model_save_dir = '../../../../IFS_SavedModel' + '/IFSModel_' + str(memory_budget) + '/' + str(current_task) + '/'
        create_parent_folder(model_save_dir)
        
        opt_save_path = '../../../../IFSS_SavedModel' + '/IFSOpt_' + str(memory_budget) + '/' + str(current_task) + '/'
        create_parent_folder(opt_save_path)
        
        # results_save_dir =  '../../IFS_SavedResults_' +'/IFS_' + str(memory_budget) + '/' 
        #create_parent_folder(results_save_dir)
        
        
        X_train, Y_train, Y_train_family = get_family_labeled_month_data(data_dir, current_task)
        X_test, Y_test, Y_test_family = get_family_labeled_task_test_data(data_dir, task_months, mlp_net=True)
        

        if current_task == all_task_months[0]:
            GFamilyDict, malware_count = GetFamilyDict(X_train, Y_train, Y_train_family,\
                                                   current_task, malware_count, GFamilyDict)
            num_Y_replay = 0
        else:
            if not madarEL2N:
                X_replay, Y_replay = IFS(model, GFamilyDict, memory_budget, goodware_ifs=args.goodware_ifs,\
                                         min_samples=min_samples, fs_option=ifs_option)
                num_Y_replay = len(Y_replay)
                
                GFamilyDict, malware_count = GetFamilyDict(X_train, Y_train, Y_train_family,\
                                               current_task, malware_count, GFamilyDict)
                
                
                X_train, Y_train = np.concatenate((X_train, X_replay)), np.concatenate((Y_train, Y_replay))
            else:
                X_replay, Y_replay = IFS(model, GFamilyDict, memory_budget, goodware_ifs=args.goodware_ifs,\
                                         min_samples=min_samples, fs_option=ifs_option, madarEL2N=True)
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
        
        
        
        model = Ember_MLP_Net()
        model = model.to(device)
        #model = torch.nn.DataParallel(model)
        #load the best model for this task
        best_model_path = model_save_dir + os.listdir(model_save_dir)[0]
        print(f'loading best model {best_model_path}')
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        
        
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.000001)
        best_optimizer = opt_save_path + os.listdir(opt_save_path)[0]
        print(f'loading best optimizer {best_optimizer}')
        optimizer.load_state_dict(torch.load(best_optimizer, weights_only=True))
        
        
        acc, precision, recall, f1score = testing_aucscore(model, X_test, Y_test, batch_size, device)
        
        
        end_time = time.time()

        print(f'Elapsed time {(end_time - start_time)/60} mins.')    
        
        task_end = time.time()
        task_run_time = (task_end - task_start)/60

        if madarEL2N:
            results_save_dir = './Results_Domain_BOSS/'
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
                results_f = open(os.path.join(results_save_dir + 'ifs_boss_' + str(ifs_option) + '_' + str(memory_budget) + '_results.txt'), 'a')
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
