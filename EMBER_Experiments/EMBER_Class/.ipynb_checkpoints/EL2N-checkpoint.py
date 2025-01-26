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


def compute_el2n_scores_v1(model, dataloader, device, num_classes=2):
    el2n_scores = []

    # Dynamically choose loss function based on the number of classes
    if num_classes == 2:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss(reduction='none')  # Multi-class

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # if num_classes == 2:
            #     # Binary classification
            #     outputs = outputs.squeeze(-1)  # Shape [batch_size]
            #     targets = targets.float()  # BCELoss expects float targets
            # else:
                # Multi-class classification
                #targets = targets.long()

            try:
                loss = criterion(outputs, targets)
                el2n_scores.extend(loss.cpu().numpy())
            except RuntimeError as e:
                print(f"Error in batch. Outputs: {outputs.shape}, Targets: {targets.shape}")
                raise e
    
    return np.array(el2n_scores)

def compute_el2n_scores_v2(model, dataloader, device, num_classes=100):
    el2n_scores = []
    labels = []

    # Dynamically choose loss function based on the number of classes
    # if num_classes == 2:
    #     criterion = nn.BCEWithLogitsLoss(reduction='none')
    # else:
    criterion = nn.CrossEntropyLoss(reduction='none')  # Multi-class
    #criterion = F.cross_entropy(reduction='mean')

    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            
            # if num_classes == 2:
            #     # Binary classification
            #     outputs = outputs.squeeze(-1)  # Shape [batch_size]
            #     targets = targets.float()  # BCELoss expects float targets
            # else:
                
                # Multi-class classification
                #targets = targets.long()  # CrossEntropyLoss expects int targets
            print(f'targets {targets}')
            #targets = targets.long()
            #print(f'targets {targets}')
            try:
                print(f'outputs {outputs}')
                loss = criterion(outputs, targets)
                print(f'loss {loss}')
                el2n_scores.extend(loss.cpu().numpy())
                labels.extend(targets.cpu().numpy())  # Store labels
            except RuntimeError as e:
                print(f"Error in batch. Outputs: {outputs.shape}, Targets: {targets.shape}")
                raise e
    
    return np.array(el2n_scores), np.array(labels)

def compute_el2n_scores(model, dataloader, device, num_classes=100):
    el2n_scores = []
    l2_loss = nn.MSELoss(reduction='none')  # Use L2 Loss for EL2N scoring

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = torch.softmax(outputs, dim=1)  # Convert logits to probabilities

            # One-hot encode targets for L2 loss computation
            targets_onehot = F.one_hot(targets, num_classes=num_classes).float()

            # Compute EL2N scores
            el2n_score = torch.sqrt(l2_loss(targets_onehot, outputs).sum(dim=1))
            el2n_scores.extend(el2n_score.cpu().numpy())
    
    return np.array(el2n_scores)


def select_top_samples_v1(data, labels, el2n_scores, budget):
    # Sort indices by EL2N scores in descending order
    top_indices = np.argsort(el2n_scores)[-budget:]

    # Extract data and labels corresponding to the selected indices
    selected_data = data[top_indices]
    selected_labels = labels[top_indices]

    return selected_data, top_indices, selected_labels

def select_top_samples_v2(data, labels, el2n_scores, budget):
    # Detect the number of unique classes
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    # Sort indices by EL2N scores in descending order
    top_indices = np.argsort(el2n_scores)[-budget:]

    # Handle one-class scenario
    if num_classes == 1:
        # Direct selection for one class
        selected_data = data[top_indices]
        selected_labels = labels[top_indices]
        return selected_data, top_indices, selected_labels

    # Multi-class scenario
    selected_data = data[top_indices]
    selected_labels = labels[top_indices]

    return selected_data, top_indices, selected_labels    


def select_top_samples_v3(data, labels, el2n_scores, budget):
    
    num_classes = 1
    # Detect the number of unique classes
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)

    # Sort indices by EL2N scores in descending order
    #top_indices = np.argsort(el2n_scores)[-budget:]

    top_indices = np.argsort(el2n_scores)[:budget]

    # Handle one-class scenario
    if num_classes == 1:
        # Direct selection for one class
        selected_subset = torch.utils.data.Subset(data, top_indices)
        selected_labels = labels[top_indices]
        return selected_subset, top_indices, selected_labels

    # Multi-class scenario
    selected_subset = torch.utils.data.Subset(data, top_indices)
    selected_labels = labels[top_indices]

    return selected_subset, top_indices, selected_labels


def select_top_samples(data, labels, el2n_scores, budget):

    data, labels = np.array(data), np.array(labels)    
    # Detect the number of unique classes
    unique_classes = np.unique(labels)
    num_classes = len(unique_classes)
    #print(f'el2n_scores {el2n_scores}')
    # Sort indices by EL2N scores in descending order
    top_indices = np.argsort(el2n_scores)[-budget:]

    #top_indices = np.argsort(el2n_scores)[:budget]

    #print(top_indices)
    #print(f'data.shape {data.shape}, len(top_indices) {len(top_indices)}, top_indices {top_indices}, budget {budget}')
    
    # Extract the actual data and labels as arrays
    selected_data = [data[i] for i in top_indices]
    selected_labels = [labels[i] for i in top_indices]

    return selected_data, top_indices, selected_labels


# Main function for subset selection based on EL2N
def el2n_subset_selection_v1(model, data, labels, budget, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Prepare dataset and dataloader
    dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), 
                            torch.tensor(labels, dtype=torch.long))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Compute EL2N scores
    el2n_scores, _ = compute_el2n_scores(model, dataloader, device)

    # Select top Beta samples
    selected_data, selected_indices, selected_labels = select_top_samples(data, labels, el2n_scores, budget)

    #print(f"Selected {len(selected_indices)} samples using EL2N.")
    return selected_data, selected_indices, selected_labels

def el2n_subset_selection(model, data, labels, budget, num_classes=100, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    
    # Prepare dataset and dataloader
    # dataset = TensorDataset(torch.tensor(data, dtype=torch.float32), 
    #                         torch.tensor(labels, dtype=torch.long))
    dataset = TensorDataset(torch.tensor(np.array(data), dtype=torch.float32),  # Convert to np.array first
                            torch.tensor(np.array(labels), dtype=torch.long))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Compute EL2N scores
    el2n_scores = compute_el2n_scores(model, dataloader, device, num_classes=num_classes)

    # Select top samples
    selected_subset, selected_indices, selected_labels = select_top_samples(
        data, labels, el2n_scores, budget
    )

    #print(f"Selected {len(selected_subset)} samples using EL2N.")
    #print(f"Number of Classes Detected: {len(np.unique(labels))}")
    
    return selected_subset, selected_indices, selected_labels, el2n_scores
