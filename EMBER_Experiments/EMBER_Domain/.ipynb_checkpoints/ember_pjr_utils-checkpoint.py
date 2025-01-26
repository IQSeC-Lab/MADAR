import numpy as np
import random
import os
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


def get_only_av_class_labeled_samples(all_X, all_Y, Y_Family):
    
    new_X = []
    new_Y = []
    new_Y_family = []
    
    for ind, y in enumerate(all_Y):
        if y == 1:
            if Y_Family[ind] != '':
                new_Y.append(y)
                new_X.append(all_X[ind])
                new_Y_family.append(Y_Family[ind])
        else:
            new_Y.append(y)
            new_X.append(all_X[ind])
            new_Y_family.append(Y_Family[ind])
            
            
    print(len(all_Y==1) == (len(new_Y) + (len(all_Y) - len(new_Y))))  
    print(len(new_X) == len(new_Y))
    
    new_X, new_Y, new_Y_family = np.array(new_X), np.array(new_Y), np.array(new_Y_family)
    return new_X, new_Y, new_Y_family


def getNOAVCLASS_partial_data(X, Y, Y_fam , replay_portion):
    indx = [i for i in range(len(Y))]
    random.shuffle(indx)

    replay_data_size = int(len(indx)*replay_portion)
    replay_index = indx[:replay_data_size]

    X_train = X[replay_index]
    Y_train = Y[replay_index]
    Y_family = Y_fam[replay_index]
    
    return X_train, Y_train, Y_family        
        
def getNOAVCLASS_PJR_random_training_data(data_dir, task_months, replay_portion):
    
    X_tr, Y_tr, Y_tr_fam = get_family_labeled_month_data(data_dir, task_months[-1])
    X_tr, Y_tr, Y_tr_fam = get_only_av_class_labeled_samples(X_tr, Y_tr, Y_tr_fam)
    
    print(f'Current Task month {task_months[-1]} data X {X_tr.shape} Y {Y_tr.shape}')
    for month in task_months[:-1]:
        pre_X_tr, pre_Y_tr, pre_Y_tr_fam = get_family_labeled_month_data(data_dir, month)
        pre_X_tr, pre_Y_tr, pre_Y_tr_fam = get_only_av_class_labeled_samples(pre_X_tr, Y_train, pre_Y_tr_fam)
        
        pre_X_tr, pre_Y_tr, pre_Y_tr_fam = getNOAVCLASS_partial_data(pre_X_tr, pre_Y_tr, pre_Y_tr_fam, replay_portion)
        
        print(f'previous month {month} data X {pre_X_tr.shape} Y {pre_Y_tr.shape} Y_family {pre_Y_tr_fam.shape}')
        
        X_tr, Y_tr, Y_tr_fam  = np.concatenate((X_tr, pre_X_tr)),\
                    np.concatenate((Y_tr, pre_Y_tr)), np.concatenate((Y_tr_fam, pre_Y_tr_fam))

    
    print()
    print(f'X_train {X_tr.shape} Y_train {Y_tr.shape} Y_family {Y_tr_fam.shape}\n')
    
    return X_tr, Y_tr, Y_tr_fam 

        
def get_family_labeled_task_test_data(data_dir, task_months, mlp_net=False):
    
    X_te, Y_te, Y_te_family = get_family_labeled_month_data(data_dir, task_months[-1], train=False)
    
    for month in task_months[:-1]:
        pre_X_te, pre_Y_te, pre_Y_te_family = get_family_labeled_month_data(data_dir, month, train=False)
        X_te, Y_te, Y_te_family = np.concatenate((X_te, pre_X_te)), np.concatenate((Y_te, pre_Y_te)),\
                                np.concatenate((Y_te_family, pre_Y_te_family))
        

    X_test, Y_test, Y_test_family = X_te, Y_te, Y_te_family
    print(f'X_test {X_test.shape} Y_test {Y_test.shape} Y_te_family {Y_te_family.shape}')
    
    return X_test, Y_test, Y_test_family



def get_family_labeled_month_data(data_dir, month, train=True):
    
    if train:
        data_dir = data_dir + str(month) + '/'
        XY_train = np.load(data_dir + 'XY_train.npz')
        X_tr, Y_tr, Y_tr_family = XY_train['X_train'], XY_train['Y_train'], XY_train['Y_family_train']

        print(f'X_train {X_tr.shape} Y_train {Y_tr.shape} Y_tr_family {Y_tr_family.shape}')
        
        return X_tr, Y_tr, Y_tr_family
    else:
        data_dir = data_dir + str(month) + '/'
        XY_test = np.load(data_dir + 'XY_test.npz')
        X_test, Y_test, Y_test_family = XY_test['X_test'], XY_test['Y_test'], XY_test['Y_family_test']

        return X_test, Y_test, Y_test_family

    
    
def get_family_labeled_task_test_data(data_dir, task_months, mlp_net=False):
    
    X_te, Y_te, Y_te_family = get_family_labeled_month_data(data_dir, task_months[-1], train=False)
    
    for month in task_months[:-1]:
        pre_X_te, pre_Y_te, pre_Y_te_family = get_family_labeled_month_data(data_dir, month, train=False)
        X_te, Y_te, Y_te_family = np.concatenate((X_te, pre_X_te)), np.concatenate((Y_te, pre_Y_te)),\
                                np.concatenate((Y_te_family, pre_Y_te_family))
        

    X_test, Y_test, Y_test_family = X_te, Y_te, Y_te_family
    print(f'X_test {X_test.shape} Y_test {Y_test.shape} Y_te_family {Y_te_family.shape}')
    
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