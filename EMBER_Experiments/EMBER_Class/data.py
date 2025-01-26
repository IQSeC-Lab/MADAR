import copy
import numpy as np
from sklearn.utils import shuffle
from torchvision import datasets, transforms
from torch.utils.data import ConcatDataset, Dataset
import torch
from sklearn.preprocessing import StandardScaler



def get_continual_ember_class_data(data_dir, train=True):
    
    if train:
        data_dir = data_dir + '/'
        XY_train = np.load(data_dir + 'XY_train.npz')
        X_tr, Y_tr = XY_train['X_train'], XY_train['Y_train']

        return X_tr, Y_tr
    else:
        data_dir = data_dir + '/'
        XY_test = np.load(data_dir + 'XY_test.npz')
        X_test, Y_test = XY_test['X_test'], XY_test['Y_test']

        return X_test, Y_test 



def get_selected_classes(target_classes):
    classes_Y = [i for i in range(100)]
    #print(classes_Y)
    np.random.seed(42)
    selected_classes = np.random.choice(classes_Y, target_classes,replace=False)
    #print(selected_classes)
    
    return selected_classes


def get_ember_selected_class_data(data_dir, selected_classes, train=True):
    
    
    if train:
        all_X, all_Y = get_continual_ember_class_data(data_dir, train=True)
    else:
        all_X, all_Y = get_continual_ember_class_data(data_dir, train=False)
    
    X_ = []
    Y_ = []

    for ind, cls in enumerate(selected_classes):
        get_ind_cls = np.where(all_Y == cls)
        cls_X = all_X[get_ind_cls]
        #cls_Y = all_Y[get_ind_cls]

        #assert len(cls_Y) == len(cls_X)

        for j in range(len(cls_X)):
            X_.append(cls_X[j])
            Y_.append(ind)

    X_ = np.float32(np.array(X_))
    Y_ = np.array(Y_, dtype=np.int64)
    X_, Y_ = shuffle(X_, Y_)

    if train:
        
        print(f' Training data X {X_.shape} Y {Y_.shape}')
    else:
        print(f' Test data X {X_.shape} Y {Y_.shape}')
    
    return X_, Y_




class malwareSubDatasetExemplars(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''
    
    
    def __init__(self, original_dataset, orig_length_features, target_length_features, sub_labels, target_transform=None):
        super().__init__()
        #print(target_transform)
        self.dataset = original_dataset
        self.orig_length_features = orig_length_features
        self.target_length_features = target_length_features
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            if hasattr(original_dataset, "targets"):
                if self.dataset.target_transform is None:
                    label = self.dataset.targets[index]
                else:
                    label = self.dataset.target_transform(self.dataset.targets[index])
            else:
                label = self.dataset[index][1]
            if label in sub_labels:
                self.sub_indeces.append(index)
        self.target_transform = target_transform
        

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        
        #self.padded_features = np.zeros(self.target_length_features - self.orig_length_features, dtype=np.float32)
        #sample = np.concatenate((self.dataset[self.sub_indeces[index]],self.padded_features))
        #target = self.origlabels[self.sub_indeces[index]]
        
        sample = self.dataset[self.sub_indeces[index]]
        if self.target_transform:
            target = self.target_transform(sample[1])
            sample = (sample[0], target)
            #print(sample)        
        
        #if self.target_transform:
        #    #print(f'target transforming here ..')
        #    target = self.target_transform(target)
        
        return sample 


class malwareSubDataset(Dataset):
    '''To sub-sample a dataset, taking only those samples with label in [sub_labels].

    After this selection of samples has been made, it is possible to transform the target-labels,
    which can be useful when doing continual learning with fixed number of output units.'''
    
    def __init__(self, original_dataset, sub_labels):
        super().__init__()
        #print(target_transform)
        self.dataset, self.origlabels = original_dataset
        
        self.sub_indeces = []
        for index in range(len(self.dataset)):
            label = self.origlabels[index]
            
            if label in sub_labels:
                self.sub_indeces.append(index)
        

    def __len__(self):
        return len(self.sub_indeces)

    def __getitem__(self, index):
        sample = self.dataset[self.sub_indeces[index]]
        target = self.origlabels[self.sub_indeces[index]]
        
        return (sample, target)
    

    
class ExemplarDataset(Dataset):
    '''Create dataset from list of <np.arrays> with shape (N, C, H, W) (i.e., with N images each).

    The images at the i-th entry of [exemplar_sets] belong to class [i], unless a [target_transform] is specified'''

    def __init__(self, exemplar_sets, target_transform=None):
        super().__init__()
        self.exemplar_sets = exemplar_sets
        self.target_transform = target_transform

    def __len__(self):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            total += len(self.exemplar_sets[class_id])
        return total

    def __getitem__(self, index):
        total = 0
        for class_id in range(len(self.exemplar_sets)):
            exemplars_in_this_class = len(self.exemplar_sets[class_id])
            if index < (total + exemplars_in_this_class):
                class_id_to_return = class_id if self.target_transform is None else self.target_transform(class_id)
                exemplar_id = index - total
                break
            else:
                total += exemplars_in_this_class
        image = torch.from_numpy(self.exemplar_sets[class_id][exemplar_id])
        return (image, class_id_to_return)
    
    
    
def get_malware_multitask_experiment(dataset_name, target_classes, init_classes,\
                                     orig_feats_length, target_feats_length,\
                                     scenario, tasks, data_dir, verbose=False):



    if dataset_name == 'EMBER':
        
        num_class = target_classes
        selected_classes = get_selected_classes(target_classes)
        
        # check for number of tasks
        if tasks > num_class:
            raise ValueError(f"EMBER experiments cannot have more than {num_class} tasks!")
            
        # configurations
        config = DATASET_CONFIGS[dataset_name]
        
        
        if scenario == 'class':
            initial_task_num_classes = init_classes
            if initial_task_num_classes > target_classes:
                raise ValueError(f"Initial Number of Classes cannot be more than {target_classes} classes!")


            left_tasks = tasks - 1 
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
            classes_per_task = int(np.floor(num_class / tasks))
            
            labels_per_task = [list(np.array(range(classes_per_task)) +\
                                classes_per_task * task_id) for task_id in range(tasks)]
        
        #data_dir = '../../../../ember2018/top_class_bases/top_classes_100' 
        x_train, y_train = get_ember_selected_class_data(data_dir, selected_classes, train=True)
        x_test, y_test = get_ember_selected_class_data(data_dir, selected_classes, train=False)


        standardization = StandardScaler()
        standard_scaler = standardization.fit(x_train)
        x_train = standard_scaler.transform(x_train)
        x_test = standard_scaler.transform(x_test)  

        ember_train, ember_test = (x_train, y_train), (x_test, y_test)



        #split them up into sub-tasks
        train_datasets = []
        test_datasets = []
        for labels in labels_per_task:
            #print(scenario)
            #print(f'task labels {labels}')
            train_datasets.append(malwareSubDataset(ember_train, labels))
            test_datasets.append(malwareSubDataset(ember_test, labels))


    # If needed, update number of (total) classes in the config-dictionary
    config['classes'] = 100 #classes_per_task if scenario=='domain' else classes_per_task*tasks

    # Return tuple of train-, validation- and test-dataset, config-dictionary and number of classes per task
    
    #return (int(y_train.shape[0]), (train_datasets, test_datasets), config, classes_per_task)
    
    return (int(y_train.shape[0]), ember_train, ember_test, train_datasets, test_datasets, config, classes_per_task)

    
# specify available transforms.
AVAILABLE_TRANSFORMS = {
    'EMBER': [
        transforms.ToTensor(),
    ],
}



# specify configurations of available data-sets.
DATASET_CONFIGS = {
    'EMBER': {'size': 49, 'channels': 1, 'classes': 100},
}



'''
print('running data.py')
(train_datasets, test_datasets), config, classes_per_task = get_malware_multitask_experiment(
    'splitMNIST', 'drebin', 2492, 2500, scenario='class', tasks=9,
    verbose=True, exception=True,
)
'''
#print(test_datasets)
