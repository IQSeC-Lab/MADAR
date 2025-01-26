# import numpy as np
# import torch
# from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score, accuracy_score
# import utils

# ####--------------------------------------------------------------------------------------------------------------####

# ####-----------------------------####
# ####----CLASSIFIER EVALUATION----####
# ####-----------------------------####


# def validate(model, dataset, batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
#              with_exemplars=False, no_task_mask=False, task=None):
#     '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

#     [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
#                             (these "active classes" are assumed to be contiguous)'''

#     # Set model to eval()-mode
#     mode = model.training
#     model.eval()

#     # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
#     if hasattr(model, "mask_dict") and model.mask_dict is not None:
#         if no_task_mask:
#             model.reset_XdGmask()
#         else:
#             model.apply_XdGmask(task=task)

#     # Loop over batches in [dataset]
#     data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda())
#     #total_tested = total_correct = 0
    
#     correct_labels = []
#     predicted_labels = []
#     #y_predicts_scores = []
    
    
#     batch_acc, batch_prec, batch_recall, batch_f1 = [], [], [], []
#     for data, labels in data_loader:
#         # -break on [test_size] (if "None", full dataset is used)
#         if test_size:
#             if total_tested >= test_size:
#                 break
#         # -evaluate model (if requested, only on [allowed_classes])
#         data, labels = data.to(model._device()), labels.to(model._device())
#         labels = labels - allowed_classes[0] if (allowed_classes is not None) else labels
#         #print(labels)
#         with torch.no_grad():
#             if with_exemplars:
#                 predicted = model.classify_with_exemplars(data, allowed_classes=allowed_classes)
#                 # - in case of Domain-IL scenario, collapse all corresponding domains into same class
#                 if max(predicted).item() >= model.classes:
#                     predicted = predicted % model.classes
#             else:
#                 scores = model(data) if (allowed_classes is None) else model(data)[:, allowed_classes]
#                 #print(scores)
                
#                 _, predicted = torch.max(scores, 1)
                
#                 #y_predicts_scores += list(predicted.detach().cpu().numpy())
                
                
                
#             # -update statistics
#             labels_batch = labels.cpu().numpy()
#             predicted_labels_batch = predicted.cpu().numpy()

# #             batch_acc.append(accuracy_score(labels, predicted_labels))
# #             batch_prec.append(precision_score(labels, predicted_labels, average='micro'))
# #             batch_recall.append(recall_score(labels, predicted_labels, average='micro'))
# #             batch_f1.append(f1_score(labels, predicted_labels, average='macro'))
        
# #         total_correct += (predicted == labels).sum().item()
# #         total_tested += len(data)
        
#         correct_labels += list(labels_batch)
#         predicted_labels += list(predicted_labels_batch)
        
#     #accuracy = total_correct / total_tested
#     #y_predicts_scores = np.array(y_predicts_scores)
#     #correct_labels = np.array(correct_labels)
    
#     #predicted_labels = np.array(predicted_labels)
    
#     accuracy = accuracy_score(correct_labels, predicted_labels)
#     precision = precision_score(correct_labels, predicted_labels, average='micro')
#     recall = recall_score(correct_labels, predicted_labels, average='micro')
#     f1score = f1_score(correct_labels, predicted_labels, average='macro')
    
    
# #     accuracy = np.mean(batch_acc)
# #     precision = np.mean(batch_prec)
# #     recall = np.mean(batch_recall)
# #     f1score = np.mean(batch_f1)
    

#     # Set model back to its initial mode, print result on screen (if requested) and return it
#     model.train(mode=mode)
#     if verbose:
#          print('=> Acc: {:.3f} Precision {:.3f} Recall {:.3f} F1 {:.3f} '.format(accuracy, precision, recall, f1score))
            
#     return accuracy, precision, recall, f1score


# def V1_precision(model, datasets, current_task, iteration, classes_per_task=None, scenario="domain",
#               test_size=None, visdom=None, verbose=False, summary_graph=True, with_exemplars=False, no_task_mask=False):
#     '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

#     [classes_per_task]  <int> number of active classes er task
#     [scenario]          <str> how to decide which classes to include during evaluating precision
#     [visdom]            None or <dict> with name of "graph" and "env" (if None, no visdom-plots are made)'''

#     n_tasks = len(datasets)

#     # Evaluate accuracy of model predictions for all tasks so far (reporting "0" for future tasks)
#     precs = []
#     for i in range(n_tasks):
#         if i+1 <= current_task:
#             if scenario=='domain':
#                 allowed_classes = None
#             elif scenario=='task':
#                 allowed_classes = list(range(classes_per_task*i, classes_per_task*(i+1)))
#             elif scenario=='class':
#                 allowed_classes = list(range(classes_per_task*current_task))
                
#             precs.append(validate(model, datasets[i], test_size=test_size, verbose=verbose,
#                                   allowed_classes=allowed_classes, with_exemplars=with_exemplars,
#                                   no_task_mask=no_task_mask, task=i+1))
#         else:
#             precs.append(0)
#     average_precs = sum([precs[task_id] for task_id in range(current_task)]) / current_task

#     # Print results on screen
#     if verbose:
#         print(' => ave precision: {:.3f}'.format(average_precs))

#     # Send results to visdom server
#     names = ['task {}'.format(i + 1) for i in range(n_tasks)]
#     if visdom is not None:
#         visual_visdom.visualize_scalars(
#             precs, names=names, title="precision ({})".format(visdom["graph"]),
#             iteration=iteration, env=visdom["env"], ylabel="test precision"
#         )
#         if n_tasks>1 and summary_graph:
#             visual_visdom.visualize_scalars(
#                 [average_precs], names=["ave"], title="ave precision ({})".format(visdom["graph"]),
#                 iteration=iteration, env=visdom["env"], ylabel="test precision"
#             )



# ####--------------------------------------------------------------------------------------------------------------####

# ####---------------------------####
# ####----METRIC CALCULATIONS----####
# ####---------------------------####


# def initiate_metrics_dict(args, n_tasks, scenario):
#     '''Initiate <dict> with all measures to keep track of.'''
#     metrics_dict = {}
#     metrics_dict["average"] = []     # ave acc over all tasks so far: Task-IL -> only classes in task
#                                      #                                Class-IL-> all classes so far (up to trained task)
#     metrics_dict["x_iteration"] = [] # total number of iterations so far
#     metrics_dict["x_task"] = []      # number of tasks so far (indicating the task on which training just finished)
#     # Accuracy matrix
#     if not scenario=="class":
#         # -in the domain-incremetnal learning scenario, each task has the same classes
#         # -in the task-incremental learning scenario, only the classes within each task are considered
#         metrics_dict["acc per task"] = {}
#         for i in range(n_tasks):
#             metrics_dict["acc per task"]["task {}".format(i+1)] = []
#     else:
#         # -in the class-incremental learning scenario, accuracy matrix can be defined in different ways
#         metrics_dict["acc per task (only classes in task)"] = {}
#         metrics_dict["acc per task (all classes up to trained task)"] = {}
#         metrics_dict["acc per task (all classes up to evaluated task)"] = {}
#         metrics_dict["acc per task (all classes)"] = {}
#         for i in range(n_tasks):
#             metrics_dict["acc per task (only classes in task)"]["task {}".format(i+1)] = []
#             metrics_dict["acc per task (all classes up to trained task)"]["task {}".format(i + 1)] = []
#             metrics_dict["acc per task (all classes up to evaluated task)"]["task {}".format(i + 1)] = []
#             metrics_dict["acc per task (all classes)"]["task {}".format(i + 1)] = []
#     return metrics_dict


# def intial_accuracy(args, model, datasets, metrics_dict, classes_per_task=None, scenario="domain", test_size=None,
#                     verbose=False, no_task_mask=False):
#     '''Evaluate precision of a classifier (=[model]) on all tasks using [datasets] before any learning.'''

#     n_tasks = len(datasets)

#     if not scenario=="class":
#         precs = []
#     else:
#         precs_all_classes = []
#         precs_only_classes_in_task = []
#         precs_all_classes_upto_task = []

#     for i in range(n_tasks):
#         #print(f'evaluating on task {i}')
#         if not scenario=="class":
#             precision, _, _, _ = validate(
#                 model, datasets[i], test_size=test_size, verbose=verbose,
#                 allowed_classes=None if scenario=="domain" else list(range(classes_per_task*i, classes_per_task*(i+1))),
#                 no_task_mask=no_task_mask, task=i+1
#             )
#             precs.append(precision)
#         else:
#             # -all classes
#             precision, _, _, _ = validate(model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=None,
#                                  no_task_mask=no_task_mask, task=i + 1)
#             precs_all_classes.append(precision)
            
#             # -only classes in task
#             #allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))

            
            
#             init_classes = args.init_classes
#             if i == 0:
#                 allowed_classes = list(range(init_classes))
#             else:
#                 allowed_classes = list(range(init_classes + classes_per_task * (i-1), init_classes + classes_per_task * (i)))
#             #print(f'allowed_classes {allowed_classes}')
            
#             precision, _, _, _ = validate(model, datasets[i], test_size=test_size, verbose=verbose,
#                                  allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1)
#             precs_only_classes_in_task.append(precision)
            
#             # -classes up to evaluated task
#             #allowed_classes = list(range(classes_per_task * (i + 1)))
#             if i == 0:
#                 allowed_classes = list(range(init_classes)) #list([0, 1, 2, 3, 4])
#             else:
#                 allowed_classes = list(range(init_classes + classes_per_task * (i - 1)))
                
#             precision, _, _, _ = validate(model, datasets[i], test_size=test_size, verbose=verbose,
#                                  allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1)
#             precs_all_classes_upto_task.append(precision)

#     if not scenario=="class":
#         metrics_dict["initial acc per task"] = precs
#     else:
#         metrics_dict["initial acc per task (all classes)"] = precs_all_classes
#         metrics_dict["initial acc per task (only classes in task)"] = precs_only_classes_in_task
#         metrics_dict["initial acc per task (all classes up to evaluated task)"] = precs_all_classes_upto_task
#     return metrics_dict


# def metric_statistics(args, model, datasets, current_task, iteration, classes_per_task=None, scenario="domain",
#                       metrics_dict=None, test_size=None, verbose=False, with_exemplars=False, no_task_mask=False):
#     '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

#     [metrics_dict]      None or <dict> of all measures to keep track of, to which results will be appended to
#     [classes_per_task]  <int> number of active classes er task
#     [scenario]          <str> how to decide which classes to include during evaluating precision'''

#     n_tasks = len(datasets)

#     # Calculate accurcies per task, possibly in various ways (if Class-IL scenario)
#     precs_all_classes = []
#     precs_all_classes_so_far = []
#     precs_only_classes_in_task = []
#     precs_all_classes_upto_task = []
    
#     init_classes = args.init_classes
    
#     for i in range(n_tasks):
#         # -all classes
#         if scenario in ('domain', 'class'):
#             precision, _, _, _ = validate(
#                 model, datasets[i], test_size=test_size, verbose=verbose, allowed_classes=None,
#                 no_task_mask=no_task_mask, task=i + 1, with_exemplars=with_exemplars
#             ) if (not with_exemplars) or (i<current_task) else 0.
#             precs_all_classes.append(precision)
            
#         # -all classes up to trained task
#         if scenario in ('class'):
            
#             #allowed_classes = list(range(classes_per_task * current_task))
            
            
            
            
#             if current_task == 0:
#                 allowed_classes = list(range(init_classes))
#             else:
#                 allowed_classes = list(range(init_classes + classes_per_task * (current_task-1)))
#             print(f'allowed_classes in evaluate - all classes up to trained task\n {allowed_classes}')
            
#             precision, _, _, _ = validate(model, datasets[i], test_size=test_size, verbose=verbose,
#                                  allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
#                                  with_exemplars=with_exemplars) if (i<current_task) else 0.
#             precs_all_classes_so_far.append(precision)
            
#         # -all classes up to evaluated task
#         if scenario in ('class'):
#             if i == 0:
#                 allowed_classes = list(range(init_classes))
#             else:
#                 allowed_classes = list(range(init_classes + classes_per_task * i))
                
#             precision, _, _, _ = validate(model, datasets[i], test_size=test_size, verbose=verbose,
#                                  allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
#                                  with_exemplars=with_exemplars) if (not with_exemplars) or (i<current_task) else 0.
#             precs_all_classes_upto_task.append(precision)
            
#         # -only classes in that task
#         if scenario in ('task', 'class'):
            
#             if scenario == 'class':
#                 if i == 0:
#                     allowed_classes = list(range(init_classes))
#                 else:
#                     allowed_classes = list(range(init_classes + classes_per_task * (i-1), init_classes + classes_per_task * (i)))
            
#             else:
#                 allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
                
                
#             precision, _, _, _ = validate(model, datasets[i], test_size=test_size, verbose=verbose,
#                                  allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
#                                  with_exemplars=with_exemplars) if (not with_exemplars) or (i<current_task) else 0.
#             precs_only_classes_in_task.append(precision)

#     # Calcualte average accuracy over all tasks thus far
#     if scenario=='task':
#         average_precs = sum([precs_only_classes_in_task[task_id] for task_id in range(current_task)]) / current_task
#     elif scenario=='domain':
#         average_precs = sum([precs_all_classes[task_id] for task_id in range(current_task)]) / current_task
#     elif scenario=='class':
#         average_precs = sum([precs_all_classes_so_far[task_id] for task_id in range(current_task)]) / current_task

#     # Append results to [metrics_dict]-dictionary
#     for task_id in range(n_tasks):
#         if scenario=="task":
#             metrics_dict["acc per task"]["task {}".format(task_id+1)].append(precs_only_classes_in_task[task_id])
#         elif scenario=="domain":
#             metrics_dict["acc per task"]["task {}".format(task_id+1)].append(precs_all_classes[task_id])
#         else:
#             metrics_dict["acc per task (all classes)"]["task {}".format(task_id+1)].append(precs_all_classes[task_id])
#             metrics_dict["acc per task (all classes up to trained task)"]["task {}".format(task_id+1)].append(
#                 precs_all_classes_so_far[task_id]
#             )
#             metrics_dict["acc per task (all classes up to evaluated task)"]["task {}".format(task_id+1)].append(
#                 precs_all_classes_upto_task[task_id]
#             )
#             metrics_dict["acc per task (only classes in task)"]["task {}".format(task_id+1)].append(
#                 precs_only_classes_in_task[task_id]
#             )
#     metrics_dict["average"].append(average_precs)
#     metrics_dict["x_iteration"].append(iteration)
#     metrics_dict["x_task"].append(current_task)

#     # Print results on screen
#     if verbose:
#         print(' => ave precision: {:.5f}'.format(average_precs))

#     return metrics_dict




import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.metrics import precision_score, recall_score, accuracy_score
import utils


####--------------------------------------------------------------------------------------------------------------####

####-----------------------------####
####----CLASSIFIER EVALUATION----####
####-----------------------------####


def validate(model, dataset, whichMetric='accuracy', batch_size=128, test_size=1024, verbose=True, allowed_classes=None,
             with_exemplars=False, no_task_mask=False, task=None):
    '''Evaluate precision (= accuracy or proportion correct) of a classifier ([model]) on [dataset].

    [allowed_classes]   None or <list> containing all "active classes" between which should be chosen
                            (these "active classes" are assumed to be contiguous)'''

    # Set model to eval()-mode
    mode = model.training
    model.eval()

    # Apply task-specifc "gating-mask" for each hidden fully connected layer (or remove it!)
    if hasattr(model, "mask_dict") and model.mask_dict is not None:
        if no_task_mask:
            model.reset_XdGmask()
        else:
            model.apply_XdGmask(task=task)

    # Loop over batches in [dataset]
    data_loader = utils.get_data_loader(dataset, batch_size, cuda=model._is_on_cuda())
    total_tested = total_correct = 0
    
    correct_labels = []
    predicted_labels = []
    y_predicts_scores = []
    normalized_scores = []
    
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
            if with_exemplars:
                predicted = model.classify_with_exemplars(data, allowed_classes=allowed_classes)
                # - in case of Domain-IL scenario, collapse all corresponding domains into same class
                if max(predicted).item() >= model.classes:
                    predicted = predicted % model.classes
            else:
                scores = model(data) if (allowed_classes is None) else model(data)[:, allowed_classes]
                #print(scores)
                _, predicted = torch.max(scores, 1)
                
                y_predicts_scores += list(predicted.detach().cpu().numpy())
                      
                
        # -update statistics
        total_correct += (predicted == labels).sum().item()
        total_tested += len(data)
        
        correct_labels += list(labels.cpu().numpy())
        predicted_labels += list(predicted.cpu().numpy())
        
    #precision = total_correct / total_tested
    #y_predicts_scores = np.array(y_predicts_scores)
    #correct_labels = np.array(correct_labels)
    
    #print(len(correct_labels), np.unique(correct_labels), len(y_predicts_scores))
    #normalized_scores = np.array(normalized_scores, dtype=np.float32)
    
    accuracy = accuracy_score(correct_labels, predicted_labels)
    precision = precision_score(correct_labels, predicted_labels, average='micro')
    recall = recall_score(correct_labels, predicted_labels, average='micro')
    f1score = f1_score(correct_labels, predicted_labels, average='macro')

    
    # Set model back to its initial mode, print result on screen (if requested) and return it
    model.train(mode=mode)
    
#     if verbose:
#          print('=> Acc: {:.3f} Precision {:.3f} Recall {:.3f} F1 {:.3f} '.format(accuracy, precision, recall, f1score))
            
    if whichMetric == 'accuracy': 
        if verbose:
            print('=> Acc: {:.3f} Precision {:.3f} Recall {:.3f} F1 {:.3f} '.format(accuracy, precision, recall, f1score))
            
        return accuracy
    if whichMetric == 'precision':        
        return precision
    if whichMetric == 'recall':        
        return recall
    if whichMetric == 'f1score':        
        return f1score




####--------------------------------------------------------------------------------------------------------------####

####---------------------------####
####----METRIC CALCULATIONS----####
####---------------------------####


def initiate_metrics_dict(args, n_tasks, scenario):
    '''Initiate <dict> with all measures to keep track of.'''
    metrics_dict = {}
    metrics_dict["average"] = []     # ave acc over all tasks so far: Task-IL -> only classes in task
                                     #                                Class-IL-> all classes so far (up to trained task)
    metrics_dict["x_iteration"] = [] # total number of iterations so far
    metrics_dict["x_task"] = []      # number of tasks so far (indicating the task on which training just finished)
    # Accuracy matrix
    if not scenario=="class":
        # -in the domain-incremetnal learning scenario, each task has the same classes
        # -in the task-incremental learning scenario, only the classes within each task are considered
        metrics_dict["acc per task"] = {}
        for i in range(n_tasks):
            metrics_dict["acc per task"]["task {}".format(i+1)] = []
    else:
        # -in the class-incremental learning scenario, accuracy matrix can be defined in different ways
        metrics_dict["acc per task (only classes in task)"] = {}
        metrics_dict["acc per task (all classes up to trained task)"] = {}
        metrics_dict["acc per task (all classes up to evaluated task)"] = {}
        metrics_dict["acc per task (all classes)"] = {}
        for i in range(n_tasks):
            metrics_dict["acc per task (only classes in task)"]["task {}".format(i+1)] = []
            metrics_dict["acc per task (all classes up to trained task)"]["task {}".format(i + 1)] = []
            metrics_dict["acc per task (all classes up to evaluated task)"]["task {}".format(i + 1)] = []
            metrics_dict["acc per task (all classes)"]["task {}".format(i + 1)] = []
    return metrics_dict

def intial_accuracy(args, model, datasets, metrics_dict, classes_per_task=None, scenario="domain", test_size=None,
                    verbose=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks using [datasets] before any learning.'''

    n_tasks = len(datasets)
    init_classes = args.init_classes
    
    if not scenario=="class":
        precs = []
    else:
        precs_all_classes = []
        precs_only_classes_in_task = []
        precs_all_classes_upto_task = []

    for i in range(n_tasks):
        #print(f'evaluating on task {i}')
        if not scenario=="class":
            precision = validate(
                model, datasets[i],whichMetric='precision', test_size=test_size, verbose=verbose,
                allowed_classes=None if scenario=="domain" else list(range(classes_per_task*i, classes_per_task*(i+1))),
                no_task_mask=no_task_mask, task=i+1
            )
            precs.append(precision)
        else:
            # -all classes
            precision = validate(
                                 model, datasets[i], whichMetric='precision',
                                test_size=test_size, verbose=verbose, allowed_classes=None,
                                 no_task_mask=no_task_mask, task=i + 1)
            precs_all_classes.append(precision)
            
            # -only classes in task
            #allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
            
            
            if i == 0:
                allowed_classes = list(range(init_classes))
            else:
                allowed_classes = list(range(init_classes + classes_per_task * (i-1), init_classes + classes_per_task * (i)))
            #print(f'allowed_classes {allowed_classes}')
            
            precision = validate(
                                 model, datasets[i], whichMetric='precision',
                                 test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1)
            precs_only_classes_in_task.append(precision)
            
            # -classes up to evaluated task
            #allowed_classes = list(range(classes_per_task * (i + 1)))
            if i == 0:
                allowed_classes = list(range(init_classes)) #list([0, 1, 2, 3, 4])
            else:
                allowed_classes = list(range(init_classes + classes_per_task * (i - 1)))
                
            precision = validate(
                                 model, datasets[i], whichMetric='precision',
                                 test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1)
            precs_all_classes_upto_task.append(precision)

    if not scenario=="class":
        metrics_dict["initial acc per task"] = precs
    else:
        metrics_dict["initial acc per task (all classes)"] = precs_all_classes
        metrics_dict["initial acc per task (only classes in task)"] = precs_only_classes_in_task
        metrics_dict["initial acc per task (all classes up to evaluated task)"] = precs_all_classes_upto_task
    return metrics_dict


def metric_statistics(args, model, datasets, current_task, iteration, classes_per_task=None, scenario="domain",
                      metrics_dict=None, test_size=None, verbose=False, with_exemplars=False, no_task_mask=False):
    '''Evaluate precision of a classifier (=[model]) on all tasks so far (= up to [current_task]) using [datasets].

    [metrics_dict]      None or <dict> of all measures to keep track of, to which results will be appended to
    [classes_per_task]  <int> number of active classes er task
    [scenario]          <str> how to decide which classes to include during evaluating precision'''

    n_tasks = len(datasets)
    init_classes = args.init_classes
    
    # Calculate accurcies per task, possibly in various ways (if Class-IL scenario)
    precs_all_classes = []
    precs_all_classes_so_far = []
    precs_only_classes_in_task = []
    precs_all_classes_upto_task = []
    for i in range(n_tasks):
        # -all classes
        if scenario in ('domain', 'class'):
            precision = validate(
                model, datasets[i], whichMetric='precision', 
                test_size=test_size, verbose=verbose, allowed_classes=None,
                no_task_mask=no_task_mask, task=i + 1, with_exemplars=with_exemplars
            ) if (not with_exemplars) or (i<current_task) else 0.
            precs_all_classes.append(precision)
        # -all classes up to trained task
        if scenario in ('class'):
            
            #allowed_classes = list(range(classes_per_task * current_task))
            
            #tmp_task = current_task + 1
            #allowed_classes = list(range(classes_per_task * tmp_task))
            
            if current_task == 0:
                allowed_classes = list(range(init_classes))
            else:
                allowed_classes = list(range(init_classes + classes_per_task * (current_task-1)))
                
                
            precision = validate(
                                 model, datasets[i], whichMetric='precision', 
                                 test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
                                 with_exemplars=with_exemplars) if (i<current_task) else 0.
            precs_all_classes_so_far.append(precision)
            
        # -all classes up to evaluated task
        if scenario in ('class'):
            if i == 0:
                allowed_classes = list(range(init_classes))
            else:
                allowed_classes = list(range(init_classes + classes_per_task * i))
                
            #allowed_classes = list(range(classes_per_task * (i+1)))
            
            precision = validate(
                                 model, datasets[i], whichMetric='precision', 
                                 test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
                                 with_exemplars=with_exemplars) if (not with_exemplars) or (i<current_task) else 0.
            precs_all_classes_upto_task.append(precision)
        # -only classes in that task
        if scenario in ('task', 'class'):
            if scenario == 'class':
                if i == 0:
                    allowed_classes = list(range(init_classes))
                else:
                    allowed_classes = list(range(init_classes + classes_per_task * (i-1), init_classes + classes_per_task * (i)))
            
            else:
                allowed_classes = list(range(classes_per_task * i, classes_per_task * (i + 1)))
                
            
            precision = validate(
                                 model, datasets[i], whichMetric='precision',
                                 test_size=test_size, verbose=verbose,
                                 allowed_classes=allowed_classes, no_task_mask=no_task_mask, task=i + 1,
                                 with_exemplars=with_exemplars) if (not with_exemplars) or (i<current_task) else 0.
            precs_only_classes_in_task.append(precision)

    # Calcualte average accuracy over all tasks thus far
    if scenario=='task':
        average_precs = sum([precs_only_classes_in_task[task_id] for task_id in range(current_task)]) / current_task
    elif scenario=='domain':
        average_precs = sum([precs_all_classes[task_id] for task_id in range(current_task)]) / current_task
    elif scenario=='class':
        average_precs = sum([precs_all_classes_so_far[task_id] for task_id in range(current_task)]) / current_task

    # Append results to [metrics_dict]-dictionary
    for task_id in range(n_tasks):
        if scenario=="task":
            metrics_dict["acc per task"]["task {}".format(task_id+1)].append(precs_only_classes_in_task[task_id])
        elif scenario=="domain":
            metrics_dict["acc per task"]["task {}".format(task_id+1)].append(precs_all_classes[task_id])
        else:
            metrics_dict["acc per task (all classes)"]["task {}".format(task_id+1)].append(precs_all_classes[task_id])
            metrics_dict["acc per task (all classes up to trained task)"]["task {}".format(task_id+1)].append(
                precs_all_classes_so_far[task_id]
            )
            metrics_dict["acc per task (all classes up to evaluated task)"]["task {}".format(task_id+1)].append(
                precs_all_classes_upto_task[task_id]
            )
            metrics_dict["acc per task (only classes in task)"]["task {}".format(task_id+1)].append(
                precs_only_classes_in_task[task_id]
            )
    metrics_dict["average"].append(average_precs)
    metrics_dict["x_iteration"].append(iteration)
    metrics_dict["x_task"].append(current_task)

    # Print results on screen
    if verbose:
        print(' => ave precision: {:.5f}'.format(average_precs))

    return metrics_dict



####--------------------------------------------------------------------------------------------------------------####
