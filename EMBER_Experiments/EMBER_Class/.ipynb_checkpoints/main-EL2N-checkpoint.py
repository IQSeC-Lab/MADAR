#!/usr/bin/env python3
import argparse
import os
import random
import numpy as np
import time
import torch
from torch import optim
import utils
import pandas as pd
from param_stamp import get_param_stamp, get_param_stamp_from_args
import evaluate
from data import get_malware_multitask_experiment
from encoder import Classifier
import callbacks as cb
from train_diversity_EL2N import train_cl
from continual_learner import ContinualLearner
from exemplars import ExemplarHandler
from replayer import Replayer
from param_values import set_default_values


parser = argparse.ArgumentParser('./main.py', description='Run individual continual learning experiment.')
parser.add_argument('--get-stamp', action='store_true', help='print param-stamp & exit')
parser.add_argument('--seed', type=int, default=0, help='random seed (for each random-module used)')
parser.add_argument('--no-gpus', action='store_false', dest='cuda', help="don't use GPUs")
parser.add_argument('--data-dir', type=str,\
                    default='/home/msrahman3/IQSeC-Datasets/EMBER_Class_Task/top_classes_100',\
                    dest='d_dir', help="default: %(default)s")
parser.add_argument('--plot-dir', type=str, default='./plots', dest='p_dir', help="default: %(default)s")
parser.add_argument('--results_dir', type=str, default='./Results_EL2N', dest='r_dir', help="default: %(default)s")

# expirimental task parameters
task_params = parser.add_argument_group('Task Parameters')
task_params.add_argument('--experiment', type=str, required=False,\
                         default='EMBER_Class', choices=['EMBER_Class', 'EMBER_Task'])
task_params.add_argument('--dataset', type=str, default='EMBER', choices=['EMBER'])

task_params.add_argument('--scenario', type=str, default='class', choices=['task', 'domain', 'class'])
task_params.add_argument('--tasks', type=int, help='number of tasks')
task_params.add_argument('--target_classes', type=int, default=100, required=False, help='number of classes')
task_params.add_argument('--init_classes', type=int, default=50, required=False, help='number of classes for the first task')
task_params.add_argument('--replay_portion', type=float, default=1.0,\
                         required=False, help='percentage of number of stored samples to replay')
task_params.add_argument('--cnt_rate', type=float, default=0.1,\
                         required=False, help='contamination rate for isolation forest')
task_params.add_argument('--num_replay_sample', type=int, default=200,\
                         required=False, help='number of stored samples per class to replay')
task_params.add_argument('--replay_config', default='ifs', type=str,\
                         choices=['grs', 'frs', 'hs', 'ifs', 'aws'],\
                         required=True, help='diversity aware replay configurations')

parser.add_argument('--layer', choices=["fcLayer4", "fc4", "fc4_bn"],  default='act4', required=False)


# specify loss functions to be used
loss_params = parser.add_argument_group('Loss Parameters')
loss_params.add_argument('--bce', action='store_true', help="use binary (instead of multi-class) classication loss")
loss_params.add_argument('--bce-distill', action='store_true', help='distilled loss on previous classes for new'
                                                                    ' examples (only if --bce & --scenario="class")')

# model architecture parameters
model_params = parser.add_argument_group('Model Parameters')
model_params.add_argument('--fc-layers', type=int, default=5, dest='fc_lay', help="# of fully-connected layers")
model_params.add_argument('--fc-units', type=int, metavar="N", help="# of units in first fc-layers")
model_params.add_argument('--fc-drop', type=float, default=0., help="dropout probability for fc-units")
model_params.add_argument('--fc-bn', type=str, default="yes", help="use batch-norm in the fc-layers (no|yes)")
model_params.add_argument('--fc-nl', type=str, default="relu", choices=["relu", "leakyrelu"])
model_params.add_argument('--singlehead', action='store_true', help="for Task-IL: use a 'single-headed' output layer   "
                                                                   " (instead of a 'multi-headed' one)")

# training hyperparameters / initialization
train_params = parser.add_argument_group('Training Parameters')
train_params.add_argument('--exp_name', type=str, help="experiment name")
train_params.add_argument('--logger_file', type=str, help="logger file name")
train_params.add_argument('--iters', type=int, default=10000, help="# batches to optimize solver")
train_params.add_argument('--lr', type=float, default=0.001, help="learning rate") #default=0.01, 
train_params.add_argument('--batch', type=int, default=256, help="batch-size")
train_params.add_argument('--epoch', type=int, default=30, help="number of training epochs")
train_params.add_argument('--optimizer', type=str, choices=['adam', 'adam_reset', 'sgd'], default='sgd')

# "memory replay" parameters
replay_params = parser.add_argument_group('Replay Parameters')
replay_params.add_argument('--replay', type=str, default='offline', choices=['offline', 'none'])
replay_params.add_argument('--memory_budget', type=int, required=False)
parser.add_argument('--grs_joint',  action="store_true", required=False)
parser.add_argument('--ifs_option', type=str, default="ratio",\
                    choices=['ratio', 'uniform', 'gifs', 'mix'])
replay_params.add_argument('--min_samples', type=int, default=1, required=False)

# data storage ('exemplars') parameters
store_params = parser.add_argument_group('Data Storage Parameters')
store_params.add_argument('--icarl', action='store_true', help="bce-distill, use-exemplars & add-exemplars")
store_params.add_argument('--use-exemplars', action='store_true', help="use exemplars for classification")
store_params.add_argument('--add-exemplars', action='store_true', help="add exemplars to current task's training set")
store_params.add_argument('--budget', type=int, default=1000, dest="budget", help="how many samples can be stored?")

# evaluation parameters
eval_params = parser.add_argument_group('Evaluation Parameters')
eval_params.add_argument('--metrics', action='store_true', help="calculate additional metrics (e.g., BWT, forgetting)")

def create_parent_folder(file_path):
    if not os.path.exists(os.path.dirname(file_path)):
        os.makedirs(os.path.dirname(file_path))


def run(args, verbose=True):
    # -if 'BCEdistill' is selected for other than scenario=="class", give error
    if args.bce_distill and not args.scenario=="class":
        raise ValueError("BCE-distill can only be used for class-incremental learning.")
    
    # -create plots- and results-directories if needed
    if not os.path.isdir(args.r_dir):
        os.mkdir(args.r_dir)
        

    scenario = args.scenario

    # If only want param-stamp, get it printed to screen and exit
    if hasattr(args, "get_stamp") and args.get_stamp:
        print(get_param_stamp_from_args(args=args))
        exit()

    # Use cuda?
    cuda = torch.cuda.is_available() and args.cuda
    device = torch.device("cuda" if cuda else "cpu")
    #print("CUDA is {}used".format("" if cuda else "NOT(!!) "))
    if verbose:
        print("CUDA is {}used".format("" if cuda else "NOT(!!) "))
    
    
    args.seed = random.randint(1, 99999)
    # Set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if cuda:
        torch.cuda.manual_seed(args.seed)

    target_classes = args.target_classes
    orig_feats_length, target_feats_length = 2381, 2381
    dataset_name = args.dataset #'EMBER'
    init_classes = args.init_classes

    if verbose:
        print("\nPreparing the data...")
        
    num_training_samples, ember_train, ember_test, train_datasets, test_datasets, config, classes_per_task = get_malware_multitask_experiment(
        dataset_name=dataset_name, target_classes=target_classes,\
        init_classes = init_classes, scenario=scenario, orig_feats_length=orig_feats_length,\
        target_feats_length=target_feats_length, tasks=args.tasks, data_dir=args.d_dir,
        verbose=verbose)
    
    
    #num_training_samples = 303331
    #args.iters = 2000

    model = Classifier(
        image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
        fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
        fc_bn=True if args.fc_bn=="yes" else False, excit_buffer=False,
        binaryCE=args.bce).to(device)
    
#     print(args)
    # Define optimizer (only include parameters that "requires_grad")
    model.optim_list = [{'params': filter(lambda p: p.requires_grad, model.parameters()), 'lr': args.lr}]
    model.optim_type = args.optimizer
    if model.optim_type in ("adam", "adam_reset"):
        model.optimizer = optim.Adam(model.optim_list, betas=(0.9, 0.999))
    elif model.optim_type=="sgd":
        model.optimizer = optim.SGD(model.optim_list, momentum=0.9, weight_decay=0.000001)
    else:
        raise ValueError("Unrecognized optimizer, '{}' is not currently a valid option".format(args.optimizer))
    
    
    
    
    #print(model)
    
    
    #-------------------------------------------------------------------------------------------------#

    #----------------------------------#
    #----- CL-STRATEGY: EXEMPLARS -----#
    #----------------------------------#
   
    
    # Store in model whether, how many and in what way to store exemplars
    if isinstance(model, ExemplarHandler) and (args.use_exemplars or args.add_exemplars or args.replay=="exemplars"):
        model.memory_budget = args.budget
        model.norm_exemplars = args.norm_exemplars
        model.herding = args.herding



    #-------------------------------------------------------------------------------------------------#

    #-------------------------------#
    #----- CL-STRATEGY: REPLAY -----#
    #-------------------------------#

    generator = None


    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- REPORTING -----#
    #---------------------#
    #print(model)
    # Get parameter-stamp (and print on screen)
    if verbose:
        print("\nParameter-stamp...")
    param_stamp = get_param_stamp(
        args, 'ember_MLP', verbose=verbose, replay=True if (not args.replay=="none") else False,
        replay_model_name=generator.name if (args.replay=="generative" and not args.feedback) else None,
    )

    # Print some model-characteristics on the screen
    if verbose:
        # -main model
        utils.print_model_info(model, title="MAIN MODEL")


    # Prepare for keeping track of statistics required for metrics (also used for plotting in pdf)
    if args.metrics:
        # -define [metrics_dict] to keep track of performance during training for storing & for later plotting in pdf
        metrics_dict = evaluate.initiate_metrics_dict(args, n_tasks=args.tasks, scenario=args.scenario)
        # -evaluate randomly initiated model on all tasks & store accuracies in [metrics_dict] (for calculating metrics)
        if not args.use_exemplars:
            metrics_dict = evaluate.intial_accuracy(args, model, test_datasets, metrics_dict,
                                                    classes_per_task=classes_per_task, scenario=scenario,
                                                    test_size=None, no_task_mask=False)
    else:
        metrics_dict = None
    
    #print(metrics_dict)
    
    #-------------------------------------------------------------------------------------------------#

    #---------------------#
    #----- CALLBACKS -----#
    #---------------------#

    # Callbacks for reporting on and visualizing loss
    
    solver_loss_cbs = [None]



    # Callbacks for reporting and visualizing accuracy
    # -visdom (i.e., after each [prec_log]
    eval_cbs =  [None]
    
    metric_cbs = [
        cb._metric_cb(args=args, log=args.iters, test_datasets=test_datasets,
                      classes_per_task=classes_per_task, metrics_dict=metrics_dict, scenario=scenario,
                      iters_per_task=args.iters, with_exemplars=args.use_exemplars),
        cb._eval_cb(log=args.iters, test_datasets=test_datasets,
                    iters_per_task=args.iters, test_size=args.prec_n, classes_per_task=classes_per_task,
                    scenario=scenario, with_exemplars=True) if args.use_exemplars else None
    ]

    #--------------------#
    #----- TRAINING -----#
    #--------------------#

    if verbose:
        print("\nTraining...")
    model_save_path = './ember_saved_models/'
    # Keep track of training-time
    start = time.time()
    
    #print(args) 
    results_save_dir = './Results_Class_EL2N/'
    create_parent_folder(results_save_dir)
    
    if args.replay_config == 'grs':
        if args.grs_joint:
            results_file = str(args.replay_config)  + '_joint' + '_results.txt'
        else:
            results_file = str(args.replay_config)  + '_' + str(args.memory_budget) + '_results.txt'
    
    if args.replay_config == 'frs':
        results_file = str(args.replay_config)  + '_' + str(args.num_replay_sample) + '_results.txt'
        
            
    if args.replay_config == 'ifs':
        if args.min_samples > 1:
            results_file = str(args.replay_config)  + '_boss_' + str(args.ifs_option) + '_' + str(args.memory_budget) + '_' + str(args.min_samples) + '_results.txt'
        else:
            results_file = str(args.replay_config)  + '_boss_' + str(args.ifs_option) + '_' + str(args.memory_budget) + '_results.txt'
    
    if args.replay_config == 'aws':
        if args.min_samples > 1:
            results_file = str(args.replay_config)  + '_' + str(args.ifs_option) + '_' + str(args.memory_budget) + '_' + str(args.min_samples) + '_results.txt'
        else:
            results_file = str(args.replay_config)  + '_' + str(args.ifs_option) + '_' + str(args.memory_budget) + '_results.txt'
    
#         results_file = str(args.replay_config) + '_' + str(args.ifs_option) + '_' + str(args.layer)  + '_' + str(args.memory_budget) + '_results.txt'
    
    
   
    args.r_dir = os.path.join(results_save_dir + results_file)
    
   
    # Train model
    train_cl(
        model, model_save_path, ember_train, ember_test, train_datasets,\
        test_datasets, args, replay_mode=args.replay, scenario=scenario,\
        classes_per_task=classes_per_task,\
        iters=args.iters, batch_size=args.batch,\
        eval_cbs=eval_cbs, loss_cbs=solver_loss_cbs,\
        metric_cbs=metric_cbs, use_exemplars=args.use_exemplars, add_exemplars=args.add_exemplars,
    )
    

          
if __name__ == '__main__':
    # -load input-arguments
    args = parser.parse_args()
    # -set default-values for certain arguments based on chosen scenario & experiment
    args = set_default_values(args)
    # -run experiment
    run(args, verbose=True)
