import data


def get_param_stamp_from_args(args):
    '''To get param-stamp a bit quicker.'''
    from encoder import Classifier

    scenario = args.scenario

    _, _, _, config, _ = data.get_malware_multitask_experiment(
        dataset_name=args.dataset, scenario=scenario,\
        tasks=args.tasks, data_dir=args.d_dir, verbose=False)



    model = Classifier(
        image_size=config['size'], image_channels=config['channels'], classes=config['classes'],
        fc_layers=args.fc_lay, fc_units=args.fc_units, fc_drop=args.fc_drop, fc_nl=args.fc_nl,
        fc_bn=True if args.fc_bn=="yes" else False, excit_buffer=True if args.xdg and args.gating_prop>0 else False,
    )

    model_name = model.name
    param_stamp = get_param_stamp(args, model_name, verbose=False, replay=False if (args.replay=="none") else True,
                                  replay_model_name=replay_model_name)
    return param_stamp



def get_param_stamp(args, model_name, verbose=True, replay=False, replay_model_name=None):
    '''Based on the input-arguments, produce a "parameter-stamp".'''

    # -for task
    multi_n_stamp = "{n}-{set}".format(n=args.tasks, set=args.scenario) if hasattr(args, "tasks") else ""
    task_stamp = "{exp}{multi_n}".format(exp=args.experiment, multi_n=multi_n_stamp)
    if verbose:
        print(" --> task:          "+task_stamp)

    # -for model
    model_stamp = model_name
    if verbose:
        print(" --> model:         "+model_stamp)

    # -for hyper-parameters
    hyper_stamp = "{i_e}{num}-lr{lr}{lrg}-b{bsz}-{optim}".format(
        i_e="e" if args.iters is None else "i", num=args.epochs if args.iters is None else args.iters, lr=args.lr,
        lrg=("" if args.lr==args.lr_gen else "-lrG{}".format(args.lr_gen)) if hasattr(args, "lr_gen") else "",
        bsz=args.batch, optim=args.optimizer,
    )
    if verbose:
        print(" --> hyper-params:  " + hyper_stamp)


    # -for binary classification loss
    binLoss_stamp = ""
    if hasattr(args, 'bce') and args.bce:
        binLoss_stamp = '--BCE_dist' if (args.bce_distill and args.scenario=="class") else '--BCE'

    # --> combine
    param_stamp = "{}--{}--{}--{}".format(
        task_stamp, model_stamp, hyper_stamp, binLoss_stamp,
        "-s{}".format(args.seed) if not args.seed==0 else "",
    )

    ## Print param-stamp on screen and return
    if verbose:
        print(param_stamp)
    return param_stamp