
def set_default_values(args, also_hyper_params=True):
    # -set default-values for certain arguments based on chosen scenario & experiment
    args.tasks= (11 if args.scenario=='class' else 20) if args.tasks is None else args.tasks
    args.iters = 5000 if args.iters is None else args.iters
    args.lr = 0.001 if args.lr is None else args.lr
    args.fc_units = 1024 if args.fc_units is None else args.fc_units

    return args
