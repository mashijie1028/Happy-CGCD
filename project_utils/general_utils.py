import os
import torch
import random
import numpy as np
import inspect
import argparse
import sys
import time
from datetime import datetime

from loguru import logger


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):

        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):

        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def set_seed(seed, cuda=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_torch(seed=1029):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def strip_state_dict(state_dict, strip_key='module.'):

    """
    Strip 'module' from start of state_dict keys
    Useful if model has been trained as DataParallel model
    """

    for k in list(state_dict.keys()):
        if k.startswith(strip_key):
            state_dict[k[len(strip_key):]] = state_dict[k]
            del state_dict[k]

    return state_dict


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def init_experiment(args, runner_name=None, exp_id=None):
    # Get filepath of calling script
    if runner_name is None:
        runner_name = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))).split(".")[-2:]

    #root_dir = os.path.join(args.exp_root, *runner_name)
    # NOTE!!! add shuffle to path
    if args.shuffle_classes:
        root_dir = os.path.join(args.exp_root, args.dataset_name + '_shuffle_seed_' + str(args.seed))
    else:
        root_dir = os.path.join(args.exp_root, args.dataset_name)

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    # Unique identifier for experiment
    now = str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    if args.base_exp_id is not None:
        now = args.base_exp_id + '_' + now

    log_dir = os.path.join(root_dir, now)
    while os.path.exists(log_dir):
        now = str(time.strftime("%Y%m%d-%H%M%S", time.localtime()))
        if args.base_exp_id is not None:
            now = args.base_exp_id + '_' + now

        log_dir = os.path.join(root_dir, now)

    if exp_id is not None:
        log_dir = log_dir + '_' + exp_id

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger.add(os.path.join(log_dir, 'log.txt'))
    args.logger = logger
    args.log_dir = log_dir

    # Instantiate directory to save models to
    model_root_dir = os.path.join(args.log_dir, 'checkpoints')
    if not os.path.exists(model_root_dir):
        os.mkdir(model_root_dir)

    args.model_dir = model_root_dir
    args.model_path = os.path.join(args.model_dir, 'model.pt')

    print(f'Experiment saved to: {args.log_dir}')
    print(runner_name)

    # print and save args
    print(args)
    save_args_path = os.path.join(log_dir, 'args.txt')
    f_args = open(save_args_path, 'w')
    f_args.write('args: \n')
    f_args.write(str(vars(args)))
    f_args.close()

    return args
