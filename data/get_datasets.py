import torch
from tqdm import tqdm
from copy import deepcopy
import pickle
import os

from data.data_utils import MergedDataset, MergedUnlabelledDataset

from data.cifar import get_cifar_10_datasets, get_cifar_100_datasets
from data.tiny_imagenet import get_tiny_imagenet_datasets
from data.imagenet import get_imagenet_100_datasets
from data.cub import get_cub_datasets
from data.fgvc_aircraft import get_aircraft_datasets
from data.stanford_cars import get_scars_datasets

from data.cifar import subsample_classes as subsample_dataset_cifar
from data.tiny_imagenet import subsample_classes as subsample_dataset_tiny_imagenet
from data.imagenet import subsample_classes as subsample_dataset_imagenet
from data.cub import subsample_classes as subsample_dataset_cub
from data.fgvc_aircraft import subsample_classes as subsample_dataset_aircraft
from data.stanford_cars import subsample_classes as subsample_dataset_scars

from config import osr_split_dir


sub_sample_class_funcs = {
    'cifar10': subsample_dataset_cifar,
    'cifar100': subsample_dataset_cifar,
    'tiny_imagenet': subsample_dataset_tiny_imagenet,
    'imagenet_100': subsample_dataset_imagenet,
    'cub': subsample_dataset_cub,
    'aircraft': subsample_dataset_aircraft,
    'scars': subsample_dataset_scars,
}

get_dataset_funcs = {
    'cifar10': get_cifar_10_datasets,
    'cifar100': get_cifar_100_datasets,
    'tiny_imagenet': get_tiny_imagenet_datasets,
    'imagenet_100': get_imagenet_100_datasets,
    'cub': get_cub_datasets,
    'aircraft': get_aircraft_datasets,
    'scars': get_scars_datasets,
}


def get_datasets(dataset_name, train_transform, test_transform, args):
    """
    :return: train_dataset: MergedDataset which concatenates labelled and unlabelled
             test_dataset,
             unlabelled_train_examples_test,
             datasets
    """
    if dataset_name not in get_dataset_funcs.keys():
        raise ValueError

    # dataset split dict
    dataset_split_config_dict = {
        'continual_session_num': args.continual_session_num,
        'online_novel_unseen_num': args.online_novel_unseen_num,
        'online_old_seen_num': args.online_old_seen_num,
        'online_novel_seen_num': args.online_novel_seen_num,
    }

    # Get datasets
    get_dataset_f = get_dataset_funcs[dataset_name]
    datasets, novel_targets_shuffle = get_dataset_f(train_transform=train_transform, test_transform=test_transform,
                                                    config_dict=dataset_split_config_dict,
                                                    train_classes=args.train_classes,
                                                    prop_train_labels=args.prop_train_labels,
                                                    split_train_val=False,
                                                    is_shuffle=args.shuffle_classes,
                                                    seed=args.seed)   # NOTE!!! seed for shuffle

    # Set target transforms:
    target_transform_dict = {}

    # for i, cls in enumerate(list(args.train_classes) + list(args.unlabeled_classes)):
    #     target_transform_dict[cls] = i
    # NOTE!!! shuffle
    for i, cls in enumerate(list(args.train_classes) + list(novel_targets_shuffle)):   # NOTE!!! novel_targets_shuffle
        target_transform_dict[cls] = i
    target_transform = lambda x: target_transform_dict[x]

    for dataset_name, dataset in datasets.items():
        if dataset is not None:
            if type(dataset) is list:
                for d in dataset:
                    d.target_transform = target_transform
            else:
                dataset.target_transform = target_transform

    # offline session data
    ####################################################################################################################
    offline_session_train_dataset = datasets['offline_train_dataset']   # offline: train dataset of labeled old classes data
    offline_session_test_dataset = datasets['offline_test_dataset']   # offline: test datasets of old classes
    ####################################################################################################################


    # online session data
    ####################################################################################################################
    online_session_train_dataset_list = []
    for old_dataset_unlabelled, novel_dataset_unlabelled in zip(
            datasets['online_old_dataset_unlabelled_list'], datasets['online_novel_dataset_unlabelled_list']):

        online_session_train_dataset = MergedUnlabelledDataset(
            old_unlabelled_dataset=deepcopy(old_dataset_unlabelled),
            novel_unlabelled_dataset=deepcopy(novel_dataset_unlabelled))

        online_session_train_dataset_list.append(online_session_train_dataset)   # online: train dataset list of unlabeled old + unlabeled novel

    online_session_test_dataset_list = datasets['online_test_dataset_list']   # online: test dataset list of old + novel
    ####################################################################################################################


    return  offline_session_train_dataset, offline_session_test_dataset,\
            online_session_train_dataset_list, online_session_test_dataset_list,\
            datasets, dataset_split_config_dict, novel_targets_shuffle



def get_class_splits(args):

    # For FGVC datasets, optionally return bespoke splits
    if args.dataset_name in ('scars', 'cub', 'aircraft'):
        if hasattr(args, 'use_ssb_splits'):
            use_ssb_splits = args.use_ssb_splits
        else:
            use_ssb_splits = False

    # -------------
    # GET CLASS SPLITS
    # -------------
    if args.dataset_name == 'cifar10':

        args.image_size = 32
        args.train_classes = range(7)
        args.unlabeled_classes = range(7, 10)

        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 10)

    elif args.dataset_name == 'cifar100':

        args.image_size = 32
        args.train_classes = range(80)
        args.unlabeled_classes = range(80, 100)

        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 100)

    elif args.dataset_name == 'tiny_imagenet':

        args.image_size = 64
        args.train_classes = range(150)
        args.unlabeled_classes = range(150, 200)

        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 200)


    elif args.dataset_name == 'imagenet_100':

        args.image_size = 224
        args.train_classes = range(50)
        args.unlabeled_classes = range(50, 100)

        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 100)


    elif args.dataset_name == 'cub':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'cub_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(100)
            args.unlabeled_classes = range(100, 200)

        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 200)


    elif args.dataset_name == 'scars':

        args.image_size = 224

        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'scars_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(98)
            args.unlabeled_classes = range(98, 196)

        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 196)


    elif args.dataset_name == 'aircraft':

        args.image_size = 224
        if use_ssb_splits:

            split_path = os.path.join(osr_split_dir, 'aircraft_osr_splits.pkl')
            with open(split_path, 'rb') as handle:
                class_info = pickle.load(handle)

            args.train_classes = class_info['known_classes']
            open_set_classes = class_info['unknown_classes']
            args.unlabeled_classes = open_set_classes['Hard'] + open_set_classes['Medium'] + open_set_classes['Easy']

        else:

            args.train_classes = range(50)
            args.unlabeled_classes = range(50, 100)

        if args.num_old_classes > 0:
            args.train_classes = range(args.num_old_classes)
            args.unlabeled_classes = range(args.num_old_classes, 100)


    elif args.dataset_name == 'herbarium_19':

        args.image_size = 224
        herb_path_splits = os.path.join(osr_split_dir, 'herbarium_19_class_splits.pkl')

        with open(herb_path_splits, 'rb') as handle:
            class_splits = pickle.load(handle)

        args.train_classes = class_splits['Old']
        args.unlabeled_classes = class_splits['New']


    elif args.dataset_name == 'chinese_traffic_signs':

        args.image_size = 224
        args.train_classes = range(28)
        args.unlabeled_classes = range(28, 56)

    else:

        raise NotImplementedError

    return args


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)]