import os
import pandas as pd
import numpy as np
from copy import deepcopy
from scipy import io as mat_io

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import car_root

class CarsDataset(Dataset):
    """
        Cars Dataset
    """
    def __init__(self, train=True, limit=0, data_dir=car_root, transform=None):

        metas = os.path.join(data_dir, 'devkit/cars_train_annos.mat') if train else os.path.join(data_dir, 'devkit/cars_test_annos_withlabels.mat')
        data_dir = os.path.join(data_dir, 'cars_train/') if train else os.path.join(data_dir, 'cars_test/')

        self.loader = default_loader
        self.data_dir = data_dir
        self.data = []
        self.target = []
        self.train = train

        self.transform = transform

        if not isinstance(metas, str):
            raise Exception("Train metas must be string location !")
        labels_meta = mat_io.loadmat(metas)

        for idx, img_ in enumerate(labels_meta['annotations'][0]):
            if limit:
                if idx > limit:
                    break

            # self.data.append(img_resized)
            self.data.append(data_dir + img_[5][0])
            # if self.mode == 'train':
            self.target.append(img_[4][0][0])

        self.uq_idxs = np.array(range(len(self)))
        self.target_transform = None

    def __getitem__(self, idx):

        image = self.loader(self.data[idx])
        target = self.target[idx] - 1

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        idx = self.uq_idxs[idx]

        return image, target, idx

    def __len__(self):
        return len(self.data)


def subsample_dataset(dataset, idxs):

    dataset.data = np.array(dataset.data)[idxs].tolist()
    dataset.target = np.array(dataset.target)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cars = np.array(include_classes) + 1     # SCars classes are indexed 1 --> 196 instead of 0 --> 195
    cls_idxs = [x for x, t in enumerate(dataset.target) if t in include_classes_cars]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    # dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


def subDataset_wholeDataset(datalist):
    wholeDataset = deepcopy(datalist[0])
    wholeDataset.data = np.concatenate([
        d.data for d in datalist], axis=0)
    wholeDataset.target = np.concatenate([
        d.target for d in datalist], axis=0).tolist()
    wholeDataset.uq_idxs = np.concatenate([
        d.uq_idxs for d in datalist], axis=0)

    return wholeDataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.target)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.target == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


# scars dataset for Continual-GCD
def get_scars_datasets(train_transform, test_transform, config_dict, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, is_shuffle=False, seed=0):
    continual_session_num = config_dict['continual_session_num']
    online_novel_unseen_num = config_dict['online_novel_unseen_num']
    online_old_seen_num = config_dict['online_old_seen_num']
    online_novel_seen_num = config_dict['online_novel_seen_num']

    # Init entire training set
    whole_training_set = CarsDataset(data_dir=car_root, transform=train_transform, train=True)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)  # 40000

    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                            for targets in list(train_classes)]  # 80*500   # NOTE!!!

    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                               for samples in each_old_all_samples]  # 80*400

    each_old_unlabeled_slices = [
        np.array(list(set(list(range(len(samples.target)))) - set(each_old_labeled_slices[i])))
        for i, samples in enumerate(each_old_all_samples)]  # 80*100

    each_old_labeled_samples = [subsample_dataset(deepcopy(samples), each_old_labeled_slices[i])
                                    for i, samples in enumerate(each_old_all_samples)]   # 80*400

    each_old_unlabeled_samples = [subsample_dataset(deepcopy(samples), each_old_unlabeled_slices[i])
                                    for i, samples in enumerate(each_old_all_samples)]   # 80*100 for online old classes unlabeled


    '''----------------------------- offline old classes labeled samples -----------------------------------------------'''
    offline_train_dataset_samples = each_old_labeled_samples   # 80*400

    offline_train_dataset_samples = subDataset_wholeDataset(
        [offline_train_dataset_samples[cls] for cls in range(len(list(train_classes)))])   # 32000   # NOTE!!!

    # Get test set for all classes
    test_dataset = CarsDataset(data_dir=car_root, transform=test_transform, train=False)
    # offline test dataset
    offline_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=list(train_classes))


    '''----------------------------- online old classes unlabeled samples ----------------------------------------------'''
    online_old_dataset_unlabelled_list = []
    for s in range(continual_session_num):
        # randomly sample old samples for each online session
        online_session_each_old_slices = [np.random.choice(np.array(list(range(len(samples.target)))), online_old_seen_num, replace=False)
                                          for samples in each_old_unlabeled_samples]
        online_session_old_samples = [subsample_dataset(deepcopy(samples), online_session_each_old_slices[i])
                                      for i, samples in enumerate(each_old_unlabeled_samples)]   # 80*50

        online_session_old_dataset = subDataset_wholeDataset(online_session_old_samples)
        online_old_dataset_unlabelled_list.append(online_session_old_dataset)


    '''---------------------------- online novel classes unlabeled samples ---------------------------------------------'''
    novel_unlabelled_indices = set(whole_training_set.uq_idxs) - set(old_dataset_all.uq_idxs) #10000
    novel_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set),
                                                 np.array(list(novel_unlabelled_indices)))  # 10000

    #novel_targets_shuffle = np.array(list(set(np.array(novel_dataset_unlabelled.target).tolist())))
    novel_targets_shuffle = np.array(list(set(np.array(novel_dataset_unlabelled.target).tolist()))) - 1   # NOTE!!! -1
    # NOTE!!! shuffle classes
    if is_shuffle:
        np.random.seed(seed)
        np.random.shuffle(novel_targets_shuffle)

    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []
    targets_per_session = len(novel_targets_shuffle) // continual_session_num
    for s in range(continual_session_num):
        online_session_targets = novel_targets_shuffle[0: s * targets_per_session + targets_per_session]
        if s == continual_session_num - 1:   # NOTE!!! for scars, 98//5
            online_session_targets = novel_targets_shuffle
        online_session_each_novel_samples = [subsample_classes(deepcopy(novel_dataset_unlabelled), include_classes=[targets])
                                             for targets in online_session_targets]   # n * 500

        # randomly sample novel samples for each online session
        online_session_each_novel_slices = []
        for i in range(len(online_session_targets)):
            if (s >= 1) and (i < s * targets_per_session):
                online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].target)))), 
                                                                         online_novel_seen_num, replace=False))
            else:
                #online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].target)))), 
                #                                                         online_novel_unseen_num, replace=False))

                # for long-tailed scars   # NOTE!!!
                if len(online_session_each_novel_samples[i].target) > online_novel_unseen_num:   # NOTE!!! for long-tailed ImageNet!!!
                    online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].target)))), 
                                                                         online_novel_unseen_num, replace=False))
                else:
                    online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].target)))), 
                                                                         len(online_session_each_novel_samples[i].target), replace=False))

        online_session_novel_samples = [subsample_dataset(deepcopy(samples), online_session_each_novel_slices[i])
                                        for i, samples in enumerate(online_session_each_novel_samples)]   # [50, 50, ...],  len=n

        online_session_novel_dataset = subDataset_wholeDataset(online_session_novel_samples)
        online_novel_dataset_unlabelled_list.append(online_session_novel_dataset)

        # online session test dataset
        online_session_test_dataset = subsample_classes(
            deepcopy(test_dataset), include_classes=list(train_classes) + online_session_targets.tolist())
        online_test_dataset_list.append(online_session_test_dataset)


    '''---------------------------------- all datasets for Continual-GCD -----------------------------------------------'''
    all_datasets = {
        'offline_train_dataset': offline_train_dataset_samples,  # 3205
        'offline_test_dataset': offline_test_dataset,   # 4002
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,  # list [588, 588, 588, 588, 588]
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,  # list: [568, 678, 798, 911, 1116]
        'online_test_dataset_list': online_test_dataset_list,  # list: [4781, 5569, 6347, 7133, 8041]
    }

    return all_datasets, novel_targets_shuffle


# if __name__ == '__main__':

#     x = get_scars_datasets(None, None, train_classes=range(98), prop_train_labels=0.5, split_train_val=False)

#     print('Printing lens...')
#     for k, v in x.items():
#         if v is not None:
#             print(f'{k}: {len(v)}')

#     print('Printing labelled and unlabelled overlap...')
#     print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
#     print('Printing total instances in train...')
#     print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))

#     print(f'Num Labelled Classes: {len(set(x["train_labelled"].target))}')
#     print(f'Num Unabelled Classes: {len(set(x["train_unlabelled"].target))}')
#     print(f'Len labelled set: {len(x["train_labelled"])}')
#     print(f'Len unlabelled set: {len(x["train_unlabelled"])}')