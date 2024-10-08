import torchvision
import numpy as np

import os

from copy import deepcopy
from data.data_utils import subsample_instances
from config import imagenet_root


# dataset_split_config_dict = {
#     'imagenet_100': {#'offline_old_cls_num': 50,
#                 #'offine_prop_train_labels': 0.8,   # offline, ratio of labeled data from old classes
#                 'continual_session_num': 5,   # num of continual learining sessions
#                 'online_novel_unseen_num': 1000,   # each continual session: num of samples per novel (unseen & first-time) class
#                 'online_old_seen_num': 60,   # each continual session: num of samples per old (labeled) class
#                 'online_novel_seen_num': 60,   # each continual session: num of samples per novel (seen) class
#                  },
# }


class ImageNetBase(torchvision.datasets.ImageFolder):

    def __init__(self, root, transform):

        super(ImageNetBase, self).__init__(root, transform)

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, item):

        img, label = super().__getitem__(item)
        uq_idx = self.uq_idxs[item]

        return img, label, uq_idx


def subsample_dataset(dataset, idxs):

    imgs_ = []
    for i in idxs:
        imgs_.append(dataset.imgs[i])
    dataset.imgs = imgs_

    samples_ = []
    for i in idxs:
        samples_.append(dataset.samples[i])
    dataset.samples = samples_

    # dataset.imgs = [x for i, x in enumerate(dataset.imgs) if i in idxs]
    # dataset.samples = [x for i, x in enumerate(dataset.samples) if i in idxs]

    dataset.targets = np.array(dataset.targets)[idxs].tolist()
    dataset.uq_idxs = dataset.uq_idxs[idxs]

    return dataset


def subsample_classes(dataset, include_classes=list(range(1000))):

    cls_idxs = [x for x, t in enumerate(dataset.targets) if t in include_classes]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)
    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset


# def subDataset_wholeDataset(datalist):
#     wholeDataset = deepcopy(datalist[0])
#     wholeDataset.data = np.concatenate([
#         d.data for d in datalist], axis=0)
#     wholeDataset.targets = np.concatenate([
#         d.targets for d in datalist], axis=0).tolist()
#     wholeDataset.uq_idxs = np.concatenate([
#         d.uq_idxs for d in datalist], axis=0)

#     return wholeDataset


def subDataset_wholeDataset(datalist):
    wholeDataset = deepcopy(datalist[0])

    wholeDataset.imgs = []
    wholeDataset.samples = []
    for d in datalist:
        wholeDataset.imgs.extend(d.imgs)
        wholeDataset.samples.extend(d.samples)

    wholeDataset.targets = np.concatenate([
        d.targets for d in datalist], axis=0).tolist()
    wholeDataset.uq_idxs = np.concatenate([
        d.uq_idxs for d in datalist], axis=0)

    return wholeDataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = list(set(train_dataset.targets))

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(np.array(train_dataset.targets) == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


def get_imagenet_100_datasets(train_transform, test_transform, config_dict, train_classes=range(80),
                           prop_train_labels=0.8, split_train_val=False, is_shuffle=False, seed=0):
    continual_session_num = config_dict['continual_session_num']
    online_novel_unseen_num = config_dict['online_novel_unseen_num']
    online_old_seen_num = config_dict['online_old_seen_num']
    online_novel_seen_num = config_dict['online_novel_seen_num']

    # NOTE!!! first shuffle to sample ImageNet-Subset, e.g., ImageNet-100
    np.random.seed(seed)

    # Subsample imagenet dataset initially to include 100 classes
    subsampled_100_classes = np.random.choice(range(1000), size=(100,), replace=False)
    subsampled_100_classes = np.sort(subsampled_100_classes)
    print(f'Constructing ImageNet-100 dataset from the following classes: {subsampled_100_classes.tolist()}')
    cls_map = {i: j for i, j in zip(subsampled_100_classes, range(100))}

    # Init entire training set
    imagenet_training_set = ImageNetBase(root=os.path.join(imagenet_root, 'train'), transform=train_transform)
    whole_training_set = subsample_classes(imagenet_training_set, include_classes=subsampled_100_classes)

    # Reset dataset   # NOTE!!!
    whole_training_set.samples = [(s[0], cls_map[s[1]]) for s in whole_training_set.samples]
    whole_training_set.targets = [s[1] for s in whole_training_set.samples]
    whole_training_set.uq_idxs = np.array(range(len(whole_training_set)))
    whole_training_set.target_transform = None

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)  # 40000

    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                            for targets in list(train_classes)]  # 80*500   # NOTE!!!

    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                               for samples in each_old_all_samples]  # 80*400

    each_old_unlabeled_slices = [
        np.array(list(set(list(range(len(samples.targets)))) - set(each_old_labeled_slices[i])))
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
    test_dataset = ImageNetBase(root=os.path.join(imagenet_root, 'val'), transform=test_transform)
    test_dataset = subsample_classes(test_dataset, include_classes=subsampled_100_classes)

    # Reset test set   # NOTE!!!
    test_dataset.samples = [(s[0], cls_map[s[1]]) for s in test_dataset.samples]
    test_dataset.targets = [s[1] for s in test_dataset.samples]
    test_dataset.uq_idxs = np.array(range(len(test_dataset)))
    test_dataset.target_transform = None

    # offline test dataset
    offline_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=list(train_classes))


    '''----------------------------- online old classes unlabeled samples ----------------------------------------------'''
    online_old_dataset_unlabelled_list = []
    for s in range(continual_session_num):
        # randomly sample old samples for each online session
        online_session_each_old_slices = [np.random.choice(np.array(list(range(len(samples.targets)))), online_old_seen_num, replace=False)
                                          for samples in each_old_unlabeled_samples]
        online_session_old_samples = [subsample_dataset(deepcopy(samples), online_session_each_old_slices[i])
                                      for i, samples in enumerate(each_old_unlabeled_samples)]   # 80*50

        online_session_old_dataset = subDataset_wholeDataset(online_session_old_samples)
        online_old_dataset_unlabelled_list.append(online_session_old_dataset)


    '''---------------------------- online novel classes unlabeled samples ---------------------------------------------'''
    novel_unlabelled_indices = set(whole_training_set.uq_idxs) - set(old_dataset_all.uq_idxs) #10000
    novel_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set),
                                                 np.array(list(novel_unlabelled_indices)))  # 10000

    novel_targets_shuffle = np.array(list(set(np.array(novel_dataset_unlabelled.targets).tolist())))
    # NOTE!!! shuffle classes
    if is_shuffle:
        np.random.seed(seed)
        np.random.shuffle(novel_targets_shuffle)

    online_novel_dataset_unlabelled_list = []
    online_test_dataset_list = []
    targets_per_session = len(novel_targets_shuffle) // continual_session_num
    for s in range(continual_session_num):
        online_session_targets = novel_targets_shuffle[0: s * targets_per_session + targets_per_session]
        online_session_each_novel_samples = [subsample_classes(deepcopy(novel_dataset_unlabelled), include_classes=[targets])
                                             for targets in online_session_targets]   # n * 500

        # randomly sample novel samples for each online session
        online_session_each_novel_slices = []
        for i in range(len(online_session_targets)):
            if (s >= 1) and (i < s * targets_per_session):
                online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].targets)))), 
                                                                         online_novel_seen_num, replace=False))
            else:
                if len(online_session_each_novel_samples[i].targets) > online_novel_unseen_num:   # NOTE!!! for long-tailed ImageNet!!!
                    online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].targets)))), 
                                                                         online_novel_unseen_num, replace=False))
                else:
                    online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].targets)))), 
                                                                         len(online_session_each_novel_samples[i].targets), replace=False))

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
        'offline_train_dataset': offline_train_dataset_samples,  # 50974
        'offline_test_dataset': offline_test_dataset,   # 2500
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,  # list [3000, 3000, 3000, 3000, 3000]
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,  # list: [9889, 10600, 11108, 11800, 12400]
        'online_test_dataset_list': online_test_dataset_list,  # list: [3000, 3500, 4000, 4500, 5000]
    }

    return all_datasets, novel_targets_shuffle



# if __name__ == '__main__':

#     all_datasets, novel_targets_shuffle = get_imagenet_100_datasets(None, None, dataset_split_config_dict['imagenet_100'], 
#                                                                     range(50), 0.8, False, False, 0)

#     # print(type(all_datasets['offline_train_dataset']))   # <class '__main__.ImageNetBase'>
#     # print(type(all_datasets['offline_test_dataset']))   # <class '__main__.ImageNetBase'>
#     # print(type(all_datasets['online_old_dataset_unlabelled_list'][0]))   # <class '__main__.ImageNetBase'>
#     # print(type(all_datasets['online_novel_dataset_unlabelled_list'][0]))   # <class '__main__.ImageNetBase'>
#     # print(type(all_datasets['online_test_dataset_list'][0]))   # <class '__main__.ImageNetBase'>


#     z = 0
