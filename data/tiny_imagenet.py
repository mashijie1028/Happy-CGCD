import os
from copy import deepcopy
import numpy as np
import pandas as pd
import warnings
from torchvision.datasets import ImageFolder
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import extract_archive, check_integrity, download_url, verify_str_arg

from data.data_utils import subsample_instances
from config import tiny_imagenet_root


# dataset_split_config_dict = {
#     'tiny_imagenet': {#'offline_old_cls_num': 100,
#                 #'offine_prop_train_labels': 0.8,   # offline, ratio of labeled data from old classes
#                 'continual_session_num': 5,   # num of continual learining sessions
#                 'online_novel_unseen_num': 400,   # each continual session: num of samples per novel (unseen & first-time) class
#                 'online_old_seen_num': 25,   # each continual session: num of samples per old (labeled) class
#                 'online_novel_seen_num': 25,   # each continual session: num of samples per novel (seen) class
#                  },
# }


'''
IMPORTANT!!! https://github.com/lvyilin/pytorch-fgvc-dataset/blob/master/tiny_imagenet.py
Also, similar to FGVCAircraft, please refer to this dataset in GCD repository.
'''


def find_classes(class_file):
    with open(class_file) as r:
        classes = list(map(lambda s: s.strip(), r.readlines()))

    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}

    return classes, class_to_idx


def make_dataset(root, base_folder, dirname, class_to_idx):
    images = []
    dir_path = os.path.join(root, base_folder, dirname)

    if dirname == 'train':
        for fname in sorted(os.listdir(dir_path)):
            cls_fpath = os.path.join(dir_path, fname)
            if os.path.isdir(cls_fpath):
                cls_imgs_path = os.path.join(cls_fpath, 'images')
                for imgname in sorted(os.listdir(cls_imgs_path)):
                    path = os.path.join(cls_imgs_path, imgname)
                    item = (path, class_to_idx[fname])
                    images.append(item)
    else:
        imgs_path = os.path.join(dir_path, 'images')
        imgs_annotations = os.path.join(dir_path, 'val_annotations.txt')

        with open(imgs_annotations) as r:
            data_info = map(lambda s: s.split('\t'), r.readlines())

        cls_map = {line_data[0]: line_data[1] for line_data in data_info}

        for imgname in sorted(os.listdir(imgs_path)):
            path = os.path.join(imgs_path, imgname)
            item = (path, class_to_idx[cls_map[imgname]])
            images.append(item)

    return images


class TinyImageNet(VisionDataset):
    """`tiny-imageNet <http://cs231n.stanford.edu/tiny-imagenet-200.zip>`_ Dataset.

        Args:
            root (string): Root directory of the dataset.
            split (string, optional): The dataset split, supports ``train``, or ``val``.
            transform (callable, optional): A function/transform that  takes in an PIL image
               and returns a transformed version. E.g, ``transforms.RandomCrop``
            target_transform (callable, optional): A function/transform that takes in the
               target and transforms it.
            download (bool, optional): If true, downloads the dataset from the internet and
               puts it in root directory. If dataset is already downloaded, it is not
               downloaded again.
    """
    base_folder = 'tiny-imagenet-200/'
    url = 'http://cs231n.stanford.edu/tiny-imagenet-200.zip'
    filename = 'tiny-imagenet-200.zip'
    md5 = '90528d7ca1a48142e341f4ef8d21d0de'

    def __init__(self, root=tiny_imagenet_root, split='train', transform=None, target_transform=None, download=False):
        super(TinyImageNet, self).__init__(root, transform=transform, target_transform=target_transform)

        self.dataset_path = os.path.join(root, self.base_folder)
        self.loader = default_loader
        self.split = verify_str_arg(split, "split", ("train", "val",))

        if self._check_integrity():
            print('Files already downloaded and verified.')
        elif download:
            self._download()
        else:
            raise RuntimeError(
                'Dataset not found. You can use download=True to download it.')
        if not os.path.isdir(self.dataset_path):
            print('Extracting...')
            extract_archive(os.path.join(root, self.filename))

        _, class_to_idx = find_classes(os.path.join(self.dataset_path, 'wnids.txt'))

        self.data = make_dataset(self.root, self.base_folder, self.split, class_to_idx)

        self.uq_idxs = np.array(range(len(self)))   # NOTE!!! for GCD

    def _download(self):
        print('Downloading...')
        download_url(self.url, root=self.root, filename=self.filename)
        print('Extracting...')
        extract_archive(os.path.join(self.root, self.filename))

    def _check_integrity(self):
        return check_integrity(os.path.join(self.root, self.filename), self.md5)

    def __getitem__(self, index):
        img_path, target = self.data[index]
        image = self.loader(img_path)

        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        #return image, target
        return image, target, self.uq_idxs[index]   # NOTE!!!

    def __len__(self):
        return len(self.data)



def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = [(p, t) for i, (p, t) in enumerate(dataset.data) if i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(100)):

    cls_idxs = [i for i, (p, t) in enumerate(dataset.data) if t in include_classes]

    # TODO: Don't transform targets for now
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
    wholeDataset.data = []
    for d in datalist:
        wholeDataset.data.extend(d.data)
    wholeDataset.uq_idxs = np.concatenate([
        d.uq_idxs for d in datalist], axis=0)

    return wholeDataset


def get_train_val_indices(train_dataset, val_split=0.2):

    all_targets = [t for i, (p, t) in enumerate(train_dataset.data)]
    train_classes = np.unique(all_targets)

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:
        cls_idxs = np.where(all_targets == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs



# tiny-imagenet dataset for Continual-GCD
def get_tiny_imagenet_datasets(train_transform, test_transform, config_dict, train_classes=range(100),
                                prop_train_labels=0.8, split_train_val=False, is_shuffle=False, seed=0):
    continual_session_num = config_dict['continual_session_num']
    online_novel_unseen_num = config_dict['online_novel_unseen_num']
    online_old_seen_num = config_dict['online_old_seen_num']
    online_novel_seen_num = config_dict['online_novel_seen_num']

    # Init entire training set
    whole_training_set = TinyImageNet(root=tiny_imagenet_root, split='train', transform=train_transform, download=False)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)

    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                            for targets in list(train_classes)]    # NOTE!!!

    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                               for samples in each_old_all_samples]

    each_old_unlabeled_slices = [
        np.array(list(set(list(range(len(samples.data)))) - set(each_old_labeled_slices[i])))
        for i, samples in enumerate(each_old_all_samples)]

    each_old_labeled_samples = [subsample_dataset(deepcopy(samples), each_old_labeled_slices[i])
                                    for i, samples in enumerate(each_old_all_samples)]

    each_old_unlabeled_samples = [subsample_dataset(deepcopy(samples), each_old_unlabeled_slices[i])
                                    for i, samples in enumerate(each_old_all_samples)]   # for online old classes unlabeled


    '''----------------------------- offline old classes labeled samples -----------------------------------------------'''
    offline_train_dataset_samples = each_old_labeled_samples

    offline_train_dataset_samples = subDataset_wholeDataset(
        [offline_train_dataset_samples[cls] for cls in range(len(list(train_classes)))])   # NOTE!!!

    # Get test set for all classes
    test_dataset = TinyImageNet(root=tiny_imagenet_root, split='val', transform=test_transform, download=False)
    # offline test dataset
    offline_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=list(train_classes))


    '''----------------------------- online old classes unlabeled samples ----------------------------------------------'''
    online_old_dataset_unlabelled_list = []
    for s in range(continual_session_num):
        # randomly sample old samples for each online session
        online_session_each_old_slices = [np.random.choice(np.array(list(range(len(samples.data)))), online_old_seen_num, replace=False)
                                          for samples in each_old_unlabeled_samples]
        online_session_old_samples = [subsample_dataset(deepcopy(samples), online_session_each_old_slices[i])
                                      for i, samples in enumerate(each_old_unlabeled_samples)]

        online_session_old_dataset = subDataset_wholeDataset(online_session_old_samples)
        online_old_dataset_unlabelled_list.append(online_session_old_dataset)


    '''---------------------------- online novel classes unlabeled samples ---------------------------------------------'''
    novel_unlabelled_indices = set(whole_training_set.uq_idxs) - set(old_dataset_all.uq_idxs)
    novel_dataset_unlabelled = subsample_dataset(deepcopy(whole_training_set),
                                                 np.array(list(novel_unlabelled_indices)))

    # NOTE!!!
    novel_cls_labels = [t for i, (p, t) in enumerate(novel_dataset_unlabelled.data)]   # NOTE!!!
    novel_targets_shuffle = np.array(list(set(np.array(novel_cls_labels).tolist())))
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
                                             for targets in online_session_targets]

        # randomly sample novel samples for each online session
        online_session_each_novel_slices = []
        for i in range(len(online_session_targets)):
            if (s >= 1) and (i < s * targets_per_session):
                online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].data)))), 
                                                                         online_novel_seen_num, replace=False))
            else:
                online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].data)))), 
                                                                         online_novel_unseen_num, replace=False))

        online_session_novel_samples = [subsample_dataset(deepcopy(samples), online_session_each_novel_slices[i])
                                        for i, samples in enumerate(online_session_each_novel_samples)]

        online_session_novel_dataset = subDataset_wholeDataset(online_session_novel_samples)
        online_novel_dataset_unlabelled_list.append(online_session_novel_dataset)

        # online session test dataset
        online_session_test_dataset = subsample_classes(
            deepcopy(test_dataset), include_classes=list(train_classes) + online_session_targets.tolist())
        online_test_dataset_list.append(online_session_test_dataset)


    '''---------------------------------- all datasets for Continual-GCD -----------------------------------------------'''
    all_datasets = {
        'offline_train_dataset': offline_train_dataset_samples,  # 40000
        'offline_test_dataset': offline_test_dataset,   # 5000
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,  # list [2500, 2500, 2500, 2500, 2500]
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,  # list: [8000, 8500, 9000, 9500, 10000]
        'online_test_dataset_list': online_test_dataset_list,  # list: [6000, 7000, 8000, 9000, 10000]
    }

    return all_datasets, novel_targets_shuffle


# if __name__ == '__main__':
#     #whole_training_set = TinyImageNet(root=tiny_imagenet_root, split='train', transform=None, download=False)
#     #print(len(whole_training_set), len(whole_training_set.data))   # 100000 100000
#     datasets = get_tiny_imagenet_datasets(train_transform=None, test_transform=None,
#                                 config_dict=dataset_split_config_dict['tiny_imagenet'],
#                                 train_classes=range(100),
#                                 prop_train_labels=0.8,
#                                 split_train_val=False)

