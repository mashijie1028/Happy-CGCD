import os
import pandas as pd
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import cub_root


# dataset_split_config_dict = {
#     'cub': {#'offline_old_cls_num': 100,
#                 #'offine_prop_train_labels': 0.8,   # offline, ratio of labeled data from old classes
#                 'continual_session_num': 5,   # num of continual learining sessions
#                 'online_novel_unseen_num': 25,   # each continual session: num of samples per novel (unseen & first-time) class
#                 'online_old_seen_num': 5,   # each continual session: num of samples per old (labeled) class
#                 'online_novel_seen_num': 5,   # each continual session: num of samples per novel (seen) class
#                  },
# }


class CustomCub2011(Dataset):
    base_folder = 'CUB_200_2011/images'
    url = 'http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz'
    filename = 'CUB_200_2011.tgz'
    tgz_md5 = '97eceeb196236b17998738112f37df78'

    def __init__(self, root, train=True, transform=None, target_transform=None, loader=default_loader, download=False):

        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        self.loader = loader
        self.train = train


        if download:
            self._download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        self.uq_idxs = np.array(range(len(self)))

    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'images.txt'), sep=' ',
                             names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'image_class_labels.txt'),
                                         sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'CUB_200_2011', 'train_test_split.txt'),
                                       sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def _check_integrity(self):
        try:
            self._load_metadata()
        except Exception:
            return False

        for index, row in self.data.iterrows():
            filepath = os.path.join(self.root, self.base_folder, row.filepath)
            if not os.path.isfile(filepath):
                print(filepath)
                return False
        return True

    def _download(self):
        import tarfile

        if self._check_integrity():
            print('Files already downloaded and verified')
            return

        download_url(self.url, self.root, self.filename, self.tgz_md5)

        with tarfile.open(os.path.join(self.root, self.filename), "r:gz") as tar:
            tar.extractall(path=self.root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, self.base_folder, sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, self.uq_idxs[idx]


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.data = dataset.data[mask]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(160)):

    include_classes_cub = np.array(include_classes) + 1     # CUB classes are indexed 1 --> 200 instead of 0 --> 199
    cls_idxs = [x for x, (_, r) in enumerate(dataset.data.iterrows()) if int(r['target']) in include_classes_cub]

    # TODO: For now have no target transform
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
    # NOTE!!! pandas data for CUB
    wholeDataset.data = pd.concat([
        d.data for d in datalist], axis=0)
    wholeDataset.uq_idxs = np.concatenate([
        d.uq_idxs for d in datalist], axis=0)

    return wholeDataset


def get_train_val_indices(train_dataset, val_split=0.2):

    train_classes = np.unique(train_dataset.data['target'])

    # Get train/test indices
    train_idxs = []
    val_idxs = []
    for cls in train_classes:

        cls_idxs = np.where(train_dataset.data['target'] == cls)[0]

        v_ = np.random.choice(cls_idxs, replace=False, size=((int(val_split * len(cls_idxs))),))
        t_ = [x for x in cls_idxs if x not in v_]

        train_idxs.extend(t_)
        val_idxs.extend(v_)

    return train_idxs, val_idxs


# CUB dataset for Continual-GCD
def get_cub_datasets(train_transform, test_transform, config_dict, train_classes=range(100),
                           prop_train_labels=0.8, split_train_val=False, is_shuffle=False, seed=0):
    continual_session_num = config_dict['continual_session_num']
    online_novel_unseen_num = config_dict['online_novel_unseen_num']
    online_old_seen_num = config_dict['online_old_seen_num']
    online_novel_seen_num = config_dict['online_novel_seen_num']

    # Init entire training set
    whole_training_set = CustomCub2011(root=cub_root, transform=train_transform, train=True, download=False)

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)

    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                            for targets in list(train_classes)]

    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                               for samples in each_old_all_samples]

    each_old_unlabeled_slices = [
        np.array(list(set(list(range(len(samples.data)))) - set(each_old_labeled_slices[i])))
        for i, samples in enumerate(each_old_all_samples)]

    each_old_labeled_samples = [subsample_dataset(deepcopy(samples), each_old_labeled_slices[i])
                                    for i, samples in enumerate(each_old_all_samples)]

    each_old_unlabeled_samples = [subsample_dataset(deepcopy(samples), each_old_unlabeled_slices[i])
                                    for i, samples in enumerate(each_old_all_samples)]


    '''----------------------------- offline old classes labeled samples -----------------------------------------------'''
    offline_train_dataset_samples = each_old_labeled_samples

    offline_train_dataset_samples = subDataset_wholeDataset(
        [offline_train_dataset_samples[cls] for cls in range(len(list(train_classes)))])

    # Get test set for all classes
    test_dataset = CustomCub2011(root=cub_root, transform=test_transform, train=False)
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
    novel_cls_labels = [int(r['target'])-1 for x, (_, r) in enumerate(novel_dataset_unlabelled.data.iterrows())]
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
        'offline_train_dataset': offline_train_dataset_samples,  # 2400
        'offline_test_dataset': offline_test_dataset,   # 2864
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,  # list [500, 500, 500, 500, 500]
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,  # list: [400, 500, 600, 700, 800]
        'online_test_dataset_list': online_test_dataset_list,  # list: [3430, 4028, 4614, 5206, 5794]
    }

    return all_datasets, novel_targets_shuffle



# if __name__ == '__main__':

#     all_datasets, novel_targets_shuffle = get_cub_datasets(None, None, dataset_split_config_dict['cub'], range(100), 0.8, False, True, 0)

#     z = 0
