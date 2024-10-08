import os
import numpy as np
from copy import deepcopy

from torchvision.datasets.folder import default_loader
from torch.utils.data import Dataset

from data.data_utils import subsample_instances
from config import aircraft_root


def make_dataset(dir, image_ids, targets):
    assert(len(image_ids) == len(targets))
    images = []
    dir = os.path.expanduser(dir)
    for i in range(len(image_ids)):
        item = (os.path.join(dir, 'data', 'images',
                             '%s.jpg' % image_ids[i]), targets[i])
        images.append(item)
    return images


def find_classes(classes_file):

    # read classes file, separating out image IDs and class names
    image_ids = []
    targets = []
    f = open(classes_file, 'r')
    for line in f:
        split_line = line.split(' ')
        image_ids.append(split_line[0])
        targets.append(' '.join(split_line[1:]))
    f.close()

    # index class names
    classes = np.unique(targets)
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    targets = [class_to_idx[c] for c in targets]

    return (image_ids, targets, classes, class_to_idx)


class FGVCAircraft(Dataset):

    """`FGVC-Aircraft <http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft>`_ Dataset.

    Args:
        root (string): Root directory path to dataset.
        class_type (string, optional): The level of FGVC-Aircraft fine-grain classification
            to label data with (i.e., ``variant``, ``family``, or ``manufacturer``).
        transform (callable, optional): A function/transform that takes in a PIL image
            and returns a transformed version. E.g. ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in the root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    url = 'http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/fgvc-aircraft-2013b.tar.gz'
    class_types = ('variant', 'family', 'manufacturer')
    splits = ('train', 'val', 'trainval', 'test')

    def __init__(self, root, class_type='variant', split='train', transform=None,
                 target_transform=None, loader=default_loader, download=False):
        if split not in self.splits:
            raise ValueError('Split "{}" not found. Valid splits are: {}'.format(
                split, ', '.join(self.splits),
            ))
        if class_type not in self.class_types:
            raise ValueError('Class type "{}" not found. Valid class types are: {}'.format(
                class_type, ', '.join(self.class_types),
            ))
        self.root = os.path.expanduser(root)
        self.class_type = class_type
        self.split = split
        self.classes_file = os.path.join(self.root, 'data',
                                         'images_%s_%s.txt' % (self.class_type, self.split))

        if download:
            self.download()

        (image_ids, targets, classes, class_to_idx) = find_classes(self.classes_file)
        samples = make_dataset(self.root, image_ids, targets)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = True if split == 'train' else False

        self.uq_idxs = np.array(range(len(self)))

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, self.uq_idxs[index]

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def _check_exists(self):
        return os.path.exists(os.path.join(self.root, 'data', 'images')) and \
            os.path.exists(self.classes_file)

    def download(self):
        """Download the FGVC-Aircraft data if it doesn't exist already."""
        from six.moves import urllib
        import tarfile

        if self._check_exists():
            return

        # prepare to download data to PARENT_DIR/fgvc-aircraft-2013.tar.gz
        print('Downloading %s ... (may take a few minutes)' % self.url)
        parent_dir = os.path.abspath(os.path.join(self.root, os.pardir))
        tar_name = self.url.rpartition('/')[-1]
        tar_path = os.path.join(parent_dir, tar_name)
        data = urllib.request.urlopen(self.url)

        # download .tar.gz file
        with open(tar_path, 'wb') as f:
            f.write(data.read())

        # extract .tar.gz to PARENT_DIR/fgvc-aircraft-2013b
        data_folder = tar_path.strip('.tar.gz')
        print('Extracting %s to %s ... (may take a few minutes)' % (tar_path, data_folder))
        tar = tarfile.open(tar_path)
        tar.extractall(parent_dir)

        # if necessary, rename data folder to self.root
        if not os.path.samefile(data_folder, self.root):
            print('Renaming %s to %s ...' % (data_folder, self.root))
            os.rename(data_folder, self.root)

        # delete .tar.gz file
        print('Deleting %s ...' % tar_path)
        os.remove(tar_path)

        print('Done!')


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = [(p, t) for i, (p, t) in enumerate(dataset.samples) if i in idxs]
    dataset.uq_idxs = dataset.uq_idxs[mask]

    return dataset


def subsample_classes(dataset, include_classes=range(60)):

    cls_idxs = [i for i, (p, t) in enumerate(dataset.samples) if t in include_classes]

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
    wholeDataset.samples = []
    for d in datalist:
        wholeDataset.samples.extend(d.samples)
    wholeDataset.uq_idxs = np.concatenate([
        d.uq_idxs for d in datalist], axis=0)

    return wholeDataset


def get_train_val_indices(train_dataset, val_split=0.2):

    all_targets = [t for i, (p, t) in enumerate(train_dataset.samples)]
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


# fgvc-aircraft dataset for Continual-GCD
def get_aircraft_datasets(train_transform, test_transform, config_dict, train_classes=range(100),
                                prop_train_labels=0.8, split_train_val=False, is_shuffle=False, seed=0):
    continual_session_num = config_dict['continual_session_num']
    online_novel_unseen_num = config_dict['online_novel_unseen_num']
    online_old_seen_num = config_dict['online_old_seen_num']
    online_novel_seen_num = config_dict['online_novel_seen_num']

    # Init entire training set
    whole_training_set = FGVCAircraft(root=aircraft_root, transform=train_transform, split='trainval')

    # Get labelled training set which has subsampled classes, then subsample some indices from that
    old_dataset_all = subsample_classes(deepcopy(whole_training_set), include_classes=train_classes)

    each_old_all_samples = [subsample_classes(deepcopy(old_dataset_all), include_classes=[targets])
                            for targets in list(train_classes)]    # NOTE!!!

    each_old_labeled_slices = [subsample_instances(samples, prop_indices_to_subsample=prop_train_labels)
                               for samples in each_old_all_samples]

    each_old_unlabeled_slices = [
        np.array(list(set(list(range(len(samples.samples)))) - set(each_old_labeled_slices[i])))
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
    test_dataset = FGVCAircraft(root=aircraft_root, transform=test_transform, split='test')
    # offline test dataset
    offline_test_dataset = subsample_classes(deepcopy(test_dataset), include_classes=list(train_classes))


    '''----------------------------- online old classes unlabeled samples ----------------------------------------------'''
    online_old_dataset_unlabelled_list = []
    for s in range(continual_session_num):
        # randomly sample old samples for each online session
        online_session_each_old_slices = [np.random.choice(np.array(list(range(len(samples.samples)))), online_old_seen_num, replace=False)
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
    novel_cls_labels = [t for i, (p, t) in enumerate(novel_dataset_unlabelled.samples)]   # NOTE!!!
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
                online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].samples)))), 
                                                                         online_novel_seen_num, replace=False))
            else:
                online_session_each_novel_slices.append(np.random.choice(np.array(list(range(len(online_session_each_novel_samples[i].samples)))), 
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
        'offline_train_dataset': offline_train_dataset_samples,  # 2634
        'offline_test_dataset': offline_test_dataset,   # 1666
        'online_old_dataset_unlabelled_list': online_old_dataset_unlabelled_list,  # list [500, 500, 500, 500, 500]
        'online_novel_dataset_unlabelled_list': online_novel_dataset_unlabelled_list,  # list: [550, 650, 750, 850, 950]
        'online_test_dataset_list': online_test_dataset_list,  # list: [2000, 2333, 2666, 3000, 3333]
    }

    return all_datasets, novel_targets_shuffle


# if __name__ == '__main__':

#     x = get_aircraft_datasets(None, None, split_train_val=False)

#     print('Printing lens...')
#     for k, v in x.items():
#         if v is not None:
#             print(f'{k}: {len(v)}')

#     print('Printing labelled and unlabelled overlap...')
#     print(set.intersection(set(x['train_labelled'].uq_idxs), set(x['train_unlabelled'].uq_idxs)))
#     print('Printing total instances in train...')
#     print(len(set(x['train_labelled'].uq_idxs)) + len(set(x['train_unlabelled'].uq_idxs)))
#     print('Printing number of labelled classes...')
#     print(len(set([i[1] for i in x['train_labelled'].samples])))
#     print('Printing total number of classes...')
#     print(len(set([i[1] for i in x['train_unlabelled'].samples])))
