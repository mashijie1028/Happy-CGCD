import numpy as np
from torch.utils.data import Dataset
import torch


def subsample_instances(dataset, prop_indices_to_subsample=0.8):

    np.random.seed(0)
    subsample_indices = np.random.choice(range(len(dataset)), replace=False,
                                         size=(int(prop_indices_to_subsample * len(dataset)),))

    return subsample_indices


class MergedDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, labelled_dataset, unlabelled_dataset):

        self.labelled_dataset = labelled_dataset
        self.unlabelled_dataset = unlabelled_dataset
        self.target_transform = None

    def __getitem__(self, item):

        if item < len(self.labelled_dataset):
            img, label, uq_idx = self.labelled_dataset[item]
            labeled_or_not = 1

        else:

            img, label, uq_idx = self.unlabelled_dataset[item - len(self.labelled_dataset)]
            labeled_or_not = 0


        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.unlabelled_dataset) + len(self.labelled_dataset)


class MergedUnlabelledDataset(Dataset):

    """
    Takes two datasets (labelled_dataset, unlabelled_dataset) and merges them
    Allows you to iterate over them in parallel
    """

    def __init__(self, old_unlabelled_dataset, novel_unlabelled_dataset, pretrain=False):

        self.old_unlabelled_dataset = old_unlabelled_dataset
        self.novel_unlabelled_dataset = novel_unlabelled_dataset
        self.target_transform = None
        self.pretrain = pretrain

    def __getitem__(self, item):

        if item < len(self.old_unlabelled_dataset):
            img, label, uq_idx = self.old_unlabelled_dataset[item]
        else:
            img, label, uq_idx = self.novel_unlabelled_dataset[item - len(self.old_unlabelled_dataset)]

        if self.pretrain:
            labeled_or_not = 1
        else:
            labeled_or_not = 0

        return img, label, uq_idx, np.array([labeled_or_not])

    def __len__(self):
        return len(self.old_unlabelled_dataset) + len(self.novel_unlabelled_dataset)


def re_assign_labels(old_labels, template_old_labels=None):
    if template_old_labels is None:
        new_labels = np.zeros_like(old_labels)
        old_labels_unique = np.unique(old_labels)
        for new_label, old_label in enumerate(old_labels_unique):
            new_labels[old_labels == old_label] = new_label
    else:
        new_labels = np.zeros_like(old_labels)
        template_old_labels_unique = np.unique(template_old_labels)
        for new_label, old_label in enumerate(template_old_labels_unique):
            new_labels[old_labels == old_label] = new_label
    return new_labels