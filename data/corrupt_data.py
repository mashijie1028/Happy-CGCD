import os
import numpy as np

import torch
from torch.utils.data import Dataset

import torchvision.transforms.functional as tfunc

ROOT_DIR_10 = "/data4/datasets/Corruptions/CIFAR-10-C/"
ROOT_DIR_100 = "/data4/datasets/Corruptions/CIFAR-100-C/"


class DatasetFromTorchTensor(Dataset):
    def __init__(self, data, target, transform=None):
        # Data type handling must be done beforehand. It is too difficult at this point.
        self.data = data
        self.target = target
        if len(self.target.shape)==1:
            self.target = target.long()
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = tfunc.to_pil_image(x)
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.data)


def get_data(data_name, dataset, test_transform=None, severity=1):
    if data_name == 'cifar10':
        ROOT_DIR = ROOT_DIR_10
    if data_name == 'cifar100':
        ROOT_DIR = ROOT_DIR_100
    data_path = os.path.join(ROOT_DIR, dataset+'.npy')
    label_path = os.path.join(ROOT_DIR, 'labels.npy')
    data = torch.tensor(np.transpose(np.load(data_path), (0,3,1,2)))
    labels = torch.tensor(np.load(label_path))
    start = 10000 * (severity - 1)
    #print(100*'#')
    #print(start, start+10000, len(data))
    data = data[start:start+10000]
    labels = labels[start:start+10000]
    test_data = DatasetFromTorchTensor(data, labels, transform=test_transform)

    return test_data


# def get_data(dataset, train_transform=None, test_transform=None, severity=1, ratio=0.03):
#     data_path = os.path.join(ROOT_DIR, dataset+'.npy')
#     label_path = os.path.join(ROOT_DIR, 'labels.npy')
#
#     data = torch.tensor(np.transpose(np.load(data_path), (0,3,1,2)))
#     labels = torch.tensor(np.load(label_path))
#
#     num_adapt = int(ratio*10000)
#     start = 10000*(severity-1)
#
#     data = data[start:start+10000]
#     labels = labels[start:start+10000]
#
#     data_train = data[:num_adapt]
#     label_train = labels[:num_adapt]
#     data_test = data[num_adapt:]
#     label_test = labels[num_adapt:]
#
#     train_data = DatasetFromTorchTensor(data_train, label_train, transform=train_transform)
#     test_data = DatasetFromTorchTensor(data_test, label_test, transform=test_transform)
#
#     return train_data, test_data


if __name__ =='__main__':
    train, test = get_data('snow')
    print(len(train), len(test))
