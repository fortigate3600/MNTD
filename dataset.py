import copy
import gzip
import struct

import numpy as np
import h5py
import torchvision.datasets

import constants
import torch
from torch.utils.data import TensorDataset, DataLoader


def flatten(data):
    """ flattens data after the first dimension. We organize samples by row: (num_samples, num_features) """
    data = data.reshape((data.shape[0], -1))
    feat_size = data.shape[1]
    return data, feat_size


def compute_mean_std(data):
    """ compute mean and stdev on Dataset of torch tensors """
    loader = DataLoader(data, batch_size=10, shuffle=False)
    mean, std = 0., 0.
    count = 0
    for data in loader:
        mean += data.mean().sum()
        std += data.std().sum()
        count += data.shape[0]
    return mean/count, std/count


def load_cat_noncat():
    # load train set
    train_dataset = h5py.File(constants.data_dir / 'train_catvnoncat.h5', "r")
    train_x = np.array(train_dataset["train_set_x"], dtype=np.float32)
    train_y = np.array(train_dataset["train_set_y"], dtype=np.float32)

    # load test set
    test_dataset = h5py.File(constants.data_dir / 'test_catvnoncat.h5', "r")
    test_x = np.array(test_dataset["test_set_x"], dtype=np.float32)
    test_y = np.array(test_dataset["test_set_y"], dtype=np.float32)

    # load classes
    classes = np.array(test_dataset["list_classes"])  # the list of classes

    train_y = train_y.reshape((train_y.shape[0], 1))
    test_y = test_y.reshape((test_y.shape[0], 1))

    return torch.from_numpy(train_x), torch.from_numpy(train_y), \
           torch.from_numpy(test_x), torch.from_numpy(test_y), \
           classes


def mnist_unpack(fname):
    with gzip.open(fname, 'rb') as f:
        magic, size = struct.unpack(">II", f.read(8))
        nrows, ncols = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.dtype(np.uint8).newbyteorder('>'))
        data = data.reshape((size, nrows, ncols))
        return data


def load_mnist():
    train = torchvision.datasets.MNIST(constants.data_dir, train=True, download=True)
    test = torchvision.datasets.MNIST(constants.data_dir, train=False, download=True)
    return train.data, train.targets, test.data, test.targets, train.classes


def get_dataset(dataset_name, bs=128, normalize_to_mean_std=False):
    train_x, train_y, test_x, test_y, classes = load_mnist() if dataset_name == 'mnist' else load_cat_noncat()
    # get shape of images, dropping the first dimension which is the number of samples
    data_shape = train_x.shape[1:]
    std, mean = None, None
    if normalize_to_mean_std:
        # normalize to 0 mean 1 stdev
        std, mean = torch.std_mean(test_x.float(), dim=0)
        train_x = (train_x - mean) / std
        test_x = (test_x - mean) / std
    else:
        # normalize to [0, 1]
        train_x, test_x = train_x/255., test_x/255.
    # flatten dataset
    train_x, feat_size = flatten(train_x)
    test_x, _ = flatten(test_x)
    trainset = TensorDataset(train_x, train_y)
    testset = TensorDataset(test_x, test_y)
    dataset = Dataset(dataset_name, DataLoader(trainset, batch_size=bs, shuffle=True), DataLoader(testset, batch_size=bs),
                      mean, std, orig_shape=data_shape, flattened_shape=feat_size, class_num=len(classes))
    return dataset


class Dataset:
    def __init__(self, name, trainloader, testloader, mean, std, orig_shape, flattened_shape, class_num):
        self.name = name
        self.trainloader = trainloader
        self.testloader = testloader
        self.mean = mean
        self.std = std
        self.orig_shape = orig_shape
        self.flattened_shape = flattened_shape
        self.class_num = class_num
        self.clean_data = None

    def is_grayscale(self):
        return self.name == 'mnist'

    def unnormalize_data(self, data):
        data = self.unflatten(data)
        data = (data * self.std) + self.mean if self.std is not None else data * 255.
        return data.long()

    def get_sub_testloader(self, max_items):
        """ return testloader with only max_items samples from the testset """
        testset = TensorDataset(*self.get_test_data(max_items))
        return DataLoader(testset, batch_size=self.testloader.batch_size)

    def get_test_data(self, max_items=None):
        """ returns max_items samples (X ,Y) from testset """
        test_data = self.testloader.dataset.tensors
        max_items = min(len(test_data[0]), max_items) if max_items is not None else len(test_data[0])
        return test_data[0][:max_items], test_data[1][:max_items]

    def get_train_data(self, max_items=None):
        """ returns max_items samples (X ,Y) from trainset """
        train_data = self.trainloader.dataset.tensors
        max_items = min(len(train_data[0]), max_items) if max_items is not None else len(train_data[0])
        return train_data[0][:max_items], train_data[1][:max_items]

    def poison_data(self, delta, poison_indices):
        """ update the given indices with the poison perturbation delta """
        self.clean_data = self.trainloader.dataset.tensors[0][poison_indices]
        self.trainloader.dataset.tensors[0][poison_indices] += delta

    def get_unnormalized_testset(self):
        test_x = self.get_test_data()[0]
        return self.unnormalize_data(test_x)

    def unflatten(self, data):
        return data.reshape((-1, *self.orig_shape))
