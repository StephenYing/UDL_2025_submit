import os
import gzip
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


def load_mnist_pkl(path='data/mnist.pkl.gz'):
    try:
        with gzip.open(path, 'rb') as f:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        return train_set, valid_set, test_set
    except FileNotFoundError:
        print(f"Error: {path} not found.")
        raise
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        raise


class PermutedMnistGenerator:
    def __init__(self, max_iter=10, data_path='data/mnist.pkl.gz'):
        train_set, valid_set, test_set = load_mnist_pkl(data_path)

        self.X_train = np.vstack([train_set[0], valid_set[0]]).astype(np.float32)
        self.Y_train = np.hstack([train_set[1], valid_set[1]]).astype(np.int64)
        self.X_test = test_set[0].astype(np.float32)
        self.Y_test = test_set[1].astype(np.int64)

        self.max_iter = max_iter
        self.cur_iter = 0
        self.num_classes = 10
        self.input_dim = self.X_train.shape[1]

        print(f"Permuted MNIST Generator: input_dim={self.input_dim}, "
              f"num_classes={self.num_classes}, "
              f"train_size={self.X_train.shape[0]}, test_size={self.X_test.shape[0]}")

    def get_dims(self):
        return self.input_dim, self.num_classes

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise StopIteration

        np.random.seed(self.cur_iter)
        perm_inds = np.arange(self.X_train.shape[1])
        np.random.shuffle(perm_inds)

        x_train_perm = self.X_train[:, perm_inds]
        x_test_perm = self.X_test[:, perm_inds]

        self.cur_iter += 1

        return (torch.from_numpy(x_train_perm),
                torch.from_numpy(self.Y_train),
                torch.from_numpy(x_test_perm),
                torch.from_numpy(self.Y_test))


class SplitMnistGenerator:
    def __init__(self, data_path='data/mnist.pkl.gz'):
        train_set, valid_set, test_set = load_mnist_pkl(data_path)

        self.X_train = np.vstack([train_set[0], valid_set[0]]).astype(np.float32)
        self.train_label = np.hstack([train_set[1], valid_set[1]]).astype(np.int64)
        self.X_test = test_set[0].astype(np.float32)
        self.test_label = test_set[1].astype(np.int64)

        self.sets = [(0,1), (2,3), (4,5), (6,7), (8,9)]
        self.max_iter = len(self.sets)
        self.cur_iter = 0
        self.num_classes_per_task = 2
        self.input_dim = self.X_train.shape[1]

        print(f"Split MNIST Generator: input_dim={self.input_dim}, "
              f"num_classes_per_task={self.num_classes_per_task}, "
              f"train_size={self.X_train.shape[0]}, test_size={self.X_test.shape[0]}")

    def get_dims(self):
        return self.input_dim, self.num_classes_per_task

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise StopIteration

        labels_in_task = self.sets[self.cur_iter]

        train_indices = np.where(np.isin(self.train_label, labels_in_task))[0]
        next_x_train = self.X_train[train_indices]
        original_train_labels = self.train_label[train_indices]
        next_y_train = np.zeros_like(original_train_labels)
        next_y_train[original_train_labels == labels_in_task[1]] = 1

        test_indices = np.where(np.isin(self.test_label, labels_in_task))[0]
        next_x_test = self.X_test[test_indices]
        original_test_labels = self.test_label[test_indices]
        next_y_test = np.zeros_like(original_test_labels)
        next_y_test[original_test_labels == labels_in_task[1]] = 1

        self.cur_iter += 1

        return (torch.from_numpy(next_x_train),
                torch.from_numpy(next_y_train),
                torch.from_numpy(next_x_test),
                torch.from_numpy(next_y_test))


class SimpleDataset(Dataset):
    def __init__(self, data, targets):
        self.data = data
        self.targets = targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]
