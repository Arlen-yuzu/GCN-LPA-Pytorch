import numpy as np
import scipy.sparse as sp
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid


def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(dataset):
    data = dataset[0]
    train_mask = torch.zeros(data.num_nodes).to(torch.bool)
    val_mask = torch.zeros(data.num_nodes).to(torch.bool)
    test_mask = torch.zeros(data.num_nodes).to(torch.bool)
    for i in range(0, int(0.6 * data.num_nodes)):
        train_mask[i] = True
    for i in range(int(0.6 * data.num_nodes), int(0.8 * data.num_nodes)):
        val_mask[i] = True
    for i in range(int(0.8 * data.num_nodes), data.num_nodes):
        test_mask[i] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    return data


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)
