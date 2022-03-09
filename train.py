import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
from model import MLP, GCN, GCN_LPA
from utils import accuracy, one_hot_embedding, load_data
import torch_geometric.transforms as T


parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False,
                    help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=42, help='Random seed.')
parser.add_argument('--epochs', type=int, default=300,
                    help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=32,
                    help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0.5,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--lpaiters', type=int, default=10,
                    help='number of LPA iterations.')
parser.add_argument('--gcnnum', type=int, default=2,
                    help='number of GCN layers.')
parser.add_argument('--lamda', type=float, default=10,
                    help='weight of LP regularization.')

args = parser.parse_args()
np.random.seed(args.seed)
torch.cuda.manual_seed(args.seed)

dataset = Planetoid(root=r'data\Cora', name='Cora', transform=T.AddSelfLoops())
data = load_data(dataset)
data = data.cuda()
labels_for_lpa = one_hot_embedding(data.y, dataset.num_classes)


# model = MLP(dataset.num_node_features, args.hidden, dataset.num_classes, args.dropout).cuda()
# model = GCN(dataset.num_node_features, args.hidden, dataset.num_classes, args.dropout).cuda()
model = GCN_LPA(dataset.num_node_features, args.hidden, dataset.num_classes, args.dropout, data.num_edges,
                args.lpaiters, args.gcnnum).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
crition = nn.CrossEntropyLoss()

for epoch in range(args.epochs):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    out, yhat = model(data, data.train_mask)
    loss_train = crition(out[data.train_mask], data.y[data.train_mask]) \
                 + args.lamda * crition(yhat[data.train_mask], data.y[data.train_mask])
    acc_train = accuracy(out[data.train_mask], data.y[data.train_mask])
    loss_train.backward()
    optimizer.step()

    model.eval()
    out, yhat = model(data, data.val_mask)
    loss_val = crition(out[data.val_mask], data.y[data.val_mask])\
               + args.lamda * crition(yhat[data.val_mask], data.y[data.val_mask])
    acc_val = accuracy(out[data.val_mask], data.y[data.val_mask])
    print('Epoch: {:04d}'.format(epoch + 1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'time: {:.4f}s'.format(time.time() - t))

model.eval()
_, pred = model(data, data.test_mask)[0].max(dim=1)
correct = int(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
acc = correct / int(data.test_mask.sum())
print('Accuracy: {:.4f}'.format(acc))
