import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from layer import LPAconv


class MLP(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_feature, hidden)
        self.fc2 = nn.Linear(hidden, out_feature)
        self.relu = nn.ReLU()
        self.dropout_rate = dropout

    def forward(self, data):
        x = data.x
        x = self.fc1(x)
        x = self.relu(x)
        x = F.dropout(x, self.dropout_rate, training=self.training)

        x = self.fc2(x)

        return x


class GCN(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feature, hidden)
        self.conv2 = GCNConv(hidden, out_feature)
        self.dropout_rate = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class GCN_LPA(nn.Module):
    def __init__(self, in_feature, hidden, out_feature, dropout, num_edges, lpaiters, gcnnum):
        super(GCN_LPA, self).__init__()
        self.edge_weight = nn.Parameter(torch.ones(num_edges))
        gc = nn.ModuleList()
        gc.append(GCNConv(in_feature, hidden))
        for i in range(gcnnum-2):
            gc.append(GCNConv(hidden, hidden))
        gc.append(GCNConv(hidden, out_feature))
        self.gc = gc
        self.lpa = LPAconv(lpaiters)
        self.dropout_rate = dropout

    def forward(self, data, mask):
        x, edge_index, y = data.x, data.edge_index, data.y

        for i in range(len(self.gc)-1):
            x = self.gc[i](x, edge_index, self.edge_weight)
            x = F.relu(x)
            x = F.dropout(x, self.dropout_rate, training=self.training)
        x = self.gc[-1](x, edge_index, self.edge_weight)

        y_hat = self.lpa(y, edge_index, mask, self.edge_weight)

        return x, y_hat
