from typing import Callable, Optional
from torch_geometric.typing import Adj, OptTensor

import torch
from torch import Tensor
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm


class LPAconv(MessagePassing):
    def __init__(self, num_layers: int):
        super(LPAconv, self).__init__(aggr='add')
        self.num_layers = num_layers

    def forward(
            self, y: Tensor, edge_index: Adj, mask: Optional[Tensor] = None,
            edge_weight: OptTensor = None,
            post_step: Callable = lambda y: y.clamp_(0., 1.)
    ) -> Tensor:

        if y.dtype == torch.int64:
            y = F.one_hot(y.view(-1)).to(torch.float)

        out = y
        if mask is not None:
            out = torch.zeros_like(y)
            out[mask] = y[mask]

        if isinstance(edge_index, SparseTensor) and not edge_index.has_value():
            edge_index = gcn_norm(edge_index, add_self_loops=False)
        elif isinstance(edge_index, Tensor) and edge_weight is None:
            edge_index, edge_weight = gcn_norm(edge_index, num_nodes=y.size(0),
                                               add_self_loops=False)

        for _ in range(self.num_layers):
            # propagate_type: (y: Tensor, edge_weight: OptTensor)
            out = self.propagate(edge_index, x=out, edge_weight=edge_weight,
                                 size=None)
            # out = post_step(out)
        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor = None) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor = None) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)
