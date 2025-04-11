import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class ECALayer(nn.Module):
    def __init__(self, in_channels, k_size=3, dropout=0.7):
        dropout=0.3
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):

        if x.dim() != 2:
            raise ValueError(f"Expected input to be 2D tensor, but got {x.dim()}D tensor.")

        residual = x
        y = self.avg_pool(x.unsqueeze(1))  # [num_nodes, 1, channels]
        y = self.conv(y)  # [num_nodes, 1, channels]
        y = self.sigmoid(y)  # [num_nodes, 1, channels]
        y = y.squeeze(1)  # [num_nodes, channels]
        y = self.dropout(y)  # Dropout after BatchNorm
        return x * y + residual
class UNIFY(nn.Module):
    def __init__(self, nfeat, nhid, k_size, nclass, dropout, heads, layers):
        super(UNIFY, self).__init__()
        self.layers = layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.ecas = nn.ModuleList()

        self.convs.append(GATv2Conv(nfeat, nhid, heads=heads, dropout=dropout, concat=True))
        self.bns.append(nn.BatchNorm1d(nhid * heads))
        self.ecas.append(ECALayer(nhid * heads, k_size, dropout))

        for _ in range(layers - 1):
            self.convs.append(GATv2Conv(nhid * heads, nhid, heads=heads, dropout=dropout, concat=True))
            self.bns.append(nn.BatchNorm1d(nhid * heads))
            self.ecas.append(ECALayer(nhid * heads, k_size, dropout))

        self.conv_out = GATv2Conv(nhid * heads, nclass, heads=heads, dropout=dropout, concat=False)
        self.bn_out = nn.BatchNorm1d(nclass)
        self.eca_out = ECALayer(nclass, k_size, dropout)

        self.dropout = dropout

    def forward(self, x, edge_index):
        for i in range(self.layers):
            x = F.dropout(x, self.dropout, training=self.training)
            x = self.convs[i](x, edge_index)
            x = self.ecas[i](x)
            x = self.bns[i](x)
            x = F.elu(x)

        x = self.conv_out(x, edge_index)
        x = self.bn_out(x)
        x = self.eca_out(x)
        return x

