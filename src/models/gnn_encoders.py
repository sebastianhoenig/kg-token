import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv, TransformerConv, GINConv
import torch.nn.functional as F


class SAGEConvTokenEncoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, use_bn: bool,
                 num_heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv((-1, -1), hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv((-1, -1), hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(SAGEConv((-1, -1), out_channels))
        self.dropout = dropout
        self.use_bn = use_bn

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATConvTokenEncoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, use_bn: bool,
                 num_heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATConv((-1, -1), hidden_channels, add_self_loops=False))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv((-1, -1), hidden_channels, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv((-1, -1), out_channels, add_self_loops=False))
        self.dropout = dropout
        self.use_bn = use_bn

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class GATv2ConvTokenEncoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, use_bn: bool,
                 num_heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GATv2Conv((-1, -1), hidden_channels, add_self_loops=False))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATv2Conv((-1, -1), hidden_channels, add_self_loops=False))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATv2Conv((-1, -1), out_channels, add_self_loops=False))
        self.dropout = dropout
        self.use_bn = use_bn

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


class TransformerConvTokenEncoder(nn.Module):
    def __init__(self, hidden_channels: int, out_channels: int, num_layers: int, dropout: float, use_bn: bool,
                 num_heads=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(TransformerConv((-1, -1), hidden_channels))
        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(TransformerConv((-1, -1), hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.convs.append(TransformerConv((-1, -1), out_channels))
        self.dropout = dropout
        self.use_bn = use_bn

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.use_bn:
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x


load_gnn_model = {
    'sage': SAGEConvTokenEncoder,
    'gat': GATConvTokenEncoder,
    'gatv2': GATv2ConvTokenEncoder,
    'transformer': TransformerConvTokenEncoder,
}