import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv, TransformerConv, GINConv
import torch.nn.functional as F

class SAGEConvTokenEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv((-1, -1), hidden_channels)
        self.conv2 = SAGEConv((-1, -1), hidden_channels)
        self.projection = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.projection(x)
        return x


class GATConvTokenEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv3 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.conv4 = GATConv((-1, -1), hidden_channels, add_self_loops=False)
        self.projection = nn.Linear(hidden_channels, out_channels)
        #self.projection = nn.Sequential(
        #    nn.Linear(hidden_channels, (out_channels+hidden_channels)//2),
        #    nn.ReLU(),
        #    nn.Sigmoid(),
        #    nn.Linear((out_channels+hidden_channels)//2, out_channels))

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)#.relu()
        #x = self.bn1(x)
        #x = self.bn2(x)
        x = self.projection(x)
        return x


class GATv2ConvTokenEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
        self.conv2 = GATv2Conv((-1, -1), hidden_channels, add_self_loops=False)
        self.projection = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.projection(x)
        return x


class TransformerConvTokenEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = TransformerConv((-1, -1), hidden_channels)
        self.conv2 = TransformerConv((-1, -1), hidden_channels)
        self.projection = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.projection(x)
        return x


class GINConvTokenEncoder(nn.Module):
    def __init__(self, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GINConv(nn.Linear(-1, hidden_channels))
        self.conv2 = GINConv(nn.Linear(-1, hidden_channels))
        self.projection = nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index).relu()
        x = self.projection(x)
        return x


def get_gnn_model(model: str, hidden_dim: int, out_dim: int):
    if model == "SAGE":
        return SAGEConvTokenEncoder(hidden_dim, out_dim)
    elif model == "GAT":
        return GATConvTokenEncoder(hidden_dim, out_dim)
    elif model == "GATv2":
        return GATv2ConvTokenEncoder(hidden_dim, out_dim)
    elif model == "Transformer":
        return TransformerConvTokenEncoder(hidden_dim, out_dim)
    elif model == "GIN":
        return GINConvTokenEncoder(hidden_dim, out_dim)
    else:
        raise ValueError(f"Invalid model name: {model}")