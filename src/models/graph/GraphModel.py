import torch.nn as nn
from torch_geometric.nn import SAGEConv, GATConv, GATv2Conv, TransformerConv, GINConv


class GraphTokenEncoder(nn.Module):
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
