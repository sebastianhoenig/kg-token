"""
GNN baseline model. The encoder gnn type can be changed by changing the gnn_model parameter in the args, but the decoder
part is always a simple feedforward neural network.
"""

import torch
import torch.nn as nn
from src.models.gnn_encoders import load_gnn_model
from torch_geometric.nn import to_hetero


class GNNPipeline(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.device = args.device
        gnn = load_gnn_model[args.gnn_model](
            hidden_channels=args.gnn_hidden_dim,
            out_channels=args.gnn_out_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            use_bn=args.gnn_use_bn,
            num_heads=args.gnn_num_heads,
        ).to(self.device)
        self.gnn = to_hetero(gnn, data.metadata, aggr=args.gnn_aggr)
        self.fc1 = nn.Linear(args.gnn_hidden_dim * 2, args.gnn_hidden_dim).to(self.device)  # *2 because of concatenation
        self.fc2 = nn.Linear(args.gnn_hidden_dim, 1).to(self.device)
        self.args = args

    def forward(self, graph):
        # Extract graph embeddings using the GNN
        graph_embeds = self.gnn(graph.x_dict, graph.edge_index_dict)

        # Edge index and labels for training
        edge_index = graph.edge_index_dict['user', 'likes', 'movie']
        edge_labels = graph.edge_label_dict['user', 'likes', 'movie']

        # Pass through EdgeDecoder
        node_embeddings = torch.cat([graph_embeds['user'], graph_embeds['movie']], dim=0)

        src = node_embeddings[edge_index[0]]  # Shape: [num_edges, in_channels]
        dst = node_embeddings[edge_index[1]]  # Shape: [num_edges, in_channels]

        # Combine user and movie node
        edge_embeddings = torch.cat([src, dst], dim=1)  # Shape: [num_edges, in_channels * 2]

        # Predict edge label probabilities
        logits = self.fc2(self.fc1(edge_embeddings).relu()).squeeze()  # Shape: [num_edges]

        # Compute binary cross-entropy loss
        if self.args.rating_type == 'binary':
            criterion = nn.BCELoss()
            probabilities = torch.sigmoid(logits)  # Probabilities for label 1
            loss = criterion(probabilities, edge_labels.float())
            return probabilities, loss, edge_labels.float()
        else:
            criterion = nn.MSELoss()
            loss = criterion(logits, edge_labels.float())

        return logits, loss, edge_labels.float()

    def inference(self, graph):
        graph_embeds = self.gnn(graph.x_dict, graph.edge_index_dict)

        edge_index = graph.edge_index_dict['user', 'likes', 'movie']

        node_embeddings = torch.cat([graph_embeds['user'], graph_embeds['movie']], dim=0)
        src = node_embeddings[edge_index[0]]  # Shape: [num_edges, in_channels]
        dst = node_embeddings[edge_index[1]]  # Shape: [num_edges, in_channels]

        edge_embeddings = torch.cat([src, dst], dim=1)  # Shape: [num_edges, in_channels * 2]

        logits = self.fc2(self.fc1(edge_embeddings).relu()).squeeze()  # Shape: [num_edges]
        if self.args.rating_type == 'binary':
            probabilities = torch.sigmoid(logits)
            return probabilities, graph.edge_label_dict['user', 'likes', 'movie'].float()
        else:
            predictions = logits.clamp(0, 5)  # Clamp predictions to the rating range
            return predictions, graph.edge_label_dict['user', 'likes', 'movie'].float()
