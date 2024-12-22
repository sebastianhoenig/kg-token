import torch
import torch.nn as nn
from src.models.gnn_encoders import load_gnn_model
from torch_geometric.nn import to_hetero


class GNNPipeline(nn.Module):
    def __init__(self, args, metadata):
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
        self.gnn = to_hetero(gnn, metadata, aggr=args.aggr)
        self.fc = nn.Linear(args.gnn_hidden_dim * 2, 1).to(self.device)  # *2 because of concatenation
        self.args = args

    def forward(self, graph):
        # Extract graph embeddings using the GNN
        graph_embeds = self.gnn(graph.x_dict, graph.edge_index_dict)

        # Edge index and labels for training
        edge_index = graph.edge_index_dict['user', 'likes', 'movie']
        edge_labels = graph.edge_labels_dict['user', 'likes', 'movie']

        # Pass through EdgeDecoder
        node_embeddings = torch.cat([graph_embeds['user'], graph_embeds['movie']], dim=0)

        src = node_embeddings[edge_index[0]]  # Shape: [num_edges, in_channels]
        dst = node_embeddings[edge_index[1]]  # Shape: [num_edges, in_channels]

        # Combine source and target node embeddings (e.g., concatenation)
        edge_embeddings = torch.cat([src, dst], dim=1)  # Shape: [num_edges, in_channels * 2]

        # Predict edge label probabilities
        logits = self.fc(edge_embeddings).squeeze()  # Shape: [num_edges]
        probabilities = torch.sigmoid(logits)  # Probabilities for label 1

        # Compute binary cross-entropy loss
        criterion = nn.BCELoss()
        loss = criterion(probabilities, edge_labels.float())

        return probabilities, loss, edge_labels.float()

    def inference(self, graph):
        # Extract graph embeddings using the GNN
        graph_embeds = self.gnn(graph.x_dict, graph.edge_index_dict)

        # Edge index for inference
        edge_index = graph.edge_index_dict['user', 'likes', 'movie']  # Example edge type

        # Pass through EdgeDecoder
        node_embeddings = torch.cat([graph_embeds['user'], graph_embeds['movie']], dim=0)
        src = node_embeddings[edge_index[0]]  # Shape: [num_edges, in_channels]
        dst = node_embeddings[edge_index[1]]  # Shape: [num_edges, in_channels]

        # Combine source and target node embeddings (e.g., concatenation)
        edge_embeddings = torch.cat([src, dst], dim=1)  # Shape: [num_edges, in_channels * 2]

        # Predict edge label probabilities
        logits = self.fc(edge_embeddings).squeeze()  # Shape: [num_edges]
        probabilities = torch.sigmoid(logits)

        return probabilities, graph.edge_labels_dict['user', 'likes', 'movie'].float()
