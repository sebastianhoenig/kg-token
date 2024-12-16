from typing import Any
import wandb
from torch import nn
from torch.utils.data import DataLoader
import torch
import tqdm
from tqdm.notebook import tqdm

from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from src.models.llm import LLM
from src.train.utils import prepare_inputs, get_batch_accuracy
from src.utils.logging import log_to_wandb
from src.utils.seed import apply_seed
from src.models.gnn_llm import GraphTokenGPT
from src.test.metrics import get_binary_batch_metrics
from src.models.gnn import SAGEConvTokenEncoder, GATConvTokenEncoder, GATv2ConvTokenEncoder, TransformerConvTokenEncoder, GINConvTokenEncoder
from torch_geometric.nn import to_hetero
from src.graph.Movielens100k import MovieLens
from src.data.Dataset import GraphQADataset


def train(config: Any):
    apply_seed(0)

    wandb.init(project="kg-token", name=config.name, config=config)

    movielens = MovieLens(path=config.dataset_path, device=config.device)
    movielens.create_graph()

    # Freeze all LLM parameters
    device = config.device
    gnn = GATConvTokenEncoder(config.gnn_hidden_dim, config.llm_embedding_dim)
    gnn = to_hetero(gnn, movielens.train.metadata(), aggr='sum')
    gnn.to(device)

    graph = movielens.train
    graph = graph.to(device)

    train_dataset = GraphQADataset(graph=movielens.train, config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_dataset = GraphQADataset(graph=movielens.test, config=config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    optimizer = torch.optim.AdamW(gnn.parameters(), lr=config.lr)

    example_ct = 0  # number of examples seen

    gnn_llm = GraphTokenGPT(config, gnn)
    for epoch in tqdm(range(config.num_epochs), desc="Epoch Progress"):

        total_loss = 0

        for batch in tqdm(train_loader, desc="Batch Progress", leave=False):

            batch_size = len(batch['question'])

            logits, loss, batch_labels, target_mask = gnn_llm(batch, graph)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            res = get_batch_accuracy(batch_labels, logits, target_mask, gnn_llm.tokenizer)

            total_loss += loss.detach().item()
            #scheduler.step(loss.item())

            example_ct += batch_size
            log_to_wandb(res, epoch, example_ct, loss, optimizer)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {avg_loss}")


def print_output_of_llm(logits, llm_wrapper, labels):
    # Get the predicted token ids
    predicted_token_ids = torch.argmax(logits, dim=-1)

    # Get the tokenizer from the LLMWrapper
    tokenizer = llm_wrapper.get_tokenizer()

    # get the indeces where labels are not -100
    mask = labels != -100

    predicted_token_ids = predicted_token_ids[mask]
    labels = labels[mask]
    # Convert the token ids to tokens
    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_token_ids.squeeze().tolist())
    target_tokens = tokenizer.convert_ids_to_tokens(labels.squeeze().tolist())

    # Print the predicted tokens
    print("Predicted tokens:")
    print(predicted_tokens)

    # Print the target tokens
    print("Target tokens:")
    print(target_tokens)


def check_updates(new_params, initial_params):
    for new_param, initial_param in zip(new_params.parameters(), initial_params):
        if not torch.equal(new_param.data, initial_param.data):  # Checks if parameter value changed
            print(f"Parameter {new_param} has changed from {initial_param} to {new_param}")
            return True
    return False


def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
