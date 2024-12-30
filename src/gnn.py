from typing import Any
import torch
from src.utils.seed import apply_seed
from src.models.gnn import GNNPipeline
from src.utils.metrics import get_accuracy_gnn_binary_task, get_rmse_gnn_regression_task
from src.graph.Movielens100k import MovieLens
from tqdm import tqdm
import wandb


def train_gnn(config: Any):
    wandb.init(project={config.project_name}, name=f"{config.gnn_model}-{config.gnn_hidden_dim}-{config.gnn_num_layers}-{config.gnn_dropout}-{config.gnn_aggr}", config=config)
    apply_seed(0)

    movielens = MovieLens(config)
    movielens.create_graph()

    # Freeze all LLM parameters
    device = config.device

    graph = movielens.train
    graph = graph.to(device)

    gnn = GNNPipeline(config, movielens.train.metadata())

    optimizer = torch.optim.AdamW(gnn.parameters(), lr=config.lr)

    for epoch in tqdm(range(config.num_epochs)):

        total_loss = 0

        probs, loss, labels = gnn(graph)
        acc = get_accuracy_gnn_binary_task(probs, labels)['acc']

        wandb.log({"loss": loss}, step=epoch)
        wandb.log({"accuracy": acc}, step=epoch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item()

    if config.rating_type == 'binary':
        train_acc = get_accuracy_gnn_binary_task(probs, labels)['acc']
    else:
        train_acc = get_rmse_gnn_regression_task(probs, labels)['rmse']
    gnn.eval()
    test_graph = movielens.test
    test_graph = test_graph.to(device)

    probs, labels = gnn.inference(test_graph)

    if config.rating_type == 'binary':
        test_acc = get_accuracy_gnn_binary_task(probs, labels)['acc']
    else:
        test_acc = get_rmse_gnn_regression_task(probs, labels)['rmse']

    wandb.log({"test-accuracy": test_acc}, step=epoch)

    # save gnn
    #torch.save(gnn.state_dict(), 'pretrained-gnn.pt')

    wandb.finish()

    return total_loss, train_acc, test_acc
