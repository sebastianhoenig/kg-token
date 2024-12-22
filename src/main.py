from typing import Any
import wandb
from torch.utils.data import DataLoader
import torch
import tqdm
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from src.utils.logging import log_train_to_wandb
from src.utils.seed import apply_seed
from src.models.gnn_llm import GraphTokenGPT
from src.models.gnn import GNNPipeline
from src.utils.metrics import get_accuracy_gnnllm, yes_no_accuracy, get_accuracy_gnn_binary_task, get_rmse_gnn_regression_task
from src.models.gnn_encoders import GATConvTokenEncoder
from torch_geometric.nn import to_hetero
from src.graph.Movielens100k import MovieLens
from src.data.Dataset import GraphQADataset
from src.utils.logging import log_test_to_wandb
from src.utils.checkpoints import save_model, load_model


def train_gnn(config: Any):
    apply_seed(0)

    #movielens = MovieLens(config)
    #movielens.create_graph()

    movielens = MovieLens(config)
    movielens.create_graph()

    # Freeze all LLM parameters
    device = config.device

    graph = movielens.train
    graph = graph.to(device)

    gnn = GNNPipeline(config, movielens.train.metadata())

    optimizer = torch.optim.AdamW(gnn.parameters(), lr=config.lr)

    for epoch in range(config.num_epochs):

        total_loss = 0

        probs, loss, labels = gnn(graph)
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

    return total_loss, train_acc, test_acc


def train_gnnllm(config: Any):
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

            res = get_accuracy_gnnllm(batch_labels, logits, target_mask, gnn_llm.tokenizer)

            total_loss += loss.detach().item()
            #scheduler.step(loss.item())

            example_ct += batch_size
            log_train_to_wandb(res, epoch, loss, optimizer, example_ct)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss}")

    print("Evaluating on Test set")
    save_model(gnn_llm, config)

    test_dataset = GraphQADataset(graph=movielens.test, config=config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    gnn_llm.eval()

    total_correct_yes_preds, total_correct_no_preds = 0, 0
    total_wrong_yes_preds, total_wrong_no_preds = 0, 0
    total_yes_targets, total_no_targets = 0, 0
    total_correct, total_items = 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            test_results = gnn_llm.inference(batch, graph)
            print("Predictions")
            print(test_results["predictions"])
            print("Answers")
            print(test_results["answers"])
            batch_res = yes_no_accuracy(test_results["answers"], test_results["predictions"])

            total_correct_yes_preds += batch_res['num_correct_yes_preds']
            total_correct_no_preds += batch_res['num_correct_no_preds']
            total_wrong_yes_preds += batch_res['num_wrong_yes_preds']
            total_wrong_no_preds += batch_res['num_wrong_no_preds']
            total_yes_targets += batch_res['num_yes_targets']
            total_no_targets += batch_res['num_no_targets']
            total_correct += batch_res['num_correct']
            total_items += batch_res['num_items']

    aggregated_res = {
        "num_correct": total_correct,
        "num_correct_yes_preds": total_correct_yes_preds,
        "num_correct_no_preds": total_correct_no_preds,
        "num_wrong_yes_preds": total_wrong_yes_preds,
        "num_wrong_no_preds": total_wrong_no_preds,
        "num_yes_targets": total_yes_targets,
        "num_no_targets": total_no_targets,
        "num_items": total_items,
    }

    log_test_to_wandb(aggregated_res)


def evaluate(config: Any):

    movielens = MovieLens(path=config.dataset_path, device=config.device)
    movielens.create_graph()

    test_dataset = GraphQADataset(graph=movielens.test, config=config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    device = config.device

    gnn = GATConvTokenEncoder(config.gnn_hidden_dim, config.llm_embedding_dim)
    gnn = to_hetero(gnn, movielens.train.metadata(), aggr='sum')
    gnn.to(device)

    graph = movielens.train
    graph = graph.to(device)

    gnn_llm = GraphTokenGPT(config, gnn)
    gnn_llm = load_model(gnn_llm, config)
    # Set model to evaluation mode
    gnn_llm.eval()

    total_correct_yes_preds, total_correct_no_preds = 0, 0
    total_wrong_yes_preds, total_wrong_no_preds = 0, 0
    total_yes_targets, total_no_targets = 0, 0
    total_correct, total_items = 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):

            test_results = gnn_llm.inference(batch, graph)
            print("Predictions")
            print(test_results["predictions"])
            print("Answers")
            print(test_results["answers"])
            batch_res = yes_no_accuracy(test_results["answers"], test_results["predictions"])

            total_correct_yes_preds += batch_res['num_correct_yes_preds']
            total_correct_no_preds += batch_res['num_correct_no_preds']
            total_wrong_yes_preds += batch_res['num_wrong_yes_preds']
            total_wrong_no_preds += batch_res['num_wrong_no_preds']
            total_yes_targets += batch_res['num_yes_targets']
            total_no_targets += batch_res['num_no_targets']
            total_correct += batch_res['num_correct']
            total_items += batch_res['num_items']

    aggregated_res = {
        "num_correct": total_correct,
        "num_correct_yes_preds": total_correct_yes_preds,
        "num_correct_no_preds": total_correct_no_preds,
        "num_wrong_yes_preds": total_wrong_yes_preds,
        "num_wrong_no_preds": total_wrong_no_preds,
        "num_yes_targets": total_yes_targets,
        "num_no_targets": total_no_targets,
        "num_items": total_items,
    }

    print(aggregated_res)

    log_test_to_wandb(aggregated_res)


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
