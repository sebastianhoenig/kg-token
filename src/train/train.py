from typing import Any

from torch import nn
from torch.utils.data import DataLoader
import torch
import tqdm
import wandb
from tqdm.notebook import tqdm
import torch.nn.functional as F

from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from src.models.language.LanguageModel import LLMWrapper


def train(llm_wrapper: LLMWrapper, gnn: nn.Module, graph: HeteroData, dataloader: DataLoader,
          optimizer: torch.optim.Optimizer, config: Any):
    # Set GNN parameters to require gradients
    for param in gnn.parameters():
        param.requires_grad = True

    # Freeze all LLM parameters
    llm = llm_wrapper.get_llm()
    for param in llm.parameters():
        param.requires_grad = False  # Freezes the LLM parameters

    device = config['device']
    gnn.to(device)
    llm.to(device)
    graph = graph.to(device)

    USER_EMB = llm_wrapper.USER_EMB
    MOVIE_EMB = llm_wrapper.MOVIE_EMB

    example_ct = 0  # number of examples seen

    for epoch in tqdm(range(config['num_epochs']), desc="Epoch Progress"):

        total_loss = 0
        for batch in tqdm(dataloader, desc="Batch Progress", leave=False):
            input_tokens, target_mask, attention_mask, user_id, movie_id = batch

            input_tokens = input_tokens.to(device)
            target_mask = target_mask.to(device)
            attention_mask = attention_mask.to(device)
            user_id = user_id.to(device)
            movie_id = movie_id.to(device)

            embedding = gnn(graph.x_dict, graph.edge_index_dict)

            movie_embedding = embedding['movie'][movie_id].to(device)
            user_embedding = embedding['user'][user_id].to(device)

            user_token_id = llm_wrapper.get_tokenizer().convert_tokens_to_ids(USER_EMB)
            movie_token_id = llm_wrapper.get_tokenizer().convert_tokens_to_ids(MOVIE_EMB)

            labels = input_tokens.clone().to(device)
            labels = labels.to(device)

            base_embeddings = llm.get_input_embeddings().weight

            # Create a modified embedding matrix
            modified_embeddings = base_embeddings.clone()
            modified_embeddings[user_token_id] = user_embedding
            modified_embeddings[movie_token_id] = movie_embedding

            # Embed input tokens using the modified embeddings
            input_embeddings = F.embedding(input_tokens, modified_embeddings)

            # Forward pass through the LLM
            outputs = llm(
                inputs_embeds=input_embeddings,  # Use custom embeddings
                attention_mask=attention_mask,
                labels=labels
            )
            # loss = outputs.loss
            logits = outputs.logits

            # Compute the loss using PyTorch's CrossEntropyLoss
            labels[target_mask == 0] = -100
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.squeeze(), labels.squeeze())

            if example_ct == 0:
                wandb.watch(llm, log="all", log_freq=10000)
                wandb.watch(gnn, log="all", log_freq=10000)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            example_ct += 1
            if example_ct % 10 == 0:
                wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
                #print_output_of_llm(logits, llm_wrapper, labels)

            total_loss += loss.detach().item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {avg_loss}")

    wandb.finish()


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
