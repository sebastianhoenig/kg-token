from typing import Any

from torch import nn
from torch.utils.data import DataLoader
import torch
import tqdm
import wandb
from tqdm.notebook import tqdm
from itertools import islice
import torch.nn.functional as F

from torch_geometric.data import HeteroData
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from src.models.language.LanguageModel import LLMWrapper
from src.test.metrics import yes_no_accuracy


def train(llm_wrapper: LLMWrapper, gnn: nn.Module, graph: HeteroData, dataloader: DataLoader,
          optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler, config: Any):
    # Set GNN parameters to require gradients
    for param in gnn.parameters():
        param.requires_grad = True

    # Freeze all LLM parameters
    llm = llm_wrapper.get_llm()
    tokenizer = llm_wrapper.get_tokenizer()
    for param in llm.parameters():
        param.requires_grad = False  # Freezes the LLM parameters

    device = config['device']
    num_examples_per_epoch = config['num_examples_per_epoch']
    gnn.to(device)
    llm.to(device)
    graph = graph.to(device)

    USER_EMB = llm_wrapper.USER_EMB
    MOVIE_EMB = llm_wrapper.MOVIE_EMB

    example_ct = 0  # number of examples seen

    for epoch in tqdm(range(config['num_epochs']), desc="Epoch Progress"):

        total_loss = 0

        if num_examples_per_epoch == -1:
            # use all examples in the dataset
            num_examples_per_epoch = len(dataloader)
        else:
            num_examples_per_epoch = min(num_examples_per_epoch, len(dataloader))

        for batch in tqdm(islice(dataloader, num_examples_per_epoch), desc="Batch Progress", leave=False):

            batch_size = len(batch[0])

            input_tokens, target_mask, attention_masks, user_ids, movie_ids = batch

            input_tokens = input_tokens.to(device)
            target_mask = target_mask.to(device)
            attention_masks = attention_masks.to(device)
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)

            embedding = gnn(graph.x_dict, graph.edge_index_dict)

            batch_embeddings = []
            batch_attention_masks = []
            batch_labels = []

            base_embeddings = llm.get_input_embeddings().weight

            for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
                movie_embedding = embedding['movie'][movie_id].to(device)
                user_embedding = embedding['user'][user_id].to(device)

                user_token_id = tokenizer.convert_tokens_to_ids(USER_EMB)
                movie_token_id = tokenizer.convert_tokens_to_ids(MOVIE_EMB)

                # Create a modified embedding matrix
                modified_embeddings = base_embeddings.clone()
                modified_embeddings[user_token_id] = user_embedding
                modified_embeddings[movie_token_id] = movie_embedding

                # Embed input tokens using the modified embeddings
                input_embeddings = F.embedding(input_tokens[i], modified_embeddings)

                batch_embeddings.append(input_embeddings)
                batch_attention_masks.append(attention_masks[i])
                batch_labels.append(input_tokens[i].clone())

            batch_embeddings = torch.stack(batch_embeddings)
            batch_attention_masks = torch.stack(batch_attention_masks)
            batch_labels = torch.stack(batch_labels)

            # Forward pass through the LLM
            outputs = llm(
                inputs_embeds=batch_embeddings,  # Use custom embeddings
                attention_mask=batch_attention_masks,
                labels=batch_labels
            )

            # loss = outputs.loss
            logits = outputs.logits
            logits = logits[:, :-1, :].contiguous()

            batch_labels = batch_labels[:, 1:]
            target_mask = target_mask[:, 1:]

            # Compute the loss using PyTorch's CrossEntropyLoss
            batch_labels[target_mask == 0] = -100
            loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(logits.flatten(0, 1), batch_labels.flatten())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits, dim=-1)
            predictions = predictions[target_mask == 1]
            labels = batch_labels[target_mask == 1]
            predicted_tokens = tokenizer.convert_ids_to_tokens(predictions)
            target_tokens = tokenizer.convert_ids_to_tokens(labels)

            accuracy = yes_no_accuracy(target_tokens, predicted_tokens)['yes_no_accuracy']

            example_ct += batch_size
            num_examples_per_epoch += batch_size

            wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
            wandb.log({"epoch": epoch, "accuracy": accuracy}, step=example_ct)
            wandb.log({"epoch": epoch, "learning_rate": optimizer.param_groups[0]['lr']}, step=example_ct)
            #print_output_of_llm(logits, llm_wrapper, labels)

            total_loss += loss.detach().item()

            scheduler.step(loss.item())

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
