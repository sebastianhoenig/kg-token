from torch import nn
from torch.utils.data import DataLoader
import torch
import tqdm
import wandb
from tqdm.notebook import tqdm

from src.models.language.LanguageModel import LLMWrapper
from torch_geometric.data import HeteroData


def train(llm_wrapper: LLMWrapper, gnn: nn.Module, graph: HeteroData, dataloader: DataLoader,
          optimizer: torch.optim.Optimizer, config: dict):
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

    wandb.watch(llm, log="all", log_freq=1)
    wandb.watch(gnn, log="all", log_freq=1)

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

            llm.get_input_embeddings().weight.data[user_token_id] = user_embedding
            llm.get_input_embeddings().weight.data[movie_token_id] = movie_embedding

            labels = input_tokens.clone().to(device)
            labels = labels.to(device)

            #  TODO: CHECK - is it correct to label like this? - target here is Yes/No + EOS token
            labels[target_mask == 0] = -100

            outputs = llm(input_tokens, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            optimizer.zero_grad()
            loss.requires_grad = True
            loss.backward()
            optimizer.step()

            example_ct += 1
            if example_ct % config['logging_steps'] == 0:
                train_log(loss, example_ct, epoch)

            total_loss += loss.detach().item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config['num_epochs']}, Loss: {avg_loss}")


def train_log(loss, example_ct, epoch):
    wandb.log({"epoch": epoch, "loss": loss}, step=example_ct)
    print(f"Loss after {str(example_ct).zfill(5)} examples: {loss:.3f}")












