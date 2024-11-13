from torch import nn
from torch.utils.data import DataLoader
import torch
import tqdm
from tqdm import tqdm

from src.models.language.LanguageModel import LLMWrapper
from src.train.Config import TrainingConfig
from torch_geometric.data import HeteroData


def train(llm_wrapper: LLMWrapper, gnn: nn.Module, graph: HeteroData, dataloader: DataLoader, config: TrainingConfig):
    device = config.device
    graph = graph.to(device)

    # Set GNN parameters to require gradients
    for param in gnn.parameters():
        param.requires_grad = True

    # Freeze all LLM parameters
    llm = llm_wrapper.get_llm()
    for param in llm.parameters():
        param.requires_grad = False  # Freezes the LLM parameters

    #params = {k: v.to(device) for k, v in llm.state_dict().items()}

    gnn.to(device)
    llm.to(device)

    USER_EMB = llm_wrapper.USER_EMB
    MOVIE_EMB = llm_wrapper.MOVIE_EMB

    # TODO function to get optimizer
    optimizer = torch.optim.Adam(gnn.parameters(), lr=config.learning_rate)

    for epoch in tqdm(range(config.num_epochs), desc="Epoch Progress"):

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

            # Update embeddings without in-place modifications
            #params['transformer.wte.weight'][user_token_id] = user_embedding
            #params['transformer.wte.weight'][movie_token_id] = movie_embedding

            llm.get_input_embeddings().weight.data[user_token_id] = user_embedding
            llm.get_input_embeddings().weight.data[movie_token_id] = movie_embedding
            # Load updated params back into the model
            #llm.load_state_dict(params)

            labels = input_tokens.clone().to(device)
            labels = labels.to(device)

            #  TODO: CHECK - is it correct to label like this? - target here is Yes/No + EOS token
            labels[target_mask == 0] = -100

            outputs = llm(input_tokens, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            total_loss += loss.detach().item()

            optimizer.zero_grad()
            loss.requires_grad = True
            loss.backward()  # Got issue here
            optimizer.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss}")


















