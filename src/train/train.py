from torch import nn
from torch.utils.data import DataLoader
import torch
import tqdm
from tqdm import tqdm

from models.language.LanguageModel import LLMWrapper
from src.train.Config import TrainingConfig
from torch_geometric.data import HeteroData


def train(llm_wrapper: LLMWrapper, gnn: nn.Module, graph: HeteroData, dataloader: DataLoader, config: TrainingConfig):
    device = config.device
    # TODO function to get optimizer
    optimizer = torch.optim.Adam(gnn.parameters(), lr=config.learning_rate)
    torch.autograd.set_detect_anomaly(True)
    # Set GNN parameters to require gradients
    for param in gnn.parameters():
        param.requires_grad = True

    # Freeze all LLM parameters
    llm = llm_wrapper.get_llm()
    for param in llm.parameters():
        param.requires_grad = False  # Freezes the LLM parameters

    gnn.to(device)
    llm.to(device)

    for epoch in tqdm(range(config.num_epochs)):

        total_loss = 0
        for batch in dataloader:
            input_tokens, target_mask, attention_mask, user_id, movie_id = batch

            input_tokens = input_tokens.to(device)
            target_mask = target_mask.to(device)
            attention_mask = attention_mask.to(device)
            user_id = user_id.to(device)
            movie_id = movie_id.to(device)

            embedding = gnn(graph.x_dict, graph.edge_index_dict)
            movie_embedding = embedding['movie'][movie_id]
            user_embedding = embedding['user'][user_id]


            #old_emb_movie = llm_wrapper.get_llm().transformer.wte.weight[llm_wrapper.get_tokenizer().convert_tokens_to_ids('<MOVIE>')]
            # Replace the movie and user embeddings in the llm embeddings
            llm_wrapper.update_embeddings(user_embedding, movie_embedding)
            #new_emb_movie = llm.transformer.wte.weight[llm_wrapper.get_tokenizer().convert_tokens_to_ids('<MOVIE>')]
            labels = input_tokens.clone()
            labels = labels.to(device)

            #  TODO: CHECK - is it correct to label like this? - target here is Yes/No + EOS token
            labels[target_mask == 0] = -100

            outputs = llm(input_tokens, attention_mask=attention_mask, labels=labels)

            loss = outputs.loss
            total_loss += loss.detach().item()

            optimizer.zero_grad()
            loss.backward()  # Got issue here
            optimizer.step()
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss}")


















