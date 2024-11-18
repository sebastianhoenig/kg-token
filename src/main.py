import torch
from torch.utils.data import DataLoader

from src.graph.Movielens100k import MovieLens
from src.models.graph.GraphModel import GraphTokenEncoder
from torch_geometric.nn import to_hetero
import torch_geometric.transforms as T
from src.models.language.LanguageModel import LLMWrapper
from src.train.Config import TrainingConfig
from src.train.train import train
from src.data.Dataset import GraphQADataset


if __name__ == '__main__':
    EMBEDDING_DIM = 768
    USER_EMB = "<USER>"  # TODO: All in config...
    MOVIE_EMB = "<MOVIE>"
    MODEL_NAME = 'gpt2'

    movielens = MovieLens('/Users/sebastian/University/Master/third term/sem-proj/kg-token/data/ml-100k/', 'cpu')
    movielens.create_graph()

    gnn = GraphTokenEncoder(EMBEDDING_DIM, EMBEDDING_DIM)
    gnn = to_hetero(gnn, movielens.train.metadata(), aggr='sum')

    llm_wrapper = LLMWrapper(model_name=MODEL_NAME, user_emb=USER_EMB, movie_emb=MOVIE_EMB)

    transform = T.RandomLinkSplit(
        num_val=0.1,  # 10% of edges for validation
        num_test=0.1,  # 10% of edges for test
        edge_types=('user', 'likes', 'movie'),
        rev_edge_types=('movie', 'rev_likes', 'user')
    )
    train_edges, val_edges, test_edges = transform(movielens.data)

    train_dataset = GraphQADataset(graph=train_edges, llm_wrapper=llm_wrapper, max_tokens=25)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    config = TrainingConfig(
        num_epochs=10,
        learning_rate=0.001,
        logging_steps=2,
        evaluation_steps=2,
        model_save_path='./model',
        optimizer='adam',
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    train(llm_wrapper, gnn, movielens.train, train_dataloader, config)
