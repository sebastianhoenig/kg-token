import torch
from torch.utils.data import DataLoader

from src.graph.Movielens100k import MovieLens
from src.models.gnn import GATConvTokenEncoder
from torch_geometric.nn import to_hetero
from src.models.llm import LLM
from src.train.train import train
from src.data.Dataset import GraphQADataset


if __name__ == '__main__':
    EMBEDDING_DIM = 768
    USER_EMB = "<USER>"  # TODO: All in config...
    MOVIE_EMB = "<MOVIE>"
    MODEL_NAME = 'gpt2'

    movielens = MovieLens('/Users/sebastian/University/Master/third term/sem-proj/kg-token/data/ml-100k/', 'cpu', 'rating')
    movielens.create_graph()

    gnn = GATConvTokenEncoder(EMBEDDING_DIM, EMBEDDING_DIM)
    gnn = to_hetero(gnn, movielens.train.metadata(), aggr='sum')

    llm_wrapper = LLM(model_name=MODEL_NAME, user_emb=USER_EMB, movie_emb=MOVIE_EMB)

    train_dataset = GraphQADataset(graph=movielens.test, llm_wrapper=llm_wrapper, max_tokens=25, edge_type='binary', random=False)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

    config = dict(
        num_epochs=10,
        learning_rate=0.001,
        logging_steps=2,
        evaluation_steps=2,
        model_save_path='./model',
        optimizer='adam',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_examples_per_epoch=-1,
        USER_EMB="<USER>",
        MOVIE_EMB="<MOVIE>",
    )

    optimizer = torch.optim.Adam(gnn.parameters(), lr=config['learning_rate'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    train(llm_wrapper, gnn, movielens.train, train_dataloader, optimizer, scheduler, config)
