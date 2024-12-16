import torch
from src.train.train import train


if __name__ == '__main__':
    from argparse import Namespace

    args = Namespace(
        # PATHS
        llm_model_path="/Users/sebastian/University/Master/third term/sem-proj/kg-token/weights/gpt2-medium",
        dataset_path="/Users/sebastian/University/Master/third term/sem-proj/kg-token/data/ml-100k/",

        # GNN
        gnn_aggr="mean",
        gnn_model="gat",
        gnn_hidden_dim=128,
        gnn_out_dim=128,
        gnn_num_layers=2,
        gnn_dropout=0.0,
        gnn_num_heads=1,
        gnn_use_bn=False,

        # PROJECT
        projection_type="linear",
        mlp_channels=[1024, 2048, 4096],
        mlp_dropout=0.0,
        mlp_use_bn=False,

        # OTHER
        lr=0.0001,  # 1e-5,
        min_lr=5e-6,
        wd=0.05,
        seed=0,
        batch_size=128,
        num_epochs=1,
        output_dir='output',
        name='updating-pipeline-using-gnn-llm-no-llm-wrapper',
        device="cuda" if torch.cuda.is_available() else "cpu",
        mini_epoch_batches=1,
        llm_embedding_dim=1024,
        show_grads=False,
        USER_EMB="<USER>",
        MOVIE_EMB="<MOVIE>",
    )

    train(args)
