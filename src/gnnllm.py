from typing import Any
import wandb
from torch.utils.data import DataLoader
import torch
import tqdm
from tqdm.notebook import tqdm
from src.utils.logging import log_train_to_wandb
from src.utils.seed import apply_seed
from src.models.gnn_llm import GraphTokenGPT
from src.utils.metrics import get_accuracy_gnnllm, yes_no_accuracy
from src.graph.Movielens100k import MovieLens
from src.data.Dataset import GraphQADataset
from src.utils.logging import log_test_to_wandb
from src.utils.checkpoints import save_model, load_model


def train_gnnllm(config: Any):
    apply_seed(0)

    wandb.init(project="kg-token", name=config.name, config=config)

    movielens = MovieLens(config)
    movielens.create_graph()

    # Freeze all LLM parameters
    device = config.device

    graph = movielens.train
    graph = graph.to(device)

    train_dataset = GraphQADataset(graph=movielens.train, config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    example_ct = 0  # number of examples seen

    gnn_llm = GraphTokenGPT(config, movielens.train.metadata())

    params = [p for _, p in gnn_llm.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': config.lr}, ],
        betas=(0.9, 0.95)
    )

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

    movielens = MovieLens(config)
    movielens.create_graph()

    test_dataset = GraphQADataset(graph=movielens.test, config=config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    device = config.device

    graph = movielens.train
    graph = graph.to(device)

    gnn_llm = GraphTokenGPT(config, movielens.test.metadata())
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

