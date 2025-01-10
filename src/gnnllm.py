from typing import Any

import pandas as pd

import wandb
from torch.utils.data import DataLoader
import torch
import tqdm
from tqdm.notebook import tqdm
from src.utils.logging import log_train_to_wandb
from src.utils.seed import apply_seed
from src.models.gnn_llm import GraphTokenLLM
from src.utils.metrics import get_accuracy_gnnllm
from src.graph.Movielens100k import MovieLens
from src.data.LinkPredictionDataset import GraphQADataset
from src.data.LinkRatingDataset import GraphRatingDataset
from src.data.NodeClassificationDataset import NodeClassificationDataset
from src.data.MoviePreference import GraphQAPreferenceDataset
from src.utils.logging import log_test_to_wandb, log_age_downstream_task_to_wandb, log_gender_downstream_task_to_wandb, log_age_train_task_to_wandb
from src.utils.checkpoints import save_model, load_model
from src.data.DownstreamAgeDataset import AgeDataset
from src.data.DownstreamGenderDataset import GenderDataset


def train_gnnllm(config: Any):
    apply_seed(0)

    wandb.init(project=config.project_name, name=config.name, config=config)

    movielens = MovieLens(config)
    movielens.create_graph()

    # Freeze all LLM parameters
    device = config.device

    graph = movielens.train
    graph = graph.to(device)

    train_dataset = GraphQADataset(graph=movielens.train, config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    example_ct = 0  # number of examples seen

    gnn_llm = GraphTokenLLM(config, movielens.train)

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
            logits, batch_labels, target_mask = gnn_llm.inference(batch, graph)

            batch_res = get_accuracy_gnnllm(batch_labels, logits, target_mask, gnn_llm.tokenizer)

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

    wandb.finish()


def train_gnnllm_rating(config: Any):
    apply_seed(0)

    wandb.init(project=config.project_name, name=config.name, config=config)

    movielens = MovieLens(config)
    movielens.create_graph()

    # Freeze all LLM parameters
    device = config.device

    graph = movielens.train
    graph = graph.to(device)

    train_dataset = GraphRatingDataset(graph=movielens.train, config=config)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    example_ct = 0  # number of examples seen

    gnn_llm = GraphTokenLLM(config, movielens.train)

    if config.use_pt:
        print("Loading model from checkpoint")
        gnn_llm = load_model(gnn_llm, config)

    params = [p for _, p in gnn_llm.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': config.lr}, ],
        betas=(0.9, 0.95)
    )

    for epoch in tqdm(range(config.num_epochs), desc="Epoch Progress"):

        total_loss = 0

        for batch in tqdm(train_loader, desc="Batch Progress", leave=False):

            batch_correct = 0
            batch_items = 0
            batch_correct_1 = 0
            batch_correct_2 = 0
            batch_correct_3 = 0
            batch_correct_4 = 0
            batch_correct_5 = 0

            batch_size = len(batch['question'])

            logits, loss, batch_labels, target_mask = gnn_llm.forward(batch, graph)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_res = get_accuracy_gnnllm(batch_labels, logits, target_mask, gnn_llm.tokenizer, task='rating')

            batch_correct += batch_res['total_correct']
            batch_items += batch_res['total_items']
            accuracy = batch_correct / batch_items if batch_items > 0 else 0


            for key, value in batch_res['correct_prediction_counts'].items():
                if key == "1":
                    batch_correct_1 += value
                elif key == "2":
                    batch_correct_2 += value
                elif key == "3":
                    batch_correct_3 += value
                elif key == "4":
                    batch_correct_4 += value
                elif key == "5":
                    batch_correct_5 += value


            total_loss += loss.detach().item()
            #scheduler.step(loss.item())

            example_ct += batch_size

            wandb.log({"epoch": epoch, "loss": loss, "accuracy": accuracy, "correct_1": batch_correct_1, "correct_2": batch_correct_2, "correct_3": batch_correct_3, "correct_4": batch_correct_4, "correct_5": batch_correct_5}, step=example_ct)

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{config.num_epochs}, Loss: {avg_loss}")

    print("Evaluating on Test set")
    save_model(gnn_llm, config)

    test_dataset = GraphRatingDataset(graph=movielens.test, config=config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    gnn_llm.eval()

    total_correct = 0
    total_items = 0
    rating_category_counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}
    correct_prediction_counts = {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            logits, _, batch_labels, target_mask = gnn_llm.forward(batch, graph)

            batch_res = get_accuracy_gnnllm(batch_labels, logits, target_mask, gnn_llm.tokenizer, task='rating')

            total_correct += batch_res['total_correct']
            total_items += batch_res['total_items']

            for key, value in batch_res['correct_prediction_counts'].items():
                correct_prediction_counts[key] += value
            for key,value in batch_res['rating_category_counts'].items():
                rating_category_counts[key] += value

    accuracy_1 = correct_prediction_counts['1'] / rating_category_counts['1'] if rating_category_counts['1'] > 0 else 0
    accuracy_2 = correct_prediction_counts['2'] / rating_category_counts['2'] if rating_category_counts['2'] > 0 else 0
    accuracy_3 = correct_prediction_counts['3'] / rating_category_counts['3'] if rating_category_counts['3'] > 0 else 0
    accuracy_4 = correct_prediction_counts['4'] / rating_category_counts['4'] if rating_category_counts['4'] > 0 else 0
    accuracy_5 = correct_prediction_counts['5'] / rating_category_counts['5'] if rating_category_counts['5'] > 0 else 0

    accuracy = total_correct / total_items if total_items > 0 else 0

    data = [
        ["accuracy", accuracy],
        ["num_items", total_items],
        ["num_correct", total_correct],
        ["accuracy_1", accuracy_1],
        ["accuracy_2", accuracy_2],
        ["accuracy_3", accuracy_3],
        ["accuracy_4", accuracy_4],
        ["accuracy_5", accuracy_5]
    ]

    table = wandb.Table(columns=["Metric", "Value"], data=data)

    wandb.log({"evaluation_table": table})

    wandb.finish()


def train_gnnllm_node_classification(config: Any):
    apply_seed(0)

    wandb.init(project=config.project_name, name=config.name, config=config)

    movielens = MovieLens(config)
    movielens.create_graph()

    # Freeze all LLM parameters
    device = config.device

    graph = movielens.train
    graph = graph.to(device)

    dataset = NodeClassificationDataset(graph=movielens.train, config=config)
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    val_dataset = NodeClassificationDataset(graph=movielens.train, config=config)
    val_dataset.set_mode("val")
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)

    gnn_llm = GraphTokenLLM(config, movielens.train)

    params = [p for _, p in gnn_llm.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': config.lr}, ],
        betas=(0.9, 0.95)
    )

    for epoch in tqdm(range(config.num_epochs), desc="Epoch Progress"):

        gnn_llm.train()
        total_loss = 0
        epoch_correct = 0
        epoch_items = 0
        epoch_correct_young = 0
        epoch_correct_adult = 0
        epoch_correct_old = 0

        for batch in tqdm(loader, desc="Batch Progress", leave=False):

            logits, loss, batch_labels, target_mask = gnn_llm.forward_classification(batch, graph)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            res = get_accuracy_gnnllm(batch_labels, logits, target_mask, gnn_llm.tokenizer, task='age')

            epoch_correct += res['total_correct']
            epoch_items += res['total_items']
            epoch_correct_young += res['correct_predictions']['Young']
            epoch_correct_adult += res['correct_predictions']['Adult']
            epoch_correct_old += res['correct_predictions']['Old']

            total_loss += loss.detach().item()

        avg_loss = total_loss / len(loader)
        log_age_train_task_to_wandb(epoch_correct, epoch_items, epoch_correct_young, epoch_correct_adult, epoch_correct_old, epoch, avg_loss)

        #VAL
        gnn_llm.eval()
        val_correct, val_items = 0, 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation Progress", leave=False):
                logits, _, batch_labels, target_mask = gnn_llm.forward_classification(batch, graph)

                res = get_accuracy_gnnllm(batch_labels, logits, target_mask, gnn_llm.tokenizer, task='age')

                val_correct += res['total_correct']
                val_items += res['total_items']

        val_accuracy = val_correct / val_items if val_items > 0 else 0
        print(f"Validation Accuracy: {val_accuracy:.2%}")
        wandb.log({
            "val_accuracy": val_accuracy,
        }, step=epoch)

    save_model(gnn_llm, config)

    print("Evaluating on Test set")
    test_dataset = NodeClassificationDataset(graph=movielens.train, config=config)
    test_dataset.set_mode("test")
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    gnn_llm.eval()

    total_correct, total_items = 0, 0
    age_category_counts = {"Young": 0, "Adult": 0, "Old": 0}
    correct_predictions = {"Young": 0, "Adult": 0, "Old": 0}

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            logits, _, batch_labels, target_mask = gnn_llm.forward_classification(batch, graph)

            batch_res = get_accuracy_gnnllm(batch_labels, logits, target_mask, gnn_llm.tokenizer, task='age')

            total_correct += batch_res['total_correct']
            total_items += batch_res['total_items']
            for key in age_category_counts:
                age_category_counts[key] += batch_res['age_category_counts'][key]
                correct_predictions[key] += batch_res['correct_predictions'][key]

    aggregated_res = {
        "total_correct": total_correct,
        "total_items": total_items,
        "young_category": age_category_counts['Young'],
        "adult_category": age_category_counts['Adult'],
        "old_category": age_category_counts['Old'],
        "correct_young": correct_predictions['Young'],
        "correct_adult": correct_predictions['Adult'],
        "correct_old": correct_predictions['Old'],
    }

    log_age_downstream_task_to_wandb(aggregated_res)

    wandb.finish()


def evaluate(config: Any):
    movielens = MovieLens(config)
    movielens.create_graph()

    test_dataset = GraphQADataset(graph=movielens.test, config=config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)
    device = config.device

    graph = movielens.train
    graph = graph.to(device)

    gnn_llm = GraphTokenLLM(config, movielens.test)
    gnn_llm = load_model(gnn_llm, config)
    # Set model to evaluation mode
    gnn_llm.eval()

    total_correct_yes_preds, total_correct_no_preds = 0, 0
    total_wrong_yes_preds, total_wrong_no_preds = 0, 0
    total_yes_targets, total_no_targets = 0, 0
    total_correct, total_items = 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            logits, batch_labels, target_mask = gnn_llm.inference(batch, graph)

            batch_res = get_accuracy_gnnllm(batch_labels, logits, target_mask, gnn_llm.tokenizer)

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


def evaluate_age(config: Any):

    wandb.init(project=config.project_name, name="AGE_DS_TASK", config=config)

    movielens = MovieLens(config)
    movielens.create_graph()

    total_correct, total_items = 0, 0
    age_category_counts = {"Young": 0, "Adult": 0, "Old": 0}
    correct_predictions = {"Young": 0, "Adult": 0, "Old": 0}

    test_dataset = AgeDataset(graph=movielens.test, config=config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    device = config.device

    gnn_llm = GraphTokenLLM(config, movielens.test)
    gnn_llm = load_model(gnn_llm, config)
    gnn_llm = gnn_llm.to(device)
    gnn_llm.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            logits, batch_labels, target_masks = gnn_llm.inference_downstream(batch, graph=movielens.test)

            batch_res = get_accuracy_gnnllm(batch_labels, logits, target_masks, gnn_llm.tokenizer, task='age')
            total_correct += batch_res['total_correct']
            total_items += batch_res['total_items']
            for key in age_category_counts:
                age_category_counts[key] += batch_res['age_category_counts'][key]
                correct_predictions[key] += batch_res['correct_predictions'][key]

    aggregated_res = {
        "total_correct": total_correct,
        "total_items": total_items,
        "young_category": age_category_counts['Young'],
        "adult_category": age_category_counts['Adult'],
        "old_category": age_category_counts['Old'],
        "correct_young": correct_predictions['Young'],
        "correct_adult": correct_predictions['Adult'],
        "correct_old": correct_predictions['Old'],
    }

    log_age_downstream_task_to_wandb(aggregated_res)

    wandb.finish()


def pretrain_preference(config: Any):
    # Initialize logging
    wandb.init(project=config.project_name, name="PREFERENCE_DS_TASK", config=config)

    # Load graph data
    movielens = MovieLens(config)
    movielens.create_graph()

    user_movie_data = pd.read_csv(config.user_pref_path)
    # Initialize dataset and dataloader
    train_dataset = GraphQAPreferenceDataset(graph=movielens.train, config=config, user_movie_data=user_movie_data)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    device = config.device

    # Load and prepare the model
    gnn_llm = GraphTokenLLM(config, movielens.test)

    if config.use_pt:
        print("Loading model from checkpoint")
        gnn_llm = load_model(gnn_llm, config)
    gnn_llm = gnn_llm.to(device)

    params = [p for _, p in gnn_llm.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': config.lr}, ],
        betas=(0.9, 0.95)
    )
    gnn_llm.train()

    for epoch in tqdm(range(config.num_epochs), desc="Epoch Progress"):
        total_correct, total_items = 0, 0
        for batch in tqdm(train_loader, desc="Evaluating"):
            logits, loss, batch_labels, target_masks = gnn_llm.forward_two_preference(batch, graph=movielens.test)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Process results for preference
            batch_res = get_accuracy_gnnllm(batch_labels, logits, target_masks, gnn_llm.tokenizer, task='preference')
            total_correct += batch_res['num_correct']
            total_items += batch_res['num_items']

        accuracy = total_correct / total_items if total_items > 0 else 0
        wandb.log({"epoch": epoch, "accuracy": accuracy})

    save_model(gnn_llm, config)

    wandb.finish()



def evaluate_gender(config: Any):

    wandb.init(project=config.project_name, name="Gender_DS_Task", config=config)

    movielens = MovieLens(config)
    movielens.create_graph()

    total_correct, total_items = 0, 0
    gender_category_counts = {"Male": 0, "Female": 0}
    correct_predictions = {"Male": 0, "Female": 0}

    test_dataset = GenderDataset(graph=movielens.test, config=config)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    device = config.device

    gnn_llm = GraphTokenLLM(config, movielens.test)
    gnn_llm = load_model(gnn_llm, config)
    gnn_llm = gnn_llm.to(device)
    gnn_llm.eval()  # Set model to evaluation mode

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            logits, batch_labels, target_masks = gnn_llm.inference_downstream(batch, graph=movielens.test)

            batch_res = get_accuracy_gnnllm(batch_labels, logits, target_masks, gnn_llm.tokenizer, task='gender')
            total_correct += batch_res['total_correct']
            total_items += batch_res['total_items']
            for key in gender_category_counts:
                gender_category_counts[key] += batch_res['gender_category_counts'][key]
                correct_predictions[key] += batch_res['correct_predictions'][key]

    aggregated_res = {
        "total_correct": total_correct,
        "total_items": total_items,
        "male_category": gender_category_counts["Male"],
        "female_category": gender_category_counts["Female"],
        "correct_male": correct_predictions["Male"],
        "correct_female": correct_predictions["Female"]
    }

    log_gender_downstream_task_to_wandb(aggregated_res)

    wandb.finish()

