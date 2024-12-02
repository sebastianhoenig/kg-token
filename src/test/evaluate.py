import torch
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.train.utils import prepare_inputs, get_batch_accuracy, log_evaluation_to_wandb


def evaluate(llm_wrapper, gnn, graph, dataloader: DataLoader, config: dict):
    device = config['device']
    llm = llm_wrapper.get_llm().to(device)
    gnn = gnn.to(device)
    graph = graph.to(device)

    # Set model to evaluation mode
    llm.eval()
    gnn.eval()

    all_logits = []
    all_labels = []
    all_target_masks = []

    USER_EMB = llm_wrapper.USER_EMB
    MOVIE_EMB = llm_wrapper.MOVIE_EMB

    tokenizer = llm_wrapper.get_tokenizer()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_tokens, target_mask, attention_masks, user_ids, movie_ids = batch

            batch_attention_masks, batch_embeddings, batch_labels, target_mask = prepare_inputs(MOVIE_EMB, USER_EMB,
                                                                                                attention_masks, device,
                                                                                                gnn, graph,
                                                                                                input_tokens, llm,
                                                                                                movie_ids, target_mask,
                                                                                                tokenizer, user_ids)

            outputs = llm(
                inputs_embeds=batch_embeddings,
                attention_mask=batch_attention_masks,
                labels=input_tokens
            )
            logits = outputs.logits

            all_logits.append(logits)
            all_labels.append(batch_labels)
            all_target_masks.append(target_mask)

    all_logits = torch.stack(all_logits)
    all_labels = torch.stack(all_labels)
    all_target_masks = torch.stack(all_target_masks)

    res = get_batch_accuracy(all_labels, all_logits, all_target_masks, tokenizer)

    log_evaluation_to_wandb(res)

    return res


def example_evaluations(llm_wrapper, gnn, graph, dataloader: DataLoader, config: dict, num_samples=10):
    device = config['device']
    llm = llm_wrapper.get_llm().to(device)
    gnn = gnn.to(device)
    graph = graph.to(device)

    # Set models to evaluation mode
    llm.eval()
    gnn.eval()

    all_preds = []
    all_labels = []

    USER_EMB = llm_wrapper.USER_EMB
    MOVIE_EMB = llm_wrapper.MOVIE_EMB

    selected_batches = random.sample(list(dataloader), num_samples)
    tokenizer = llm_wrapper.get_tokenizer()

    with torch.no_grad():
        for batch in selected_batches:
            input_tokens, target_mask, attention_masks, user_ids, movie_ids = batch

            batch_attention_masks, batch_embeddings, batch_labels, target_mask = prepare_inputs(MOVIE_EMB, USER_EMB,
                                                                                                attention_masks, device,
                                                                                                gnn, graph,
                                                                                                input_tokens, llm,
                                                                                                movie_ids, target_mask,
                                                                                                tokenizer, user_ids)
            outputs = llm(
                inputs_embeds=batch_embeddings,
                attention_mask=batch_attention_masks,
                labels=input_tokens
            )
            logits = outputs.logits
            logits = logits[0, :-1]

            target_mask = target_mask[0, 1:]
            labels = input_tokens[0, 1:]
            labels = labels[target_mask == 1]
            # Convert logits to predictions (max probability)
            predictions = torch.argmax(logits, dim=-1)

            # Mask out the padding tokens (target_mask == 0)
            predictions = predictions[target_mask == 1]

            predicted_tokens = tokenizer.convert_ids_to_tokens(predictions.squeeze().tolist())
            target_tokens = tokenizer.convert_ids_to_tokens(labels.squeeze().tolist())

            all_preds.append(predicted_tokens)
            all_labels.append(target_tokens)

    print(f"Selected {num_samples} Examples: ")
    for pred, true_label in zip(all_preds[:num_samples], all_labels[:num_samples]):
        print(f"Predicted: {pred}, True: {true_label}")
