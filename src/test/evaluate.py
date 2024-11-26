import torch
import random
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.test.metrics import yes_no_accuracy


def evaluate(llm_wrapper, gnn, graph, dataloader: DataLoader, config: dict):
    device = config['device']
    llm = llm_wrapper.get_llm().to(device)
    gnn = gnn.to(device)
    graph = graph.to(device)

    # Set model to evaluation mode
    llm.eval()
    gnn.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_tokens, target_mask, attention_mask, user_id, movie_id = batch

            input_tokens = input_tokens.to(device)
            target_mask = target_mask.to(device)
            attention_mask = attention_mask.to(device)
            user_id = user_id.to(device)
            movie_id = movie_id.to(device)

            embedding = gnn(graph.x_dict, graph.edge_index_dict)

            movie_embedding = embedding['movie'][movie_id].to(device)
            user_embedding = embedding['user'][user_id].to(device)

            user_token_id = llm_wrapper.get_tokenizer().convert_tokens_to_ids(llm_wrapper.USER_EMB)
            movie_token_id = llm_wrapper.get_tokenizer().convert_tokens_to_ids(llm_wrapper.MOVIE_EMB)

            base_embeddings = llm.get_input_embeddings().weight
            modified_embeddings = base_embeddings.clone()
            modified_embeddings[user_token_id] = user_embedding
            modified_embeddings[movie_token_id] = movie_embedding

            input_embeddings = F.embedding(input_tokens, modified_embeddings)

            outputs = llm(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
                labels=input_tokens
            )
            logits = outputs.logits

            # Convert logits to predictions (max probability)
            predictions = torch.argmax(logits, dim=-1)

            # Mask out the padding tokens (target_mask == 0)
            predictions = predictions[target_mask == 1]
            labels = input_tokens[target_mask == 1]

            tokenizer = llm_wrapper.get_tokenizer()
            predicted_tokens = tokenizer.convert_ids_to_tokens(predictions.squeeze().tolist())
            target_tokens = tokenizer.convert_ids_to_tokens(labels.squeeze().tolist())

            predicted_strings = ' '.join(predicted_tokens)
            target_strings = ' '.join(target_tokens)

            all_preds.append(predicted_strings)
            all_labels.append(target_strings)


    results = yes_no_accuracy(all_labels, all_preds)
    print(f"Accuracy: {results['yes_no_accuracy']}, Ambiguous: {results['yes_no_ambiguous']}, "
          f"Indeterminate: {results['yes_no_indeterminate']}")

    return results


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

    selected_batches = random.sample(list(dataloader), num_samples)

    with torch.no_grad():
        for batch in selected_batches:
            input_tokens, target_mask, attention_mask, user_id, movie_id = batch

            input_tokens = input_tokens.to(device)
            target_mask = target_mask.to(device)
            attention_mask = attention_mask.to(device)
            user_id = user_id.to(device)
            movie_id = movie_id.to(device)

            embedding = gnn(graph.x_dict, graph.edge_index_dict)

            movie_embedding = embedding['movie'][movie_id].to(device)
            user_embedding = embedding['user'][user_id].to(device)

            user_token_id = llm_wrapper.get_tokenizer().convert_tokens_to_ids(llm_wrapper.USER_EMB)
            movie_token_id = llm_wrapper.get_tokenizer().convert_tokens_to_ids(llm_wrapper.MOVIE_EMB)

            base_embeddings = llm.get_input_embeddings().weight
            modified_embeddings = base_embeddings.clone()
            modified_embeddings[user_token_id] = user_embedding
            modified_embeddings[movie_token_id] = movie_embedding

            input_embeddings = F.embedding(input_tokens, modified_embeddings)

            outputs = llm(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
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

            tokenizer = llm_wrapper.get_tokenizer()
            predicted_tokens = tokenizer.convert_ids_to_tokens(predictions.squeeze().tolist())
            target_tokens = tokenizer.convert_ids_to_tokens(labels.squeeze().tolist())
            # Join tokenized predictions and targets into single strings
            predicted_strings = ' '.join(predicted_tokens)
            target_strings = ' '.join(target_tokens)

            # Add these to the lists for printing
            all_preds.append(predicted_strings)
            all_labels.append(target_strings)

    # Print the results for the selected 10 examples
    print(f"Selected {num_samples} Examples:")
    for pred, true_label in zip(all_preds[:num_samples], all_labels[:num_samples]):
        print(f"Predicted: {pred}, True: {true_label}")

