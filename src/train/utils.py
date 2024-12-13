import torch
import torch.nn.functional as F
from src.test.metrics import yes_no_accuracy


def prepare_inputs(MOVIE_EMB, USER_EMB, attention_masks, device, gnn, graph, input_tokens, llm, movie_ids, target_mask,
                   tokenizer, user_ids):
    input_tokens = input_tokens.to(device)
    target_mask = target_mask.to(device)
    attention_masks = attention_masks.to(device)
    user_ids = user_ids.to(device)
    movie_ids = movie_ids.to(device)
    embedding = gnn(graph.x_dict, graph.edge_index_dict)
    batch_embeddings = []
    batch_attention_masks = []
    batch_labels = []
    base_embeddings = llm.model.get_input_embeddings().weight
    for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
        movie_embedding = embedding['movie'][movie_id].to(device)
        user_embedding = embedding['user'][user_id].to(device)

        user_token_id = tokenizer.convert_tokens_to_ids(USER_EMB)
        movie_token_id = tokenizer.convert_tokens_to_ids(MOVIE_EMB)

        # Create a modified embedding matrix
        modified_embeddings = base_embeddings.clone()
        modified_embeddings[user_token_id] = user_embedding
        modified_embeddings[movie_token_id] = movie_embedding

        # Embed input tokens using the modified embeddings
        input_embeddings = F.embedding(input_tokens[i], modified_embeddings)

        batch_embeddings.append(input_embeddings)
        batch_attention_masks.append(attention_masks[i])
        batch_labels.append(input_tokens[i].clone())
    batch_embeddings = torch.stack(batch_embeddings)
    batch_attention_masks = torch.stack(batch_attention_masks)
    batch_labels = torch.stack(batch_labels)
    return batch_attention_masks, batch_embeddings, batch_labels, target_mask


def get_batch_accuracy(batch_labels, logits, target_mask, tokenizer):
    predictions = torch.argmax(logits, dim=-1)
    predictions = predictions[target_mask == 1]
    labels = batch_labels[target_mask == 1]
    predicted_tokens = tokenizer.convert_ids_to_tokens(predictions)
    target_tokens = tokenizer.convert_ids_to_tokens(labels)
    res = yes_no_accuracy(target_tokens, predicted_tokens)
    return res


