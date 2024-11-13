

def update_embeddings(user_embedding, movie_embedding, user_token_id, movie_token_id, params, model_name):
    if model_name == "gpt2":
        params['transformer.wte.weight'][user_token_id] = user_embedding
        params['transformer.wte.weight'][movie_token_id] = movie_embedding
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        pass
    # TODO: Add support for other models