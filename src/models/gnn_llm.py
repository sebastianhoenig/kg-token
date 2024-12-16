import torch
import contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import to_hetero


IGNORE_INDEX = -100

class GraphTokenGPT(nn.Module):
    # Adapted from https://github.com/franciscoliu/graphprompter/tree/main
    def __init__(self, config, llm, tokenizer, gnn):
        super().__init__()
        self.tokenizer = tokenizer#AutoTokenizer.from_pretrained(args.llm_model_path)
        model = llm#AutoModelForCausalLM.from_pretrained(args.llm_model_path)
        self.device = config['device']
        self.gnn = gnn
        #for name, param in model.named_parameters():
        #    param.requires_grad = False

        self.model = model

        special_tokens_dict = {'additional_special_tokens': [config['USER_EMB'], config['MOVIE_EMB']]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.config = config

        self.model.to(self.device)
        self.embedding_layer = self.model.get_input_embeddings()

    def maybe_autocast(self, dtype=torch.float16):
        # If on CPU, don't use autocast
        # If on GPU, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, batch, graph):
        input_tokens, target_mask, attention_masks, user_ids, movie_ids = batch

        input_tokens = input_tokens.to(self.device)
        target_mask = target_mask.to(self.device)
        attention_masks = attention_masks.to(self.device)
        user_ids = user_ids.to(self.device)
        movie_ids = movie_ids.to(self.device)

        batch_embeddings = []
        batch_attention_masks = []
        batch_labels = []

        graph_embeds = self.gnn(graph.x_dict, graph.edge_index_dict)

        for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
            movie_embedding = graph_embeds['movie'][movie_id].to(self.device)
            user_embedding = graph_embeds['user'][user_id].to(self.device)

            user_token_id = self.tokenizer.convert_tokens_to_ids(self.config['USER_EMB'])
            movie_token_id = self.tokenizer.convert_tokens_to_ids(self.config['MOVIE_EMB'])

            # Create a modified embedding matrix
            modified_embs = self.embedding_layer.weight.clone()
            modified_embs[user_token_id] = user_embedding
            modified_embs[movie_token_id] = movie_embedding

            # Embed input tokens using the modified embeddings
            input_embeddings = F.embedding(input_tokens[i], modified_embs)

            batch_embeddings.append(input_embeddings)
            batch_attention_masks.append(attention_masks[i])
            batch_labels.append(input_tokens[i].clone())

        batch_embeddings = torch.stack(batch_embeddings)
        batch_attention_masks = torch.stack(batch_attention_masks)
        batch_labels = torch.stack(batch_labels)

        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=batch_embeddings,
                attention_mask=batch_attention_masks,
                labels=batch_labels
            )

        logits = outputs.logits
        logits = logits[:, :-1, :].contiguous()

        batch_labels = batch_labels[:, 1:]
        target_mask = target_mask[:, 1:]

        # Compute the loss using PyTorch's CrossEntropyLoss
        batch_labels[target_mask == 0] = -100
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.flatten(0, 1), batch_labels.flatten())
        return logits, loss, batch_labels, target_mask