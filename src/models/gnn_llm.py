import torch
import contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList
from torch import nn
import torch.nn.functional as F
import numpy as np
from src.models.gnn_encoders import load_gnn_model
from torch_geometric.nn import to_hetero


IGNORE_INDEX = -100


class RestrictVocabLogitsProcessor:
    def __init__(self, tokenizer, device):
        self.allowed_tokens = torch.tensor(tokenizer.convert_tokens_to_ids(["Yes", "No"]))
        self.device = device

    def __call__(self, input_ids, scores):
        for i, token_scores in enumerate(scores):
            scores[i] = token_scores.masked_fill(~torch.isin(torch.arange(len(token_scores), device=self.device), self.allowed_tokens),-float('inf'))
        return scores


class GraphTokenGPT(nn.Module):
    # Adapted from https://github.com/franciscoliu/graphprompter/tree/main
    def __init__(self, args, metadata):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(args.llm_model_path)
        model = AutoModelForCausalLM.from_pretrained(args.llm_model_path)
        self.device = args.device
        self.gnn = gnn = load_gnn_model[args.gnn_model](
            hidden_channels=args.gnn_hidden_dim,
            out_channels=args.gnn_out_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            use_bn=args.gnn_use_bn,
            num_heads=args.gnn_num_heads,
        ).to(self.device)
        self.gnn = to_hetero(gnn, metadata, aggr=args.gnn_aggr)
        for name, param in model.named_parameters():
            param.requires_grad = False

        self.model = model
        self.fc1 = nn.Linear(args.gnn_hidden_dim, args.llm_embedding_dim).to(self.device)
        special_tokens_dict = {'additional_special_tokens': [args.USER_EMB, args.MOVIE_EMB]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.args = args

        if "gpt" in args.llm_model_path:
            self.model.to(self.device)
            self.embedding_layer = self.model.get_input_embeddings()
        else:
            self.embedding_layer = self.model.model.get_input_embeddings()
        self.logits_processors = LogitsProcessorList([RestrictVocabLogitsProcessor(self.tokenizer, self.device)])


    def maybe_autocast(self, dtype=torch.float16):
        # If on CPU, don't use autocast
        # If on GPU, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def forward(self, batch, graph):

        attention_masks = []
        user_ids = []
        movie_ids = []
        target_masks = []
        input_tokens = []

        for question, answer, user_id, movie_id in zip(batch["question"], batch["answer"], batch["user_id"], batch["movie_id"]):
            query_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            BOS_TOKEN = self.tokenizer.bos_token_id
            EOS_TOKEN = self.tokenizer.eos_token_id
            PAD_TOKEN = self.tokenizer.pad_token_id
            max_tokens = 35
            input_token = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_token)
            target_mask[len(query_tokens) + 1] = 1  # TRYING THIS OUT - REMOVING EOS TOKEN FROM TARGET MASK
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            input_token = np.pad(input_token, [[0, max_tokens - orig_len-1]], constant_values=PAD_TOKEN)
            target_mask = np.pad(target_mask, [[0, max_tokens - orig_len-1]], constant_values=0)
            attention_mask = np.ones_like(input_token)
            attention_mask[input_token == PAD_TOKEN] = 0

            attention_masks.append(torch.tensor(attention_mask))
            user_ids.append(user_id)
            movie_ids.append(movie_id)
            target_masks.append(torch.tensor(target_mask))
            input_tokens.append(torch.tensor(input_token))

        attention_masks = torch.stack(attention_masks).to(self.device)
        user_ids = torch.stack(user_ids).to(self.device)
        movie_ids = torch.stack(movie_ids).to(self.device)
        target_masks = torch.stack(target_masks).to(self.device)
        input_tokens = torch.stack(input_tokens).to(self.device)

        batch_embeddings = []
        batch_attention_masks = []
        batch_labels = []

        graph_embeds = self.gnn(graph.x_dict, graph.edge_index_dict)

        user_embeds = self.fc1(graph_embeds['user'])
        movie_embeds = self.fc1(graph_embeds['movie'])

        for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
            movie_embedding = movie_embeds[movie_id].to(self.device)
            user_embedding = user_embeds[user_id].to(self.device)

            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)
            movie_token_id = self.tokenizer.convert_tokens_to_ids(self.args.MOVIE_EMB)

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
                labels=batch_labels,
            )

        logits = outputs.logits
        logits = logits[:, :-1, :].contiguous()

        batch_labels = batch_labels[:, 1:]
        target_masks = target_masks[:, 1:]

        # Compute the loss using PyTorch's CrossEntropyLoss
        batch_labels[target_masks == 0] = -100
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.flatten(0, 1), batch_labels.flatten())
        return logits, loss, batch_labels, target_masks

    def inference(self, batch, graph):

        attention_masks = []
        user_ids = []
        movie_ids = []
        input_tokens = []

        for question, answer, user_id, movie_id in zip(batch["question"], batch["answer"], batch["user_id"],
                                                       batch["movie_id"]):
            query_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            BOS_TOKEN = self.tokenizer.bos_token_id
            PAD_TOKEN = self.tokenizer.pad_token_id
            max_tokens = 35
            input_token = np.array([BOS_TOKEN] + query_tokens)
            orig_len = len(query_tokens)
            input_token = np.pad(input_token, [[0, max_tokens - orig_len - 1]], constant_values=PAD_TOKEN)
            attention_mask = np.ones_like(input_token)
            attention_mask[input_token == PAD_TOKEN] = 0

            attention_masks.append(torch.tensor(attention_mask))
            user_ids.append(user_id)
            movie_ids.append(movie_id)
            input_tokens.append(torch.tensor(input_token))

        attention_masks = torch.stack(attention_masks).to(self.device)
        user_ids = torch.stack(user_ids).to(self.device)
        movie_ids = torch.stack(movie_ids).to(self.device)
        input_tokens = torch.stack(input_tokens).to(self.device)

        batch_embeddings = []
        batch_attention_masks = []

        graph_embeds = self.gnn(graph.x_dict, graph.edge_index_dict)

        user_embeds = self.fc1(graph_embeds['user'])
        movie_embeds = self.fc1(graph_embeds['movie'])

        for i, (user_id, movie_id) in enumerate(zip(user_ids, movie_ids)):
            movie_embedding = movie_embeds[movie_id].to(self.device)
            user_embedding = user_embeds[user_id].to(self.device)

            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)
            movie_token_id = self.tokenizer.convert_tokens_to_ids(self.args.MOVIE_EMB)

            # Create a modified embedding matrix
            modified_embs = self.embedding_layer.weight.clone()
            modified_embs[user_token_id] = user_embedding
            modified_embs[movie_token_id] = movie_embedding

            # Embed input tokens using the modified embeddings
            input_embeddings = F.embedding(input_tokens[i], modified_embs)

            batch_embeddings.append(input_embeddings)
            batch_attention_masks.append(attention_masks[i])

        batch_embeddings = torch.stack(batch_embeddings)
        batch_attention_masks = torch.stack(batch_attention_masks)

        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=batch_embeddings,
                attention_mask=batch_attention_masks,
                use_cache=True,
                max_new_tokens=self.args.max_new_tokens,
                logits_processor=self.logits_processors
            )

        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return {
            'questions': batch["question"],
            'answers': batch["answer"],
            'predictions': pred,
            'users': user_ids,
            'movies': movie_ids
        }