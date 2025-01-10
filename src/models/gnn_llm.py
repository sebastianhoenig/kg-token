import torch
import contextlib
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn
import torch.nn.functional as F
import numpy as np
from src.models.gnn_encoders import load_gnn_model
from torch_geometric.nn import to_hetero


IGNORE_INDEX = -100


class GraphTokenLLM(nn.Module):
    # Adapted from https://github.com/franciscoliu/graphprompter/tree/main
    def __init__(self, args, data):
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
        self.gnn = to_hetero(gnn, data.metadata(), aggr=args.gnn_aggr)

        if args.use_pretrained_gnn:
            pretrained_weights = torch.load(args.gnn_model_path, map_location=self.device)
            self.gnn.load_state_dict(pretrained_weights)
            if args.freeze_pretrained_gnn:
                for param in self.gnn.parameters():
                    param.requires_grad = False

        for name, param in model.named_parameters():
            param.requires_grad = False

        self.model = model
        self.fc1 = nn.Linear(args.gnn_hidden_dim, args.llm_embedding_dim).to(self.device)
        #self.fc2 = nn.Linear(args.llm_embedding_dim//2, args.llm_embedding_dim).to(self.device)
        special_tokens_dict = {'additional_special_tokens': args.SPECIAL_TOKENS}#[args.USER_EMB, args.MOVIE_EMB]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.args = args

        if "gpt" in args.llm_model_path:
            self.model.to(self.device)
            self.embedding_layer = self.model.get_input_embeddings()
        else:
            self.embedding_layer = self.model.model.get_input_embeddings()

        if args.embed_user_ids == True:
            self.user_id_emb = nn.Embedding(data['user'].num_nodes, args.user_id_dim).to(self.device)


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
            input_token = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_token)
            target_mask[len(query_tokens) + 1] = 1  # TRYING THIS OUT - REMOVING EOS TOKEN FROM TARGET MASK
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            max_tokens = orig_len + 5
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

        x_dict = graph.x_dict
        if self.args.embed_user_ids == True:
            x_dict['user'] = self.user_id_emb(x_dict['user'][:, 0].long())
        graph_embeds = self.gnn(x_dict, graph.edge_index_dict)

        #user_embeds = self.fc2(F.relu(self.fc1(graph_embeds['user'])))
        #movie_embeds = self.fc2(F.relu(self.fc1(graph_embeds['movie'])))

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

    def forward_classification(self, batch, graph):
        attention_masks = []
        user_ids = []
        target_masks = []
        input_tokens = []

        for question, answer, user_id in zip(batch["question"], batch["answer"], batch["user_id"]):
            query_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            BOS_TOKEN = self.tokenizer.bos_token_id
            EOS_TOKEN = self.tokenizer.eos_token_id
            PAD_TOKEN = self.tokenizer.pad_token_id
            input_token = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_token)
            target_mask[len(query_tokens) + 1] = 1  # Focus target on the answer part
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            max_tokens = orig_len + 5
            input_token = np.pad(input_token, [[0, max_tokens - orig_len - 1]], constant_values=PAD_TOKEN)
            target_mask = np.pad(target_mask, [[0, max_tokens - orig_len - 1]], constant_values=0)
            attention_mask = np.ones_like(input_token)
            attention_mask[input_token == PAD_TOKEN] = 0

            attention_masks.append(torch.tensor(attention_mask))
            user_ids.append(user_id)
            target_masks.append(torch.tensor(target_mask))
            input_tokens.append(torch.tensor(input_token))

        attention_masks = torch.stack(attention_masks).to(self.device)
        user_ids = torch.stack(user_ids).to(self.device)
        target_masks = torch.stack(target_masks).to(self.device)
        input_tokens = torch.stack(input_tokens).to(self.device)

        batch_embeddings = []
        batch_attention_masks = []
        batch_labels = []

        x_dict = graph.x_dict
        if self.args.embed_user_ids == True:
            x_dict['user'] = self.user_id_emb(x_dict['user'][:, 0].long())
        graph_embeds = self.gnn(x_dict, graph.edge_index_dict)

        #user_embeds = self.fc2(self.fc1(graph_embeds['user']))

        user_embeds = self.fc1(graph_embeds['user'])

        for i, user_id in enumerate(user_ids):
            user_embedding = user_embeds[user_id].to(self.device)

            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)

            # Create a modified embedding matrix
            modified_embs = self.embedding_layer.weight.clone()
            modified_embs[user_token_id] = user_embedding

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
        target_masks = []
        input_tokens = []

        for question, answer, user_id, movie_id in zip(batch["question"], batch["answer"], batch["user_id"],
                                                       batch["movie_id"]):
            query_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            BOS_TOKEN = self.tokenizer.bos_token_id
            EOS_TOKEN = self.tokenizer.eos_token_id
            PAD_TOKEN = self.tokenizer.pad_token_id
            max_tokens = 70
            input_token = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_token)
            target_mask[len(query_tokens) + 1] = 1  # TRYING THIS OUT - REMOVING EOS TOKEN FROM TARGET MASK
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            input_token = np.pad(input_token, [[0, max_tokens - orig_len - 1]], constant_values=PAD_TOKEN)
            target_mask = np.pad(target_mask, [[0, max_tokens - orig_len - 1]], constant_values=0)
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

        x_dict = graph.x_dict
        if self.args.embed_user_ids == True:
            x_dict['user'] = self.user_id_emb(x_dict['user'][:, 0].long())
        graph_embeds = self.gnn(x_dict, graph.edge_index_dict)

        # user_embeds = self.fc2(F.relu(self.fc1(graph_embeds['user'])))
        # movie_embeds = self.fc2(F.relu(self.fc1(graph_embeds['movie'])))

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

        return logits, batch_labels, target_masks

    def inference_preference_downstream(self, batch, graph):
        attention_masks = []
        user_ids = []
        movie1_ids = []
        movie2_ids = []
        movie1_embs = []
        movie2_embs = []
        os_movie1_ids = []
        os_movie2_ids = []
        os_movie1_embs = []
        os_movie2_embs = []
        target_masks = []
        input_tokens = []

        for question, answer, user_id, movie1_id, movie2_id, movie1_emb, movie2_emb, \
            os_movie1_id, os_movie2_id, os_movie1_emb, os_movie2_emb in zip(
            batch["question"], batch["answer"], batch["user_id"], batch["movie1_id"], batch["movie2_id"], batch["movie1_emb"], batch["movie2_emb"],
            batch["one_shot_movie1_id"], batch["one_shot_movie2_id"], batch["one_shot_movie1_emb"], batch["one_shot_movie2_emb"]
        ):
            query_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            BOS_TOKEN = self.tokenizer.bos_token_id
            EOS_TOKEN = self.tokenizer.eos_token_id
            PAD_TOKEN = self.tokenizer.pad_token_id
            input_token = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_token)
            target_mask[len(query_tokens) + 1] = 1
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            max_tokens = orig_len + 5
            input_token = np.pad(input_token, [[0, max_tokens - orig_len - 1]], constant_values=PAD_TOKEN)
            target_mask = np.pad(target_mask, [[0, max_tokens - orig_len - 1]], constant_values=0)
            attention_mask = np.ones_like(input_token)
            attention_mask[input_token == PAD_TOKEN] = 0

            attention_masks.append(torch.tensor(attention_mask))
            user_ids.append(user_id)
            movie1_ids.append(movie1_id)
            movie2_ids.append(movie2_id)
            movie1_embs.append(movie1_emb)
            movie2_embs.append(movie2_emb)
            os_movie1_ids.append(os_movie1_id)
            os_movie2_ids.append(os_movie2_id)
            os_movie1_embs.append(os_movie1_emb)
            os_movie2_embs.append(os_movie2_emb)
            target_masks.append(torch.tensor(target_mask))
            input_tokens.append(torch.tensor(input_token))

        attention_masks = torch.stack(attention_masks).to(self.device)
        user_ids = torch.tensor(user_ids).to(self.device)
        movie1_ids = torch.tensor(movie1_ids).to(self.device)
        movie2_ids = torch.tensor(movie2_ids).to(self.device)
        os_movie1_ids = torch.tensor(os_movie1_ids).to(self.device)
        os_movie2_ids = torch.tensor(os_movie2_ids).to(self.device)
        target_masks = torch.stack(target_masks).to(self.device)
        input_tokens = torch.stack(input_tokens).to(self.device)

        batch_embeddings = []
        batch_attention_masks = []
        batch_labels = []

        x_dict = graph.x_dict
        if self.args.embed_user_ids:
            x_dict['user'] = self.user_id_emb(x_dict['user'][:, 0].long())
        graph_embeds = self.gnn(x_dict, graph.edge_index_dict)

        user_embeds = self.fc1(graph_embeds['user'])
        movie_embeds = self.fc1(graph_embeds['movie'])

        for i, (user_id, movie1_id, movie2_id, movie1_emb, movie2_emb, os_movie1_id, os_movie2_id, os_movie1_emb, os_movie2_emb) in enumerate(zip(user_ids, movie1_ids, movie2_ids, movie1_embs, movie2_embs, os_movie1_ids, os_movie2_ids, os_movie1_embs, os_movie2_embs)):
            movie1_embedding = movie_embeds[movie1_id].to(self.device)
            movie2_embedding = movie_embeds[movie2_id].to(self.device)
            user_embedding = user_embeds[user_id].to(self.device)

            os_movie1_embedding = movie_embeds[os_movie1_id].to(self.device)
            os_movie2_embedding = movie_embeds[os_movie2_id].to(self.device)

            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)
            movie1_token_id = self.tokenizer.convert_tokens_to_ids(movie1_emb)
            movie2_token_id = self.tokenizer.convert_tokens_to_ids(movie2_emb)

            os_movie1_token_id = self.tokenizer.convert_tokens_to_ids(os_movie1_emb)
            os_movie2_token_id = self.tokenizer.convert_tokens_to_ids(os_movie2_emb)

            # Create a modified embedding matrix
            modified_embs = self.embedding_layer.weight.clone()
            modified_embs[user_token_id] = user_embedding
            modified_embs[movie1_token_id] = movie1_embedding
            modified_embs[movie2_token_id] = movie2_embedding
            modified_embs[os_movie1_token_id] = os_movie1_embedding
            modified_embs[os_movie2_token_id] = os_movie2_embedding

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

        return logits, batch_labels, target_masks


    def forward_two_preference(self, batch, graph):
        attention_masks = []
        user_ids = []
        movie1_ids = []
        movie2_ids = []
        movie1_embs = []
        movie2_embs = []
        target_masks = []
        input_tokens = []

        for question, answer, user_id, movie1_id, movie2_id, movie1_emb, movie2_emb in zip(
            batch["question"], batch["answer"], batch["user_id"], batch["movie1_id"], batch["movie2_id"], batch["movie1_emb"], batch["movie2_emb"],
        ):
            query_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            BOS_TOKEN = self.tokenizer.bos_token_id
            EOS_TOKEN = self.tokenizer.eos_token_id
            PAD_TOKEN = self.tokenizer.pad_token_id
            input_token = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_token)
            target_mask[len(query_tokens) + 1] = 1
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            max_tokens = orig_len + 5
            input_token = np.pad(input_token, [[0, max_tokens - orig_len - 1]], constant_values=PAD_TOKEN)
            target_mask = np.pad(target_mask, [[0, max_tokens - orig_len - 1]], constant_values=0)
            attention_mask = np.ones_like(input_token)
            attention_mask[input_token == PAD_TOKEN] = 0

            attention_masks.append(torch.tensor(attention_mask))
            user_ids.append(user_id)
            movie1_ids.append(movie1_id)
            movie2_ids.append(movie2_id)
            movie1_embs.append(movie1_emb)
            movie2_embs.append(movie2_emb)
            target_masks.append(torch.tensor(target_mask))
            input_tokens.append(torch.tensor(input_token))

        attention_masks = torch.stack(attention_masks).to(self.device)
        user_ids = torch.tensor(user_ids).to(self.device)
        movie1_ids = torch.tensor(movie1_ids).to(self.device)
        movie2_ids = torch.tensor(movie2_ids).to(self.device)
        target_masks = torch.stack(target_masks).to(self.device)
        input_tokens = torch.stack(input_tokens).to(self.device)

        batch_embeddings = []
        batch_attention_masks = []
        batch_labels = []

        x_dict = graph.x_dict
        if self.args.embed_user_ids:
            x_dict['user'] = self.user_id_emb(x_dict['user'][:, 0].long())
        graph_embeds = self.gnn(x_dict, graph.edge_index_dict)

        user_embeds = self.fc1(graph_embeds['user'])
        movie_embeds = self.fc1(graph_embeds['movie'])

        for i, (user_id, movie1_id, movie2_id, movie1_emb, movie2_emb) in enumerate(zip(user_ids, movie1_ids, movie2_ids, movie1_embs, movie2_embs)):
            movie1_embedding = movie_embeds[movie1_id].to(self.device)
            movie2_embedding = movie_embeds[movie2_id].to(self.device)
            user_embedding = user_embeds[user_id].to(self.device)

            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)
            movie1_token_id = self.tokenizer.convert_tokens_to_ids(movie1_emb)
            movie2_token_id = self.tokenizer.convert_tokens_to_ids(movie2_emb)

            # Create a modified embedding matrix
            modified_embs = self.embedding_layer.weight.clone()
            modified_embs[user_token_id] = user_embedding
            modified_embs[movie1_token_id] = movie1_embedding
            modified_embs[movie2_token_id] = movie2_embedding

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

        batch_labels[target_masks == 0] = -100
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(logits.flatten(0, 1), batch_labels.flatten())
        return logits, loss, batch_labels, target_masks


    def inference_three_preference(self, batch, graph):
        attention_masks = []
        user_ids = []
        movie1_ids = []
        movie2_ids = []
        movie3_ids = []
        movie1_embs = []
        movie2_embs = []
        movie3_embs = []
        target_masks = []
        input_tokens = []

        for question, answer, user_id, movie1_id, movie2_id, movie3_id, movie1_emb, movie2_emb, movie3_emb in zip(
            batch["question"], batch["answer"], batch["user_id"], batch["movie1_id"], batch["movie2_id"], batch["movie3_id"], batch["movie1_emb"], batch["movie2_emb"], batch["movie3_emb"],
        ):
            query_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            BOS_TOKEN = self.tokenizer.bos_token_id
            EOS_TOKEN = self.tokenizer.eos_token_id
            PAD_TOKEN = self.tokenizer.pad_token_id
            input_token = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_token)
            target_mask[len(query_tokens) + 1] = 1
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            max_tokens = orig_len + 5
            input_token = np.pad(input_token, [[0, max_tokens - orig_len - 1]], constant_values=PAD_TOKEN)
            target_mask = np.pad(target_mask, [[0, max_tokens - orig_len - 1]], constant_values=0)
            attention_mask = np.ones_like(input_token)
            attention_mask[input_token == PAD_TOKEN] = 0

            attention_masks.append(torch.tensor(attention_mask))
            user_ids.append(user_id)
            movie1_ids.append(movie1_id)
            movie2_ids.append(movie2_id)
            movie1_embs.append(movie1_emb)
            movie2_embs.append(movie2_emb)
            target_masks.append(torch.tensor(target_mask))
            input_tokens.append(torch.tensor(input_token))

        attention_masks = torch.stack(attention_masks).to(self.device)
        user_ids = torch.tensor(user_ids).to(self.device)
        movie1_ids = torch.tensor(movie1_ids).to(self.device)
        movie2_ids = torch.tensor(movie2_ids).to(self.device)
        movie3_ids = torch.tensor(movie3_ids).to(self.device)
        target_masks = torch.stack(target_masks).to(self.device)
        input_tokens = torch.stack(input_tokens).to(self.device)

        batch_embeddings = []
        batch_attention_masks = []
        batch_labels = []

        x_dict = graph.x_dict
        if self.args.embed_user_ids:
            x_dict['user'] = self.user_id_emb(x_dict['user'][:, 0].long())
        graph_embeds = self.gnn(x_dict, graph.edge_index_dict)

        user_embeds = self.fc1(graph_embeds['user'])
        movie_embeds = self.fc1(graph_embeds['movie'])

        for i, (user_id, movie1_id, movie2_id, movie3_id, movie1_emb, movie2_emb, movie3_emb) in enumerate(zip(user_ids, movie1_ids, movie2_ids, movie3_ids, movie1_embs, movie2_embs, movie3_embs)):
            movie1_embedding = movie_embeds[movie1_id].to(self.device)
            movie2_embedding = movie_embeds[movie2_id].to(self.device)
            movie3_embedding = movie_embeds[movie3_id].to(self.device)
            user_embedding = user_embeds[user_id].to(self.device)

            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)
            movie1_token_id = self.tokenizer.convert_tokens_to_ids(movie1_emb)
            movie2_token_id = self.tokenizer.convert_tokens_to_ids(movie2_emb)
            movie3_token_id = self.tokenizer.convert_tokens_to_ids(movie3_emb)

            # Create a modified embedding matrix
            modified_embs = self.embedding_layer.weight.clone()
            modified_embs[user_token_id] = user_embedding
            modified_embs[movie1_token_id] = movie1_embedding
            modified_embs[movie2_token_id] = movie2_embedding
            modified_embs[movie3_token_id] = movie3_embedding

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

        batch_labels[target_masks == 0] = -100
        return logits, batch_labels, target_masks

    def add_few_shot_tokens(self, modified_embs, user_embeds):
        for i, token in enumerate(self.args.SPECIAL_TOKENS[2:]):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            modified_embs[token_id] = user_embeds[i].to(self.device)
        return modified_embs

    def inference_downstream(self, batch, graph):
        attention_masks = []
        user_ids = []
        target_masks = []
        input_tokens = []

        for question, answer, user_id in zip(batch["question"], batch["answer"], batch["user_id"]):
            query_tokens = self.tokenizer(question, add_special_tokens=False)["input_ids"]
            answer_tokens = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
            BOS_TOKEN = self.tokenizer.bos_token_id
            EOS_TOKEN = self.tokenizer.eos_token_id
            PAD_TOKEN = self.tokenizer.pad_token_id
            input_token = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_token)
            target_mask[len(query_tokens) + 1] = 1  # Focus target on the answer part
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            max_tokens = orig_len + 5
            input_token = np.pad(input_token, [[0, max_tokens - orig_len - 1]], constant_values=PAD_TOKEN)
            target_mask = np.pad(target_mask, [[0, max_tokens - orig_len - 1]], constant_values=0)
            attention_mask = np.ones_like(input_token)
            attention_mask[input_token == PAD_TOKEN] = 0

            attention_masks.append(torch.tensor(attention_mask))
            user_ids.append(user_id)
            target_masks.append(torch.tensor(target_mask))
            input_tokens.append(torch.tensor(input_token))

        attention_masks = torch.stack(attention_masks).to(self.device)
        user_ids = torch.stack(user_ids).to(self.device)
        target_masks = torch.stack(target_masks).to(self.device)
        input_tokens = torch.stack(input_tokens).to(self.device)

        batch_embeddings = []
        batch_attention_masks = []
        batch_labels = []

        x_dict = graph.x_dict
        if self.args.embed_user_ids == True:
            x_dict['user'] = self.user_id_emb(x_dict['user'][:, 0].long())
        graph_embeds = self.gnn(x_dict, graph.edge_index_dict)

        #user_embeds = self.fc2(self.fc1(graph_embeds['user']))

        user_embeds = self.fc1(graph_embeds['user'])

        for i, user_id in enumerate(user_ids):
            user_embedding = user_embeds[user_id].to(self.device)

            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)

            # Create a modified embedding matrix
            modified_embs = self.embedding_layer.weight.clone()
            modified_embs[user_token_id] = user_embedding

            modified_embs = self.add_few_shot_tokens(modified_embs, user_embeds)

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

        return logits, batch_labels, target_masks


