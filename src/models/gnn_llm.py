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

    def _prepare_sample(self, question, answer, max_tokens_extra=5):
        """
        Tokenizes question and answer, constructs input sequence, attention mask, and target mask.
        Returns: input_token (np.array), attention_mask (torch.Tensor), target_mask (torch.Tensor)
        """
        tokens_q = self.tokenizer(question, add_special_tokens=False)["input_ids"]
        tokens_a = self.tokenizer(answer, add_special_tokens=False)["input_ids"]
        BOS = self.tokenizer.bos_token_id
        EOS = self.tokenizer.eos_token_id
        PAD = self.tokenizer.pad_token_id

        # Construct token sequence
        input_token = np.array([BOS] + tokens_q + tokens_a + [EOS])
        target_mask = np.zeros_like(input_token)
        # Mark a single position (after question tokens) as the target.
        target_mask[len(tokens_q) + 1] = 1

        orig_len = len(input_token)
        max_tokens = orig_len + max_tokens_extra
        # Pad tokens and masks to max_tokens length
        input_token = np.pad(input_token, (0, max_tokens - orig_len), constant_values=PAD)
        target_mask = np.pad(target_mask, (0, max_tokens - orig_len), constant_values=0)
        attention_mask = np.ones_like(input_token)
        attention_mask[input_token == PAD] = 0

        return torch.tensor(input_token), torch.tensor(attention_mask), torch.tensor(target_mask)

    def _extract_graph_embeddings(self, graph):
        """
        Extracts graph embeddings using the GNN. Optionally embeds user IDs if required.
        Returns a dictionary with transformed embeddings.
        """
        x_dict = graph.x_dict
        if self.args.embed_user_ids:
            x_dict['user'] = self.user_id_emb(x_dict['user'][:, 0].long())
        graph_embeds = self.gnn(x_dict, graph.edge_index_dict)
        # Transform embeddings with fc1
        user_embeds = self.fc1(graph_embeds['user'])
        movie_embeds = self.fc1(graph_embeds['movie'])
        return {'user': user_embeds, 'movie': movie_embeds}

    def _modify_embedding_matrix(self, replacements):
        """
        Clones the current embedding weight and replaces indices specified in the
        replacements dict. 'replacements' is expected to be a dict mapping token_id (int)
        to the new embedding (tensor).
        """
        modified = self.embedding_layer.weight.clone()
        for token_id, new_emb in replacements.items():
            modified[token_id] = new_emb.to(self.device)
        return modified

    def _run_model(self, batch_embeddings, attention_masks, input_tokens, target_masks=None):
        """
        Runs the underlying LLM model and optionally computes the loss.
        If target_masks is provided, computes cross-entropy loss using IGNORE_INDEX for padded positions.
        """
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=batch_embeddings,
                attention_mask=attention_masks,
                labels=input_tokens,
            )
        logits = outputs.logits[:, :-1, :].contiguous()
        labels = input_tokens[:, 1:]
        if target_masks is not None:
            # Align target mask with labels and mask out positions with 0
            target_masks = target_masks[:, 1:]
            labels = labels.clone()
            labels[target_masks == 0] = IGNORE_INDEX
            loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
            loss = loss_fn(logits.flatten(0, 1), labels.flatten())
            return logits, loss, labels, target_masks
        return logits, labels

    def forward(self, batch, graph):
        """
        Forward pass for the main binary classification task.
        """
        input_tokens_list, attn_masks_list, target_masks_list = [], [], []
        for question, answer, _, _ in zip(batch["question"], batch["answer"], batch["user_id"], batch["movie_id"]):
            inp, attn, tgt = self._prepare_sample(question, answer)
            input_tokens_list.append(inp)
            attn_masks_list.append(attn)
            target_masks_list.append(tgt)

        input_tokens = torch.stack(input_tokens_list).to(self.device)
        attention_masks = torch.stack(attn_masks_list).to(self.device)
        target_masks = torch.stack(target_masks_list).to(self.device)

        # Get graph embeddings
        embeds = self._extract_graph_embeddings(graph)
        user_embeds, movie_embeds = embeds['user'], embeds['movie']

        batch_embeddings = []
        for i, (user_id, movie_id) in enumerate(zip(torch.tensor(batch["user_id"]), torch.tensor(batch["movie_id"]))):
            user_embedding = user_embeds[user_id].to(self.device)
            movie_embedding = movie_embeds[movie_id].to(self.device)

            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)
            movie_token_id = self.tokenizer.convert_tokens_to_ids(self.args.MOVIE_EMB)
            replacements = {
                user_token_id: user_embedding,
                movie_token_id: movie_embedding,
            }
            modified_embs = self._modify_embedding_matrix(replacements)
            # Embed the input tokens with the modified embeddings.
            input_emb = F.embedding(input_tokens[i], modified_embs)
            batch_embeddings.append(input_emb)

        batch_embeddings = torch.stack(batch_embeddings)
        return self._run_model(batch_embeddings, attention_masks, input_tokens, target_masks)

    def forward_classification(self, batch, graph):
        """
        Forward pass for node classification tasks (e.g., age prediction).
        """
        input_tokens_list, attn_masks_list, target_masks_list = [], [], []
        for question, answer, _ in zip(batch["question"], batch["answer"], batch["user_id"]):
            inp, attn, tgt = self._prepare_sample(question, answer)
            input_tokens_list.append(inp)
            attn_masks_list.append(attn)
            target_masks_list.append(tgt)

        input_tokens = torch.stack(input_tokens_list).to(self.device)
        attention_masks = torch.stack(attn_masks_list).to(self.device)
        target_masks = torch.stack(target_masks_list).to(self.device)

        embeds = self._extract_graph_embeddings(graph)
        user_embeds = embeds['user']

        batch_embeddings = []
        for i, user_id in enumerate(torch.tensor(batch["user_id"])):
            user_embedding = user_embeds[user_id].to(self.device)
            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)
            replacements = {user_token_id: user_embedding}
            modified_embs = self._modify_embedding_matrix(replacements)
            input_emb = F.embedding(input_tokens[i], modified_embs)
            batch_embeddings.append(input_emb)

        batch_embeddings = torch.stack(batch_embeddings)
        return self._run_model(batch_embeddings, attention_masks, input_tokens, target_masks)

    def inference(self, batch, graph):
        """
        Inference for evaluation tasks. No loss is computed.
        """
        input_tokens_list, attn_masks_list, target_masks_list = [], [], []
        max_tokens_fixed = 70
        for question, answer, _, _ in zip(batch["question"], batch["answer"], batch["user_id"], batch["movie_id"]):
            # Use a fixed extra length for inference.
            inp, attn, tgt = self._prepare_sample(question, answer, max_tokens_extra=(max_tokens_fixed - len(self._prepare_sample(question, answer)[0])))
            input_tokens_list.append(inp)
            attn_masks_list.append(attn)
            target_masks_list.append(tgt)

        input_tokens = torch.stack(input_tokens_list).to(self.device)
        attention_masks = torch.stack(attn_masks_list).to(self.device)
        target_masks = torch.stack(target_masks_list).to(self.device)

        embeds = self._extract_graph_embeddings(graph)
        user_embeds, movie_embeds = embeds['user'], embeds['movie']

        batch_embeddings = []
        for i, (user_id, movie_id) in enumerate(zip(torch.tensor(batch["user_id"]), torch.tensor(batch["movie_id"]))):
            user_embedding = user_embeds[user_id].to(self.device)
            movie_embedding = movie_embeds[movie_id].to(self.device)
            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)
            movie_token_id = self.tokenizer.convert_tokens_to_ids(self.args.MOVIE_EMB)
            replacements = {
                user_token_id: user_embedding,
                movie_token_id: movie_embedding,
            }
            modified_embs = self._modify_embedding_matrix(replacements)
            input_emb = F.embedding(input_tokens[i], modified_embs)
            batch_embeddings.append(input_emb)

        batch_embeddings = torch.stack(batch_embeddings)
        return self._run_model(batch_embeddings, attention_masks, input_tokens, target_masks)

    def forward_two_preference(self, batch, graph):
        """
        Forward pass for the supervised learning task of User Preferences (2 Movies).
        """
        input_tokens_list, attn_masks_list, target_masks_list = [], [], []
        for question, answer, _ in zip(batch["question"], batch["answer"], batch["user_id"]):
            inp, attn, tgt = self._prepare_sample(question, answer)
            input_tokens_list.append(inp)
            attn_masks_list.append(attn)
            target_masks_list.append(tgt)

        input_tokens = torch.stack(input_tokens_list).to(self.device)
        attention_masks = torch.stack(attn_masks_list).to(self.device)
        target_masks = torch.stack(target_masks_list).to(self.device)

        user_ids = torch.tensor(batch["user_id"]).to(self.device)
        movie1_ids = torch.tensor(batch["movie1_id"]).to(self.device)
        movie2_ids = torch.tensor(batch["movie2_id"]).to(self.device)

        embeds = self._extract_graph_embeddings(graph)
        user_embeds, movie_embeds = embeds['user'], embeds['movie']

        batch_embeddings = []
        for i, (user_id, m1, m2, m1_tok, m2_tok) in enumerate(
            zip(user_ids, movie1_ids, movie2_ids, batch["movie1_emb"], batch["movie2_emb"])
        ):
            replacements = {
                self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB): user_embeds[user_id].to(self.device),
                self.tokenizer.convert_tokens_to_ids(m1_tok): movie_embeds[m1].to(self.device),
                self.tokenizer.convert_tokens_to_ids(m2_tok): movie_embeds[m2].to(self.device),
            }
            modified_embs = self._modify_embedding_matrix(replacements)
            input_emb = F.embedding(input_tokens[i], modified_embs)
            batch_embeddings.append(input_emb)

        batch_embeddings = torch.stack(batch_embeddings)
        return self._run_model(batch_embeddings, attention_masks, input_tokens, target_masks)

    def inference_three_preference(self, batch, graph):
        """
        Inference function for the Downstream Task User Preferences (3 Movies).
        """
        input_tokens_list, attn_masks_list, target_masks_list = [], [], []
        for question, answer, _ in zip(batch["question"], batch["answer"], batch["user_id"]):
            inp, attn, tgt = self._prepare_sample(question, answer)
            input_tokens_list.append(inp)
            attn_masks_list.append(attn)
            target_masks_list.append(tgt)

        input_tokens = torch.stack(input_tokens_list).to(self.device)
        attention_masks = torch.stack(attn_masks_list).to(self.device)
        target_masks = torch.stack(target_masks_list).to(self.device)

        user_ids = torch.tensor(batch["user_id"]).to(self.device)
        movie1_ids = torch.tensor(batch["movie1_id"]).to(self.device)
        movie2_ids = torch.tensor(batch["movie2_id"]).to(self.device)
        movie3_ids = torch.tensor(batch["movie3_id"]).to(self.device)

        embeds = self._extract_graph_embeddings(graph)
        user_embeds, movie_embeds = embeds['user'], embeds['movie']

        batch_embeddings = []
        for i, (user_id, m1, m2, m3, m1_tok, m2_tok, m3_tok) in enumerate(
            zip(user_ids, movie1_ids, movie2_ids, movie3_ids,
                batch["movie1_emb"], batch["movie2_emb"], batch["movie3_emb"])
        ):
            replacements = {
                self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB): user_embeds[user_id].to(self.device),
                self.tokenizer.convert_tokens_to_ids(m1_tok): movie_embeds[m1].to(self.device),
                self.tokenizer.convert_tokens_to_ids(m2_tok): movie_embeds[m2].to(self.device),
                self.tokenizer.convert_tokens_to_ids(m3_tok): movie_embeds[m3].to(self.device),
            }
            modified_embs = self._modify_embedding_matrix(replacements)
            input_emb = F.embedding(input_tokens[i], modified_embs)
            batch_embeddings.append(input_emb)

        batch_embeddings = torch.stack(batch_embeddings)
        # Run model and compute loss manually (since loss is computed here)
        logits, labels, tgt = self._run_model(batch_embeddings, attention_masks, input_tokens, target_masks)
        return logits, labels, tgt

    def _add_few_shot_tokens(self, modified_embs, user_embeds):
        """
        Adds few-shot tokens to the modified embedding matrix.
        """
        # Assumes that the few-shot tokens start at index 2 of SPECIAL_TOKENS.
        for i, token in enumerate(self.args.SPECIAL_TOKENS[2:]):
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            modified_embs[token_id] = user_embeds[i].to(self.device)
        return modified_embs

    def inference_downstream(self, batch, graph):
        """
        Inference for downstream tasks that only concern the user node, so node classification tasks. Few-shot tokens
        can be added (although this was not part of the report)
        """
        input_tokens_list, attn_masks_list, target_masks_list = [], [], []
        for question, answer, _ in zip(batch["question"], batch["answer"], batch["user_id"]):
            inp, attn, tgt = self._prepare_sample(question, answer)
            input_tokens_list.append(inp)
            attn_masks_list.append(attn)
            target_masks_list.append(tgt)

        input_tokens = torch.stack(input_tokens_list).to(self.device)
        attention_masks = torch.stack(attn_masks_list).to(self.device)
        target_masks = torch.stack(target_masks_list).to(self.device)

        embeds = self._extract_graph_embeddings(graph)
        user_embeds = embeds['user']

        batch_embeddings = []
        for i, user_id in enumerate(torch.tensor(batch["user_id"])):
            user_embedding = user_embeds[user_id].to(self.device)
            user_token_id = self.tokenizer.convert_tokens_to_ids(self.args.USER_EMB)
            replacements = {user_token_id: user_embedding}
            modified_embs = self._modify_embedding_matrix(replacements)
            # Add few-shot tokens into the modified embeddings.
            modified_embs = self._add_few_shot_tokens(modified_embs, user_embeds)
            input_emb = F.embedding(input_tokens[i], modified_embs)
            batch_embeddings.append(input_emb)

        batch_embeddings = torch.stack(batch_embeddings)
        return self._run_model(batch_embeddings, attention_masks, input_tokens, target_masks)


