from transformers import AutoModelForCausalLM, AutoTokenizer
from torch import nn


class LLMWrapper(nn.Module):
    def __init__(self, model_name="gpt2", user_emb: str = "<USER>", movie_emb: str = "<MOVIE>"):
        super().__init__()
        self.model_name = model_name
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.USER_EMB = user_emb
        self.MOVIE_EMB = movie_emb
        self._add_user_and_movie_embed()

        for param in self.llm.parameters():
            param.requires_grad = False

    def get_llm(self):
        return self.llm

    def get_tokenizer(self):
        return self.tokenizer

    def _add_user_and_movie_embed(self):
        special_tokens_dict = {'additional_special_tokens': [self.USER_EMB, self.MOVIE_EMB]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.llm.resize_token_embeddings(len(self.tokenizer))

    def update_embeddings(self, user_embedding, movie_embedding):
        user_token_id = self.tokenizer.convert_tokens_to_ids(self.USER_EMB)
        movie_token_id = self.tokenizer.convert_tokens_to_ids(self.MOVIE_EMB)
        if self.model_name == "gpt2":
            self.llm.transformer.wte.weight[movie_token_id] = movie_embedding.clone()
            self.llm.transformer.wte.weight[user_token_id] = user_embedding.clone()
        elif self.model_name == "meta-llama/Llama-3.1-8B-Instruct":
            pass
        # TODO: Add support for other models



