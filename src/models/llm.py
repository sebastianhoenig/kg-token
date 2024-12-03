from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from torch import nn


class LLM(nn.Module):
    def __init__(self, model_name="gpt2", user_emb: str = "<USER>", movie_emb: str = "<MOVIE>"):
        super().__init__()
        self.model_name = model_name
        self.llm = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.USER_EMB = user_emb
        self.MOVIE_EMB = movie_emb
        self._add_user_and_movie_embed()

    def get_llm(self):
        return self.llm

    def get_tokenizer(self):
        return self.tokenizer

    def _add_user_and_movie_embed(self):
        special_tokens_dict = {'additional_special_tokens': [self.USER_EMB, self.MOVIE_EMB]}
        self.tokenizer.add_special_tokens(special_tokens_dict)
        self.llm.resize_token_embeddings(len(self.tokenizer))