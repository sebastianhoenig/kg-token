from torch_geometric.data import HeteroData
from typing import List, Dict, Union
import numpy as np
from torch.utils.data import Dataset
from src.models.language.LanguageModel import LLMWrapper
from random_word import RandomWords


class GraphQADataset(Dataset):
    """QA LinkPrediction Dataset for Movielens Graph"""

    def __init__(self, graph: HeteroData, llm_wrapper: LLMWrapper, max_tokens: int = 25, random: bool = False):
        super().__init__()
        self.graph = graph
        self.llm_wrapper = llm_wrapper
        self.random = random
        self.data = self.graphqa_ds(max_tokens)

    def _prepare_qa_dict(self) -> Dict[int, Dict[str, Union[str, List[int]]]]:
        qa_dict = {}

        user_movie_edges = self.graph["user", "likes", "movie"].edge_index
        likes = self.graph["user", "likes", "movie"].edge_labels

        if not self.random:
            for ind, (user_idx, movie_idx) in enumerate(zip(*user_movie_edges)):
                answer = "Yes" if likes[ind] == 1 else "No"
                qa_dict[ind] = {
                    "question": f"Q: Does user {self.llm_wrapper.USER_EMB} like movie {self.llm_wrapper.MOVIE_EMB}?\nA: ",
                    "answer": answer,
                    "user_id": user_idx,
                    "movie_id": movie_idx,
                }
        else:
            for ind, (user_idx, movie_idx) in enumerate(zip(*user_movie_edges)):
                r = RandomWords()
                answer = "Yes" if likes[ind] == 1 else "No"
                r1 = r.get_random_word()
                r2 = r.get_random_word()
                r3 = r.get_random_word()
                r4 = r.get_random_word()
                qa_dict[ind] = {
                    "question": f"Q: {r1} {r2} {self.llm_wrapper.USER_EMB} {r3} {r4} {self.llm_wrapper.MOVIE_EMB}?\nA: ",
                    "answer": answer,
                    "user_id": user_idx,
                    "movie_id": movie_idx,
                }

        return qa_dict

    def graphqa_ds(self, max_tokens: int = 25):
        qa_dict = self._prepare_qa_dict()
        encoded_qa = list(qa_dict.values())
        out = []

        tokenizer = self.llm_wrapper.get_tokenizer()
        for qa in encoded_qa:
            question = qa["question"]
            answer = qa["answer"]
            user_id = qa["user_id"]
            movie_id = qa["movie_id"]
            query_tokens = tokenizer(question)["input_ids"]
            answer_tokens = tokenizer(answer)["input_ids"]
            BOS_TOKEN = tokenizer.eos_token_id  # TODO CHANGE WHEN NO LONGER GPT2
            EOS_TOKEN = tokenizer.eos_token_id
            PAD_TOKEN = tokenizer.eos_token_id  # TODO CHANGE AS WELL WHEN NO LONGER GPT2

            input_tokens = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_tokens)
            target_mask[len(query_tokens) + 1:] = 1
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            input_tokens = np.pad(input_tokens, [[0, max_tokens - orig_len-1]], constant_values=PAD_TOKEN)
            target_mask = np.pad(target_mask, [[0, max_tokens - orig_len-1]], constant_values=0)
            attention_mask = np.ones_like(input_tokens)
            attention_mask[input_tokens == PAD_TOKEN] = 0
            out.append((input_tokens, target_mask, attention_mask, user_id, movie_id))

        return out

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
