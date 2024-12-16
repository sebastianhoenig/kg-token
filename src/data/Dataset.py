from torch_geometric.data import HeteroData
from typing import List, Dict, Union
import numpy as np
import random
from torch.utils.data import Dataset
from src.models.llm import LLM


class GraphQADataset(Dataset):
    """QA LinkPrediction Dataset for Movielens Graph"""

    def __init__(self, graph: HeteroData, config):
        super().__init__()
        self.graph = graph
        self.config = config
        self.data = self._prepare_qa_dict()

    def _prepare_qa_dict(self) -> Dict[int, Dict[str, Union[str, List[int]]]]:
        qa_dict = {}

        user_movie_edges = self.graph["user", "likes", "movie"].edge_index
        labels = self.graph["user", "likes", "movie"].edge_labels

        for ind, (user_idx, movie_idx) in enumerate(zip(*user_movie_edges)):
            qa_dict[ind] = {
                "question": f"Question: Does user {self.config.USER_EMB} like movie {self.config.MOVIE_EMB}?\n\nAnswer: ",
                "answer": "Yes" if labels[ind] == 1 else "No",
                "user_id": user_idx,
                "movie_id": movie_idx,
            }

        return qa_dict

    def graphqa_ds(self):
        pass
        """
        encoded_qa = list(qa_dict.values())
        out = []

        tokenizer = self.llm_wrapper.get_tokenizer()
        for qa in encoded_qa:
            question = qa["question"]
            answer = qa["answer"]
            user_id = qa["user_id"]
            movie_id = qa["movie_id"]
            query_tokens = tokenizer(question, add_special_tokens=False)["input_ids"]
            answer_tokens = tokenizer(answer, add_special_tokens=False)["input_ids"]
            BOS_TOKEN = tokenizer.bos_token_id  # TODO CHANGE WHEN NO LONGER GPT2
            EOS_TOKEN = tokenizer.eos_token_id
            PAD_TOKEN = tokenizer.eos_token_id  # TODO CHANGE AS WELL WHEN NO LONGER GPT2
            max_tokens = 20
            input_tokens = np.array([BOS_TOKEN] + query_tokens + answer_tokens + [EOS_TOKEN])
            target_mask = np.zeros_like(input_tokens)
            target_mask[len(query_tokens) + 1] = 1  # TRYING THIS OUT - REMOVING EOS TOKEN FROM TARGET MASK
            orig_len = len(query_tokens) + len(answer_tokens) + 1
            input_tokens = np.pad(input_tokens, [[0, max_tokens - orig_len-1]], constant_values=PAD_TOKEN)
            target_mask = np.pad(target_mask, [[0, max_tokens - orig_len-1]], constant_values=0)
            attention_mask = np.ones_like(input_tokens)
            attention_mask[input_tokens == PAD_TOKEN] = 0
            out.append((input_tokens, target_mask, attention_mask, user_id, movie_id))

        return out"""

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
