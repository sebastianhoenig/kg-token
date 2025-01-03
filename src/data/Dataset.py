from torch_geometric.data import HeteroData
from typing import List, Dict, Union
from torch.utils.data import Dataset


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
        labels = self.graph["user", "likes", "movie"].edge_label

        occupations = self.graph["user"].occupation
        ages = self.graph["user"].age
        genders = self.graph["user"].gender

        for ind, (user_idx, movie_idx) in enumerate(zip(*user_movie_edges)):
            qa_dict[ind] = {
                "question": f"""Output only "Yes" or "No".\nQuestion: Does user {self.config.USER_EMB} like movie {self.config.MOVIE_EMB}?\nAnswer: """,
                "answer": "Yes" if labels[ind] == 1 else "No",
                "user_id": user_idx,
                "movie_id": movie_idx,
            }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
