from torch_geometric.data import HeteroData
from typing import Dict, Union
from torch.utils.data import Dataset


class GraphRegressionDataset(Dataset):

    def __init__(self, graph: HeteroData, config):
        super().__init__()
        self.graph = graph
        self.config = config
        self.data = self._prepare_qa_dict()

    def _prepare_qa_dict(self) -> Dict[int, Dict[str, Union[str, int]]]:
        qa_dict = {}

        user_movie_edges = self.graph["user", "likes", "movie"].edge_index
        ratings = self.graph["user", "likes", "movie"].edge_label  # Ratings assumed to be 1 to 5

        for ind, (user_idx, movie_idx) in enumerate(zip(*user_movie_edges)):
            qa_dict[ind] = {
                "question": f"""Predict the rating (from 1 to 5) for the following interaction.\n"""
                             f"User embedding: {self.config.USER_EMB}\n"
                             f"Movie embedding: {self.config.MOVIE_EMB}\n"
                             f"Rating: ",
                "answer": str(ratings[ind].item()),  # Ensure rating is a scalar value
                "user_id": user_idx,
                "movie_id": movie_idx,
            }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]