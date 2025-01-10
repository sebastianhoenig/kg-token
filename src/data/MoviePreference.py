from torch_geometric.data import HeteroData
from typing import List, Dict, Union
from torch.utils.data import Dataset
import numpy as np


class GraphQAPreferenceDataset(Dataset):
    """QA Preference Dataset for MovieLens Graph"""

    def __init__(self, graph: HeteroData, config, user_movie_data):
        super().__init__()
        self.graph = graph
        self.config = config
        self.user_movie_data = user_movie_data  # DataFrame with user and movie pairs
        self.data = self._prepare_qa_dict()

    def _prepare_qa_dict(self) -> Dict[int, Dict[str, Union[str, List[int]]]]:
        qa_dict = {}

        # Iterate through each user and construct movie pairs
        for user_idx, row in self.user_movie_data.iterrows():
            user_id = int(row['user_id'])
            gt_3_movies = eval(row['movies_rated_gt_3'])
            lte_3_movies = eval(row['movies_rated_lte_3'])

            pairs = list(zip(gt_3_movies, lte_3_movies))
            if len(pairs) < 3:
                continue

            for i, (gt_movie, lt_movie) in enumerate(pairs):
                # Randomly assign Movie1 and Movie2 roles
                if np.random.rand() > 0.5:
                    movie1, movie2 = gt_movie, lt_movie
                    label = "1"
                    movie1_emb = self.config.MOVIE_GT_EMB
                    movie2_emb = self.config.MOVIE_LTE_EMB
                else:
                    movie1, movie2 = lt_movie, gt_movie
                    label = "2"
                    movie1_emb = self.config.MOVIE_LTE_EMB
                    movie2_emb = self.config.MOVIE_GT_EMB

                question = (
                  f"""IMPORTANT: Output must strictly be a single word: either "1" or "2". Do not include any additional words, punctuation, or characters.\n
                  Question: Does user {self.config.USER_EMB} prefer the first movie (1: {movie1_emb}) or the second movie (2: {movie2_emb})?\nAnswer: """
                 )

                qa_dict[len(qa_dict)] = {
                    "question": question,
                    "answer": label,
                    "user_id": user_id,
                    "movie1_id": movie1,
                    "movie2_id": movie2,
                    "movie1_emb": movie1_emb,
                    "movie2_emb": movie2_emb,
                }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

