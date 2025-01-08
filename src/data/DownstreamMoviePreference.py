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
            gt_3_movies = eval(row['random_movies_gt_3'])
            lte_3_movies = eval(row['random_movies_lte_3'])

            # Generate 5 pairs for each user
            for i, (gt_movie, lt_movie) in enumerate(zip(gt_3_movies, lte_3_movies)):
                # Randomly assign Movie1 and Movie2 roles
                if np.random.rand() > 0.5:
                    movie1, movie2 = gt_movie, lt_movie
                    label = "Movie1"
                    movie1_emb = self.config.MOVIE_GT_EMB
                    movie2_emb = self.config.MOVIE_LTE_EMB
                else:
                    movie1, movie2 = lt_movie, gt_movie
                    label = "Movie2"
                    movie1_emb = self.config.MOVIE_LTE_EMB
                    movie2_emb = self.config.MOVIE_GT_EMB

                question = (
                    f"""Output only "Movie1" or "Movie2".
                        Question: Does user {self.config.USER_EMB} prefer the first movie, "Movie1" ({movie1_emb}), 
                        over the second movie, "Movie2" ({movie2_emb})?
                        Answer: """
                )

                # Store data for this pair
                qa_dict[len(qa_dict)] = {
                    "question": question,
                    "answer": label,  # Either "Movie1" or "Movie2"
                    "user_id": user_id,
                    "movie1_id": movie1,
                    "movie2_id": movie2,
                }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

