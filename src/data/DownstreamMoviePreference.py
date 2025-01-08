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

            # Ensure we have at least one pair for one-shot and 4 pairs for evaluation
            pairs = list(zip(gt_3_movies, lte_3_movies))[:5]
            if len(pairs) < 5:
                continue

            # Use the first pair as the one-shot example
            one_shot_gt_movie, one_shot_lt_movie = pairs[0]
            if np.random.rand() > 0.5:
                one_shot_movie1, one_shot_movie2 = one_shot_gt_movie, one_shot_lt_movie
                one_shot_label = "Yes"
                one_shot_movie1_emb = self.config.OS_MOVIE_GT_EMB
                one_shot_movie2_emb = self.config.OS_MOVIE_LTE_EMB
            else:
                one_shot_movie1, one_shot_movie2 = one_shot_lt_movie, one_shot_gt_movie
                one_shot_label = "No"
                one_shot_movie1_emb = self.config.OS_MOVIE_LTE_EMB
                one_shot_movie2_emb = self.config.OS_MOVIE_GT_EMB

            one_shot_example = (
                f"""Output only the single word "Yes" or "No" without additional punctuation.\n"
                Question: Does user {self.config.USER_EMB} prefer the first movie (ID: {one_shot_movie1_emb})
                over the second movie (ID: {one_shot_movie2_emb})?\nAnswer: {one_shot_label}"""
            )

            # Use the remaining 4 pairs for evaluation
            for i, (gt_movie, lt_movie) in enumerate(pairs[1:]):
                # Randomly assign Movie1 and Movie2 roles
                if np.random.rand() > 0.5:
                    movie1, movie2 = gt_movie, lt_movie
                    label = "Yes"
                    movie1_emb = self.config.MOVIE_GT_EMB
                    movie2_emb = self.config.MOVIE_LTE_EMB
                else:
                    movie1, movie2 = lt_movie, gt_movie
                    label = "No"
                    movie1_emb = self.config.MOVIE_LTE_EMB
                    movie2_emb = self.config.MOVIE_GT_EMB

                question = (
                    f"""{one_shot_example}Output only the single word "Yes" or "No" without additional punctuation.\n"
                    f"Question: Does user {self.config.USER_EMB} prefer the first movie (ID: {movie1_emb}) \
                    orover the second movie (ID: {movie2_emb})?\nAnswer: """
                )

                # Store data for this pair
                qa_dict[len(qa_dict)] = {
                    "question": question,
                    "answer": label,
                    "user_id": user_id,
                    "movie1_id": movie1,
                    "movie2_id": movie2,
                    "movie1_emb": movie1_emb,
                    "movie2_emb": movie2_emb,
                    "one_shot_movie1_id": one_shot_movie1,
                    "one_shot_movie2_id": one_shot_movie2,
                    "one_shot_movie1_emb": one_shot_movie1_emb,
                    "one_shot_movie2_emb": one_shot_movie2_emb,
                }

        return qa_dict


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

