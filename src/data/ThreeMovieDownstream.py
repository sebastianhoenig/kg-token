import numpy as np
from torch.utils.data import Dataset
from typing import Dict, Union, List


class ThreeMoviePreferenceDataset(Dataset):
    def __init__(self, graph, config, user_movie_data):
        super().__init__()
        self.graph = graph
        self.config = config
        self.user_movie_data = user_movie_data
        self.data = self._prepare_qa_dict()

    def _prepare_qa_dict(self) -> Dict[int, Dict[str, Union[str, List[int]]]]:
        qa_dict = {}

        # Iterate through each user and construct movie triplets
        for user_idx, row in self.user_movie_data.iterrows():
            user_id = int(row['user_id'])
            gt_3_movies = eval(row['movies_rated_gt_3'])  # Movies rated greater than 3
            lte_3_movies = eval(row['movies_rated_lte_3'])  # Movies rated less than or equal to 3

            # Ensure there are enough movies to form triplets
            if len(gt_3_movies) < 1 or len(lte_3_movies) < 2:
                continue

            for gt_movie in gt_3_movies:
                # Select two LTE movies for pairing with each GT movie
                lt_movies = np.random.choice(lte_3_movies, size=2, replace=False)

                movies = [gt_movie, lt_movies[0], lt_movies[1]]
                labels = ["1", "2", "3"]

                # Randomize the order of the movies and corresponding labels
                combined = list(zip(movies, labels))
                np.random.shuffle(combined)
                shuffled_movies, shuffled_labels = zip(*combined)

                movie1, movie2, movie3 = shuffled_movies
                movie1_emb, movie2_emb, movie3_emb = (
                    (self.config.MOVIE_GT_EMB if movie == gt_movie else self.config.MOVIE_LTE_EMB) for movie in shuffled_movies
                )

                correct_label = shuffled_labels[movies.index(gt_movie)]

                question = (
                    f"""IMPORTANT: Output must strictly be a single word: either "1", "2", or "3". Do not include any additional words, punctuation, or characters.\n
                    Question: Does user {self.config.USER_EMB} prefer the first movie (1: {movie1_emb}), the second movie (2: {movie2_emb}), or the third movie (3: {movie3_emb})?\nAnswer: """
                )

                qa_dict[len(qa_dict)] = {
                    "question": question,
                    "answer": correct_label,
                    "user_id": user_id,
                    "movie1_id": movie1,
                    "movie2_id": movie2,
                    "movie3_id": movie3,
                    "movie1_emb": movie1_emb,
                    "movie2_emb": movie2_emb,
                    "movie3_emb": movie3_emb,
                }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
