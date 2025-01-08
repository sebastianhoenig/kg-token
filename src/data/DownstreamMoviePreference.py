from torch_geometric.data import HeteroData
from typing import List, Dict, Union
from torch.utils.data import Dataset


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
            user_id = row['user_id']
            gt_3_movies = row['random_movies_gt_3']
            lte_3_movies = row['random_movies_lte_3']

            # Generate 5 pairs for each user
            for i, (movie_gt, movie_lte) in enumerate(zip(gt_3_movies, lte_3_movies)):
                question = (
                    f"""Output only "Yes" or "No".
                    Question: Does user {self.config.USER_EMB} prefer movie {self.config.MOVIE_GT_EMB} 
                    over movie {self.config.MOVIE_LTE_EMB}?
                    Answer: """
                )

                # Store data for this pair
                qa_dict[len(qa_dict)] = {
                    "question": question,
                    "answer": "Yes",
                    "user_id": int(user_id),
                    "movie_gt_id": int(movie_gt),
                    "movie_lte_id": int(movie_lte),
                }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

