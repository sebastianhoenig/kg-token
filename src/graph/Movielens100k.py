import pandas as pd
import torch
import numpy as np
from src.graph.FeatureEncoder import GenresEncoder, TextEncoder
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected


class MovieLens:
    """The MovieLens dataset with 100k ratings
    Args:
        path (str): The path to the unzipped MovieLens 100k dataset from
        https://files.grouplens.org/datasets/movielens/ml-100k.zip
    """
    def __init__(self, path: str, device: str):
        self.path = path
        self.device = device
        self.genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
                       'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.data = None

    def _load_data(self):
        movies = pd.read_csv(self.path + 'u.item', sep='|', encoding='latin-1', header=None)
        movies.columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + self.genres

        ratings = pd.read_csv(self.path + 'u.data', sep='\t', header=None)
        ratings.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

        users = pd.read_csv(self.path + 'u.user', sep='|', header=None)
        users.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

        movie_id_mapping = {movie_id: idx for idx, movie_id in enumerate(movies['movie_id'])}
        user_id_mapping = {user_id: idx for idx, user_id in enumerate(users['user_id'])}
        movies['movie_id'] = movies['movie_id'].map(movie_id_mapping)
        users['user_id'] = users['user_id'].map(user_id_mapping)
        ratings['movie_id'] = ratings['movie_id'].map(movie_id_mapping)
        ratings['user_id'] = ratings['user_id'].map(user_id_mapping)
        return movies, ratings, users

    def _encode_ratings(self, ratings: pd.DataFrame) -> tuple[np.array, np.array]:
        filtered_ratings_pos = ratings[ratings['rating'] > 3]
        src_pos = filtered_ratings_pos['user_id'].values
        dst_pos = filtered_ratings_pos['movie_id'].values

        filtered_ratings_neg = ratings[ratings['rating'] <= 3]
        src_neg = filtered_ratings_neg['user_id'].values
        dst_neg = filtered_ratings_neg['movie_id'].values

        src_all = np.concatenate([src_pos, src_neg])
        dst_all = np.concatenate([dst_pos, dst_neg])
        labels = np.concatenate([np.ones(len(src_pos)), np.zeros(len(src_neg))])  # 1 for likes, 0 for dislikes

        return np.array([src_all, dst_all]), labels

    def create_graph(self):
        movies, ratings, users = self._load_data()

        movie_titles = TextEncoder()(movies, ['movie_title'])
        movie_genres = GenresEncoder()(movies, self.genres)
        user_ages = torch.Tensor(users['age'].values).to(self.device)  # Move to device
        movie_features = torch.cat((movie_titles, movie_genres), dim=1).to(self.device)  # Move to device
        index, labels = self._encode_ratings(ratings)

        data = HeteroData()
        data['movie'].x = movie_features
        data['user'].x = user_ages.view(-1, 1)
        data['user', 'likes', 'movie'].edge_index = torch.tensor(index, dtype=torch.long).to(self.device)  # Move to device
        data['user', 'likes', 'movie'].edge_labels = torch.tensor(labels, dtype=torch.float).to(self.device)
        data = ToUndirected()(data).to(self.device)  # Convert to undirected and move to device
        self.data = data

