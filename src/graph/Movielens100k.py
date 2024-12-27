import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected
from typing import Any, Dict, Optional


class MovieLens:
    def __init__(self, args):
        self.path = args.dataset_path
        self.device = args.device
        self.args = args
        self.genres = ['unknown', 'Action', 'Adventure', 'Animation', 'Childrens', 'Comedy', 'Crime',
                       'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery',
                       'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']
        self.train = None
        self.test = None

    @staticmethod
    def read_csv(path: str, header: Optional[Any] = None, sep: str = "|") -> pd.DataFrame:
        return pd.read_csv(path, header=header, sep=sep, encoding='latin-1')

    @staticmethod
    def load_node_csv(df: pd.DataFrame, encoders: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        node_mapping = {index: i for i, index in enumerate(df.index.unique())}
        node_embedding = None
        if encoders is not None:
            node_name_tmp = []
            for col, encoder in encoders.items():
                if col in df.columns:
                    encoded = encoder(df[col])
                    if encoded.dim() == 1:
                        node_name_tmp.append(encoded.view(-1, 1))
                    else:
                        node_name_tmp.append(encoded)
                else:
                    encoded = encoder(col)
                    if encoded.dim() == 1:
                        node_name_tmp.append(encoded.view(-1, 1))
                    else:
                        node_name_tmp.append(encoded)

            node_embedding = torch.cat(node_name_tmp, dim=-1)
        return {"node_embedding": node_embedding, "node_mapping": node_mapping}

    @staticmethod
    def load_edge_csv(df: pd.DataFrame, **kwargs: Any) -> Dict[str, Any]:
        df.reset_index(inplace=True)
        src = [kwargs["src_mapping"][index] for index in df[kwargs["src_index_col"]]]
        dst = [kwargs["dst_mapping"][index] for index in df[kwargs["dst_index_col"]]]
        edge_index = torch.tensor([src, dst])

        edge_label = None
        if kwargs["encoders"] is not None:
            edge_attrs = [encoder(df[col]) for col, encoder in kwargs["encoders"].items()]
            edge_label = torch.cat(edge_attrs, dim=-1)

        return {"edge_index": edge_index, "edge_label": edge_label}

    @staticmethod
    def assign_node_property(
            data: HeteroData, node_name: str, is_feature_available: bool, node_tensor: Dict[str, Any]
    ) -> HeteroData:
        if is_feature_available:
            data[node_name].x = node_tensor[node_name]["node_embedding"]
            return data

        data[node_name].num_nodes = len(node_tensor[node_name]["node_mapping"])
        return data

    @staticmethod
    def assign_edge_property(
            data: HeteroData, src_name: str, dst_name: str, relation: str, edge_properties: Dict[str, Any]
    ) -> HeteroData:
        data[(src_name, relation, dst_name)].edge_index = edge_properties["edge_index"]
        data[(src_name, relation, dst_name)].edge_label = edge_properties["edge_label"]
        return data

    def create_graph(self):
        # Load movies and users
        users = self.read_csv(self.path + 'u.user', sep='|', header=None)
        users.columns = ['user_id', 'age', 'gender', 'occupation', 'zip_code']

        movies = self.read_csv(path=self.path + 'u.item', header=None, sep='|')
        movies.columns = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL'] + self.genres

        ratings_train = self.read_csv(path=self.path + 'ua.base', header=None, sep='\t')
        ratings_train.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

        ratings_test = self.read_csv(path=self.path + 'ua.test', header=None, sep='\t')
        ratings_test.columns = ['user_id', 'movie_id', 'rating', 'timestamp']

        # Prepare nodes
        src_node = dict()
        dst_node = dict()

        src_node["user"] = self.load_node_csv(
            df=users.set_index('user_id'),
            encoders={
                "occupation": OneHotColumn(device=self.device),
                "gender": OneHotColumn(device=self.device),
                #"age": NumericalColumn(),
            })
        dst_node["movie"] = self.load_node_csv(
            df=movies.set_index('movie_id'),
            encoders={
                "movie_title": TokenEmbedding(device=self.device),
                "genres": GenresColumn(movies[self.genres])
            },
        )

        def prepare_edges(ratings: pd.DataFrame):
            src = ratings['user_id'].values - 1
            dst = ratings['movie_id'].values - 1
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            if self.args.rating_type == 'binary':
                edge_label = torch.tensor((ratings['rating'] > 3).astype(float).values, dtype=torch.float)
            else:
                edge_label = torch.tensor(ratings['rating'].astype(float).values, dtype=torch.float)
            return {"edge_index": edge_index, "edge_label": edge_label}

        edge_properties_train = prepare_edges(ratings_train)
        edge_properties_test = prepare_edges(ratings_test)

        # Create data object
        data_train = HeteroData()
        data_train = self.assign_node_property(
            data=data_train, node_name="user", is_feature_available=True, node_tensor=src_node
        )
        data_train = self.assign_node_property(
            data=data_train, node_name="movie", is_feature_available=True, node_tensor=dst_node
        )
        data_train = self.assign_edge_property(
            data=data_train, src_name="user", dst_name="movie", relation="likes", edge_properties=edge_properties_train
        )

        data_train["user"].x = torch.eye(data_train["user"].num_nodes, device=self.device)#torch.cat([data_train["user"].x, torch.eye(data_train["user"].num_nodes, device=self.device)], dim=1)

        del data_train["user"].num_nodes

        data_test = HeteroData()
        data_test = self.assign_node_property(
            data=data_test, node_name="user", is_feature_available=True, node_tensor=src_node
        )
        data_test = self.assign_node_property(
            data=data_test, node_name="movie", is_feature_available=True, node_tensor=dst_node
        )
        data_test = self.assign_edge_property(
            data=data_test, src_name="user", dst_name="movie", relation="likes", edge_properties=edge_properties_test
        )

        data_test["user"].x = torch.eye(data_test["user"].num_nodes, device=self.device)#torch.cat([data_test["user"].x, torch.eye(data_test["user"].num_nodes, device=self.device)], dim=1)

        del data_test["user"].num_nodes

        # Convert to undirected and finalize
        self.train = ToUndirected()(data_train).to(self.device)
        self.test = ToUndirected()(data_test).to(self.device)

        del self.train["movie", "rev_likes", "user"].edge_label
        del self.test["movie", "rev_likes", "user"].edge_label


class GenresColumn:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __call__(self, col) -> torch.Tensor:
        return torch.tensor(self.df.values, dtype=torch.float32)


class NumericalColumn:
    def __call__(self, df) -> torch.Tensor:
        return torch.tensor(df.values, dtype=torch.float32)


class OneHotColumn:
    def __init__(self, device):
        self.device = device

    def __call__(self, df) -> torch.Tensor:
        return torch.tensor(pd.get_dummies(df).values, dtype=torch.float32, device=self.device)


class TokenEmbedding:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: Any = None):
        self.device = device
        self.model = SentenceTransformer(model_name, device=device)

    @torch.no_grad()
    def __call__(self, df) -> torch.Tensor:
        x = self.model.encode(
            df.values,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device,
        )
        return x.cpu()


