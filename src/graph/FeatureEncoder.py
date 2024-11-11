import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from typing import List


class Encoder:

    def __call__(self, df: pd.DataFrame, columns: List[str] = None):
        if columns is None:
            raise ValueError("No columns specified.")
        return self.encode(df, columns)

    def encode(self, df: pd.DataFrame, columns: List[str]) -> torch.Tensor:
        raise NotImplementedError("Subclasses should implement this method.")


class GenresEncoder(Encoder):

    def encode(self, df: pd.DataFrame, columns: List[str]) -> torch.Tensor:
        return torch.Tensor(df[columns].values)


class TextEncoder(Encoder):

    def __init__(self, device: str = 'cpu'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

    def encode(self, df: pd.DataFrame, columns: List[str]) -> torch.Tensor:
        with torch.no_grad():
            column = df[columns].values
            if column.ndim == 2 and column.shape[1] == 1:
                column = column.squeeze()
            x = self.model.encode(column)
        return torch.Tensor(x)

