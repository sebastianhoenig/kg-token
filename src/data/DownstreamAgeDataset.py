from torch_geometric.data import HeteroData
from typing import List, Dict, Union
from torch.utils.data import Dataset


# Good prompt for no-nonsense answers
# Question: Is the user {self.config.USER_EMB} "Young", "Adult" or "Old"? Answer strictly with one of the following: Young, Adult, or Old. No additional words, punctuation, or formatting.\nAnswer:
class AgeDataset(Dataset):
    def __init__(self, graph: HeteroData, config):
        super().__init__()
        self.graph = graph
        self.config = config
        self.data = self._prepare_dict()

    def _prepare_dict(self) -> Dict[int, Dict[str, Union[str, List[int]]]]:
        qa_dict = {}

        labels = self.graph["user"].age

        for ind, age in enumerate(labels):
            label = "Young" if age < 35 else "Adult" if age < 50 else "Old"
            qa_dict[ind] = {
                "question": f"""Estimate the age of the user, based on the following representation of him: {self.config.USER_EMB}. Answer strictly with Young, Adult, or Old. No additional words, punctuation, or formatting.\nAnswer: """,
                "answer": label,
                "user_id": ind,
            }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]