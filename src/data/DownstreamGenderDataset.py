from torch_geometric.data import HeteroData
from typing import List, Dict, Union
from torch.utils.data import Dataset


class GenderDataset(Dataset):
    def __init__(self, graph: HeteroData, config):
        super().__init__()
        self.graph = graph
        self.config = config
        self.data = self._prepare_dict()

    def _prepare_dict(self) -> Dict[int, Dict[str, Union[str, List[int]]]]:
        qa_dict = {}

        labels = self.graph["user"].gender

        for ind, gender in enumerate(labels):
            label = "Male" if gender == "M" else "Female"
            qa_dict[ind] = {
                "question": f"""Question: Is the user {self.config.USER_EMB} "Male" or "Female"? Answer strictly with one of the following: Male or Female. No additional words, punctuation, or formatting. \nAnswer: """,
                "answer": label,
                "user_id": ind,
            }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]