from torch_geometric.data import HeteroData
from typing import Dict, Union
from torch.utils.data import Dataset


class MoviePreferenceDataset(Dataset):
    def __init__(self, graph: HeteroData, config):
        super().__init__()
        self.graph = graph
        self.config = config
        self.data = self._prepare_dict()

    def _prepare_dict(self) -> Dict[int, Dict[str, Union[str, int]]]:
        qa_dict = {}

        # Iterate over user embeddings or features in the graph
        for ind, user in enumerate(self.graph["user"]):
            qa_dict[ind] = {
                "question": (
                    f"Question: Would this user {self.config.USER_EMB} like a new action movie and why?\n"
                    f"Answer strictly in the following format: 'Yes, because [reason]' or 'No, because [reason]'."
                    f"Provide exactly one sentence as the answer, with no additional words, punctuation, or formatting.\nAnswer: "
                ),
                "user_id": ind,
            }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
