from torch_geometric.data import HeteroData
from typing import List, Dict, Union
from torch.utils.data import Dataset


# Good prompt for no-nonsense answers
# Question: Is the user {self.config.USER_EMB} "Young", "Adult" or "Old"? Answer strictly with one of the following: Young, Adult, or Old. No additional words, punctuation, or formatting.\nAnswer:
#f"""Estimate the age of one of the users based on the following representation: {self.config.USER_EMB}. Answer strictly with Young, Adult, or Old. No additional words, punctuation, or formatting.\nAnswer: """,
class AgeDataset(Dataset):
    def __init__(self, graph: HeteroData, config):
        super().__init__()
        self.graph = graph
        self.config = config
        self.data = self._prepare_dict()

    def _prepare_dict(self) -> Dict[int, Dict[str, Union[str, List[int]]]]:
        qa_dict = {}

        labels = self.graph["user"].age

        few_shot_examples = []
        for i in range(self.config.num_few_shot):
            if get_label(labels[i]) == "Young":
                continue
            few_shot_examples.append(f"Question: Is the user {self.config.SPECIAL_TOKENS[i+2]} Young, Adult or Old?.\nAnswer: {get_label(labels[i])}")

        few_shot_prompt = "\n\n".join(few_shot_examples)

        for ind, age in enumerate(labels):
            label = get_label(age)
            qa_dict[ind] = {
                "question": f"""{few_shot_prompt}\n\n Question: Is the user {self.config.USER_EMB} Young, Adult or Old?\nAnswer: """,
                "answer": label,
                "user_id": ind,
            }

        return qa_dict

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def get_label(age):
    label = "Young" if age < 25 else "Adult" if age < 50 else "Old"
    return label
