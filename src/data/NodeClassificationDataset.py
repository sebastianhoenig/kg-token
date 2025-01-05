from torch_geometric.data import HeteroData
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class NodeClassificationDataset(Dataset):
    def __init__(self, graph: HeteroData, config):
        super().__init__()
        self.graph = graph
        self.config = config
        self.data = None
        self.test = None
        self.mode = "train"
        self._prepare_dict()

    def _prepare_dict(self):
        train_dict = {}
        test_dict = {}

        labels = self.graph["user"].age

        # Perform an 80/20 train-test split
        train_indices, test_indices = train_test_split(
            range(len(labels)), test_size=0.2, random_state=42
        )

        # Create train and test dictionaries
        for ind in train_indices:
            age = labels[ind]
            label = get_label_age(age)
            train_dict[ind] = {
                "question": f"""Question: Is the user {self.config.USER_EMB} Young, Adult or Old?\nAnswer: """,
                "answer": label,
                "user_id": ind,
            }

        for ind in test_indices:
            age = labels[ind]
            label = get_label_age(age)
            test_dict[ind] = {
                "question": f"""Question: Is the user {self.config.USER_EMB} Young, Adult or Old?\nAnswer: """,
                "answer": label,
                "user_id": ind,
            }

        self.data = train_dict
        self.test = test_dict

    def set_mode(self, mode: str):
        """
        Set the mode for the dataset ('train' or 'test').
        """
        if mode not in ["train", "test"]:
            raise ValueError("Mode must be 'train' or 'test'.")
        self.mode = mode

    def __len__(self):
        """
        Return the length of the current dataset based on the mode.
        """
        return len(self.data if self.mode == "train" else self.test)

    def __getitem__(self, idx):
        """
        Get an item from the dataset based on the mode.
        """
        current_data = self.data if self.mode == "train" else self.test
        return current_data[idx]


def get_label_age(age):
    label = "Young" if age < 35 else "Adult" if age < 50 else "Old"
    return label
