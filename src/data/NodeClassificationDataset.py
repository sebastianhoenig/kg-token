"""
Dataset for the Age Classification Supervised Learning Task.
"""

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
        self.val = None
        self.mode = "train"
        self._prepare_dict()

    def _prepare_dict(self):
        train_lst = []
        val_lst = []
        test_lst = []

        labels = self.graph["user"].age

        # Perform an 80/20 train-test split
        train_indices, test_indices = train_test_split(
            range(len(labels)), test_size=0.2, random_state=42
        )

        test_indices, val_indices = train_test_split(
            test_indices, test_size=0.5, random_state=42
        )

        # Create train and test dictionaries
        for ind in train_indices:
            age = labels[ind]
            label = get_label_age(age)
            train_lst.append({
                "question": f"""Question: Is the user {self.config.USER_EMB} Young, Adult or Old?\nAnswer: """,
                "answer": label,
                "user_id": ind,
            })

        for ind in val_indices:
            age = labels[ind]
            label = get_label_age(age)
            val_lst.append({
                "question": f"""Question: Is the user {self.config.USER_EMB} Young, Adult or Old?\nAnswer: """,
                "answer": label,
                "user_id": ind,
            })

        for ind in test_indices:
            age = labels[ind]
            label = get_label_age(age)
            test_lst.append({
                "question": f"""Question: Is the user {self.config.USER_EMB} Young, Adult or Old?\nAnswer: """,
                "answer": label,
                "user_id": ind,
            })

        self.data = train_lst
        self.val = val_lst
        self.test = test_lst

    def set_mode(self, mode: str):
        """
        Set the mode for the dataset ('train' or 'test').
        """
        if mode not in ["train", "val", "test"]:
            raise ValueError("Mode must be 'train', 'val', or 'test'.")
        self.mode = mode

    def __len__(self):
        """
        Return the length of the current dataset based on the mode.
        """
        if self.mode == "train":
            return len(self.data)
        elif self.mode == "val":
            return len(self.val)
        else:
            return len(self.test)

    def __getitem__(self, idx):
        """
        Get an item from the dataset based on the mode.
        """
        if self.mode == "train":
            current_data = self.data
        elif self.mode == "val":
            current_data = self.val
        else:
            current_data = self.test

        return current_data[idx]


def get_label_age(age):
    label = "Young" if age < 35 else "Adult" if age < 50 else "Old"
    return label
