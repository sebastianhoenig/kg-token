import torch


class TrainingConfig:
    def __init__(self,
                 num_epochs: int = 10,
                 learning_rate: float = 0.001,
                 logging_steps: int = 2,
                 evaluation_steps: int = 2,
                 model_save_path: str = './model',
                 optimizer: str = 'adam',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.logging_steps = logging_steps
        self.evaluation_steps = evaluation_steps
        self.model_save_path = model_save_path
        self.device = device
        self.optimizer = optimizer
