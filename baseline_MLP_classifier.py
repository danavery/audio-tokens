import torch
import torch.nn as nn


class BaselineMLPClassifier(nn.Module):
    def __init__(self, num_classes):
        super(BaselineMLPClassifier, self).__init__()
        self.num_classes = num_classes
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.initialized = False

    def _setup_model(self, x):
        input_size = x.shape[1]  # Infer input size from the batch
        self.fc1 = nn.Linear(input_size, 512).to(self.device)
        self.fc2 = nn.Linear(512, 256).to(self.device)
        self.fc3 = nn.Linear(256, self.num_classes).to(self.device)
        self.initialized = True  # Set the flag to avoid re-initialization

    def forward(self, x, options):
        if not self.initialized:
            self._setup_model(x)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
