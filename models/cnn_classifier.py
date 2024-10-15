import torch
import torch.nn as nn


class CNNClassifier(nn.Module):
    def __init__(self, num_classes):
        super(CNNClassifier, self).__init__()
        self.num_classes = num_classes
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        self.initialized = False

    def _setup_model(self, x):
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1).to(self.device)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1).to(self.device)
        self.pool = nn.MaxPool2d(2, 2).to(self.device)
        self.relu = nn.ReLU()

        with torch.no_grad():
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))

        input_size = x.view(x.size(0), -1).size(1)
        self.fc1 = nn.Linear(input_size, 256).to(self.device)
        self.fc2 = nn.Linear(256, self.num_classes).to(self.device)

    def forward(self, x, options):
        x = x.unsqueeze(1).to(self.device)
        if not self.initialized:
            self._setup_model(x)
            self.initialized = True

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(x.size(0), -1)  # Flatten

        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
