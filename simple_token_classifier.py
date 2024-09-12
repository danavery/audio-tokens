# simple_token_classifier.py
import torch.nn as nn


class SimpleTokenClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x, attention_mask=None):
        x = self.embedding(x)
        x = x.transpose(1, 2)  # [batch, hidden, seq_len]
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)
