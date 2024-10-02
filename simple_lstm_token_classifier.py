import torch
import torch.nn as nn


class SimpleLSTMTokenClassifier(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.0
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, attention_masks=None):
        lengths = attention_masks.sum(1)
        lengths = lengths.cpu().to(torch.int64)  # pack_padded_sequence needs this

        embedded = self.embedding(x,)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        _, (hidden, _) = self.lstm(packed)
        last_output = torch.cat((hidden[-2], hidden[-1]), dim=1)
        last_output = self.dropout(last_output)
        return self.fc(last_output)
