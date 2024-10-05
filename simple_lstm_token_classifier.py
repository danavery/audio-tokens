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
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x, options):
        lengths = options["attention_masks"].sum(1)
        lengths = lengths.cpu().to(torch.int64)  # pack_padded_sequence needs this

        if not options.get("use_precomputed_embeddings"):
            # Use the embedding layer if tokens are provided
            embedded = self.embedding(x)
        else:
            # Use the provided embeddings directly
            embedded = x

        # Compact LSTM weights for better memory management (fixes warning)
        self.lstm.flatten_parameters()

        embedded = embedded.float()
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)

        _, (hidden, _) = self.lstm(packed)
        last_output = torch.cat((hidden[-2], hidden[-1]), dim=1)
        last_output = self.relu(last_output)
        last_output = self.dropout(last_output)
        return self.fc(last_output)
