import torch.nn as nn
from transformers import BertModel, BertConfig


class CustomBertClassifier(nn.Module):
    def __init__(
        self, vocab_size, num_hidden_layers, num_classes, hidden_size=768
    ):
        super(CustomBertClassifier, self).__init__()
        config = BertConfig(
            vocab_size=vocab_size,
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
        )
        self.bert = BertModel(config)
        self.classifier = nn.Linear(
            config.hidden_size, num_classes
        )  # Add a linear layer for classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[
            :, 0, :
        ]  # Use the [CLS] token representation
        logits = self.classifier(cls_output)  # Pass through the linear layer
        return logits
