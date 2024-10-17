import torch.nn as nn

from audio_tokens_config import AudioTokensConfig
from models import (
    BaselineMLPClassifier,
    CNNClassifier,
    CustomBertClassifier,
    SimpleLSTMTokenClassifier,
    SimpleTokenClassifier,
)


def get_model(config: AudioTokensConfig) -> nn.Module:
    if config.model_type == "lstm":
        return SimpleLSTMTokenClassifier(
            vocab_size=config.vocab_size,
            embed_dim=config.lstm_embed_dim,
            hidden_dim=config.lstm_hidden_dim,
            num_layers=config.num_layers,
            num_classes=config.num_classes,
            dropout=config.dropout,
        )
    elif config.model_type == "simple":
        return SimpleTokenClassifier(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_classes=config.num_classes,
        )
    elif config.model_type == "bert":
        return CustomBertClassifier(
            vocab_size=config.vocab_size,
            num_hidden_layers=config.num_layers,
            num_classes=config.num_classes,
            hidden_size=config.hidden_size,
        )
    elif config.model_type == "cnn":
        return CNNClassifier(num_classes=config.num_classes)
    elif config.model_type == "baseline":
        return BaselineMLPClassifier(
            num_classes=config.num_classes,
        )
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
