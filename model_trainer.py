import logging
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from torch.nn.utils.rnn import pad_sequence
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from audio_tokens_config import AudioTokensConfig
from audioset_metadata_processor import AudiosetMetadataProcessor
from custom_bert_classifier import CustomBertClassifier
from model_diagnostics import ModelDiagnostics
from simple_lstm_token_classifier import SimpleLSTMTokenClassifier
from simple_token_classifier import SimpleTokenClassifier
from tokenized_spec_dataset import TokenizedSpecDataset
from set_seed import set_seed

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"


class ModelTrainer:
    def __init__(self, config: AudioTokensConfig):
        self.config = config
        set_seed(self.config.random_seed)
        self.logger = logging.getLogger()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.setup_train_val()
        self.model = self._initialize_model()
        self.optimizer = self._initialize_optimizer()
        self.criterion = nn.BCEWithLogitsLoss()
        self.diagnostics = ModelDiagnostics(model=self.model, criterion=self.criterion)
        self.diagnostic_interval = 1
        if self.config.use_wandb:
            self.run_name = self._setup_wandb()
            self.logger.info(f"wandb run name: {self.run_name}")
        else:
            self.run_name = "no-wandb"

    # def setup_train_val(self):
    #     self.train_files = list(Path(self.config.train_dir).glob("*.npy"))
    #     self.val_files = list(Path(self.config.val_dir).glob("*.npy"))
    #     self.logger.info(f"Training files: {len(self.train_files)}")
    #     self.logger.info(f"Validation files: {len(self.val_files)}")

    def _initialize_model(self):
        if self.config.model_type == "simple":
            return self._get_simple_model()
        elif self.config.model_type == "bert":
            model = CustomBertClassifier(
                vocab_size=self.config.vocab_size,
                num_hidden_layers=self.config.num_layers,
                num_classes=self.config.num_classes,
                device=self.device,
                hidden_size=self.config.hidden_size,
            )
        elif self.config.model_type == "lstm":
            return self._get_lstm_model()
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        model.to(self.device)
        return model

    def _initialize_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def _initialize_data_loaders(self):
        metadata_manager = AudiosetMetadataProcessor(self.config)
        train_dataset = TokenizedSpecDataset(
            config, metadata_manager, split="train",
        )
        val_dataset = TokenizedSpecDataset(
            config, metadata_manager, split="validation"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )
        return train_loader, val_loader

    def collate_fn(self, batch):
        input_ids = [item[0][:512] for item in batch]
        attention_masks = [item[1][:512] for item in batch]
        labels = [item[2] for item in batch]

        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0).long()
        attention_masks = pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        ).float()
        labels = torch.stack(labels).float()
        return input_ids, attention_masks, labels

    def train(self):
        self.train_loader, self.val_loader = self._initialize_data_loaders()
        best_metric = 0

        for epoch in range(self.config.epochs):
            train_loss, train_metrics = self._train_epoch()
            val_loss, val_metrics = self._validate_epoch()

            self._log_epoch_results(
                epoch, train_loss, train_metrics, val_loss, val_metrics
            )

            if epoch % self.diagnostic_interval == 0:
                self.logger.info("starting gradient recording")
                self.diagnostics.check_gradient_flow(
                    epoch=epoch, run_name=self.run_name
                )
                # # this is taking a very long time. investigate.
                # self.logger.info("plotting loss landscape")
                # self.diagnostics.plot_loss_landscape(
                #     epoch=epoch, val_loader=self.val_loader, run_name=self.run_name
                # )
                # self.logger.info("done plotting")

            if val_metrics["mAP"] > best_metric:
                best_metric = val_metrics["mAP"]
                self._save_best_model()

            if self._should_stop_early():
                break
        self._save_best_model()
        return val_loss, val_metrics

    def _train_epoch(self):
        self.model.train()
        return self._run_epoch(self.train_loader, is_training=True)

    def _validate_epoch(self):
        self.model.eval()
        with torch.no_grad():
            return self._run_epoch(self.val_loader, is_training=False)

    def _run_epoch(self, data_loader, is_training):
        total_loss = 0
        all_predictions = []
        all_labels = []

        progress_bar = tqdm(
            data_loader, desc="Training" if is_training else "Validating"
        )
        for batch in progress_bar:
            loss, predictions, labels = self._process_batch(batch, is_training)
            total_loss += loss
            all_predictions.extend(predictions.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())

            if is_training:
                progress_bar.set_postfix({"loss": loss})

        # Concatenate all predictions and labels
        all_predictions = np.concatenate(all_predictions, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)

        metrics = self._compute_metrics(all_predictions, all_labels)
        return total_loss / len(data_loader), metrics

    def _process_batch(self, batch, is_training):
        input_ids, attention_masks, labels = [b.to(self.device) for b in batch]
        outputs = self.model(input_ids, attention_mask=attention_masks)
        loss = self.criterion(outputs, labels)

        # self.logger.debug("Raw outputs:", outputs[0][:5].tolist())
        # self.logger.debug("Sigmoid outputs:", torch.sigmoid(outputs[0][:50]).tolist())
        # self.logger.debug("Labels:", labels[0][:50].tolist())

        if is_training:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return loss.item(), torch.sigmoid(outputs).detach(), labels

    def _compute_metrics(self, predictions, labels):
        # For F1 and Hamming loss, we need binary predictions
        binary_predictions = (predictions > self.config.prediction_threshold).astype(
            int
        )
        return {
            "f1_score_micro": f1_score(labels, binary_predictions, average="micro"),
            "f1_score_macro": f1_score(labels, binary_predictions, average="macro"),
            "hamming_loss": hamming_loss(labels, binary_predictions),
            "mAP": average_precision_score(labels, predictions, average="macro"),
        }

    def _log_epoch_results(
        self, epoch, train_loss, train_metrics, val_loss, val_metrics
    ):
        self.logger.info(f"Epoch {epoch}")
        self.logger.info(
            f"Train Loss: {train_loss:.4f}, Train F1 (macro): {train_metrics['f1_score_macro']:.4f}, Train F1 (micro): {train_metrics['f1_score_micro']:.4f}, Train Hamming Loss: {train_metrics['hamming_loss']:.4f}, Train mAP: {train_metrics['mAP']:.4f}"
        )
        self.logger.info(
            f"Val Loss: {val_loss:.4f}, Val F1 (macro): {val_metrics['f1_score_macro']:.4f}, Val F1 (micro): {val_metrics['f1_score_micro']:.4f}, Val Hamming Loss: {val_metrics['hamming_loss']:.4f}, Val mAP: {val_metrics['mAP']:.4f}"
        )
        if self.config.use_wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_mAP": train_metrics["mAP"],
                    "val_loss": val_loss,
                    "val_mAP": val_metrics["mAP"],
                }
            )

    def _setup_wandb(self):
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="audio-tokens",
            # track hyperparameters and run metadata. send it all!
            config=self.config
        )
        return run.name

    def _should_stop_early(self):
        """early stopping logic"""

    def _save_best_model(self):
        torch.save(self.model.state_dict(), "output/best_model.pth")

    def _get_bert_model(self):
        model = CustomBertClassifier(
            vocab_size=self.vocab_size,
            num_hidden_layers=self.num_layers,
            num_classes=self.config.num_classes,
            device=self.device,
            dropout=self.dropout,
            hidden_size=self.hidden_size,
        ).to(self.device)
        return model

    def _get_simple_model(self):
        model = SimpleTokenClassifier(
            vocab_size=self.config.vocab_size,
            hidden_size=self.config.hidden_size,
            num_classes=self.config.num_classes,
        ).to(self.device)
        return model

    def _get_lstm_model(self):
        model = SimpleLSTMTokenClassifier(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.lstm_embed_dim,
            hidden_dim=self.config.lstm_hidden_dim,
            num_layers=self.config.num_layers,
            num_classes=self.config.num_classes,
            dropout=self.config.dropout,
        )
        model.to(self.device)
        return model


class RNNClassifier(nn.Module):
    def __init__(
        self, num_embeddings, embed_len, hidden_dim, n_layers, target_classes=10
    ):
        super(RNNClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding_layer = nn.Embedding(
            num_embeddings=num_embeddings, embedding_dim=embed_len
        )
        self.rnn = nn.RNN(
            input_size=embed_len,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.linear = nn.Linear(hidden_dim, target_classes)

    def forward(self, X_batch, attention_mask=None):
        embeddings = self.embedding_layer(X_batch)
        hidden = torch.randn(
            self.n_layers, X_batch.size(0), self.hidden_dim, device="cuda"
        )
        output, hidden = self.rnn(embeddings, hidden)
        return self.linear(output[:, -1])


if __name__ == "__main__":
    config = AudioTokensConfig()
    trainer = ModelTrainer(config)
    val_loss, val_accuracy = trainer.train()
    logging.getLogger().info(
        f"Final Validation Loss: {val_loss}, Final Validation Accuracy: {val_accuracy}"
    )
