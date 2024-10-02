import logging
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from audio_tokens_config import AudioTokensConfig
from audioset_metadata_processor import AudiosetMetadataProcessor
from model_diagnostics import ModelDiagnostics
from models import get_model
from set_seed import set_seed
from tokenized_spec_dataset import TokenizedSpecDataset
from metrics_calculator import MetricsCalculator

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"


class ModelTrainer:
    def __init__(self, config: AudioTokensConfig):
        self.config = config
        set_seed(self.config.random_seed)
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        self.model = get_model(self.config).to(self.device)
        self.optimizer = self._initialize_optimizer()
        self.criterion = nn.BCEWithLogitsLoss()
        self.metrics_calculator = MetricsCalculator()

        self.diagnostics = ModelDiagnostics(model=self.model, criterion=self.criterion)
        self.diagnostic_interval = 1
        self.run_name = self._initialize_wandb()

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
                pass
                # self._run_diagnostics(epoch)

            best_metric = self._save_if_best_model(best_metric, val_metrics)

            if self._should_stop_early():
                break
        return val_loss, val_metrics

    def _initialize_data_loaders(self):
        metadata_manager = AudiosetMetadataProcessor(self.config)
        dataset_model = TokenizedSpecDataset
        train_dataset = dataset_model(
            self.config,
            metadata_manager,
            split="train",
        )
        val_dataset = dataset_model(
            self.config, metadata_manager, split="validation"
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=dataset_model.collate_fn,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training_batch_size,
            num_workers=self.config.num_workers,
            collate_fn=dataset_model.collate_fn,
        )
        return train_loader, val_loader

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
            data_loader,
            desc="Training" if is_training else "Validating",
            mininterval=1,
        )
        for batch in progress_bar:
            loss, predictions, labels = self._process_batch(batch, is_training)
            total_loss += loss
            all_predictions.append(predictions.cpu().detach().numpy())
            all_labels.append(labels.cpu().numpy())

        metrics = self.metrics_calculator.compute_metrics(all_predictions, all_labels)
        return total_loss / len(data_loader), metrics

    def _process_batch(self, batch, is_training):
        sequences, metadata = batch
        sequences = sequences.to(self.device)
        attention_masks = metadata["attention_masks"].to(self.device)
        labels = metadata["labels"].to(self.device)

        outputs = self.model(sequences, attention_masks=attention_masks, )
        loss = self.criterion(outputs, labels)
        if is_training:
            self._backpropagate(loss)
        return loss.item(), torch.sigmoid(outputs).detach(), labels

    def _backpropagate(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _initialize_optimizer(self):
        return AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def _run_diagnostics(self, epoch):
        self.logger.info("starting gradient recording")
        self.diagnostics.check_gradient_flow(epoch=epoch, run_name=self.run_name)
        # this is taking a very long time. investigate.
        self.logger.info("plotting loss landscape")
        self.diagnostics.plot_loss_landscape(
            epoch=epoch, val_loader=self.val_loader, run_name=self.run_name
        )
        self.logger.info("done plotting")

    def _log_epoch_results(
        self, epoch, train_loss, train_metrics, val_loss, val_metrics
    ):
        self.logger.info(f"Epoch {epoch}")
        self.logger.info(
            # f"Train Loss: {train_loss:.4f}, Train F1 (macro): {train_metrics['f1_score_macro']:.4f}, Train F1 (micro): {train_metrics['f1_score_micro']:.4f}, Train Hamming Loss: {train_metrics['hamming_loss']:.4f}, Train mAP: {train_metrics['mAP']:.4f}, alt Train mAP: {train_metrics['alt_mAP']:.4f}"
            f"Train Loss: {train_loss:.4f}, Train mAP: {train_metrics['mAP']:.4f}"
        )

        self.logger.info(
            # f"Val Loss: {val_loss:.4f}, Val F1 (macro): {val_metrics['f1_score_macro']:.4f}, Val F1 (micro): {val_metrics['f1_score_micro']:.4f}, Val Hamming Loss: {val_metrics['hamming_loss']:.4f}, Val mAP: {val_metrics['mAP']:.4f}, alt Train mAP: {val_metrics['alt_mAP']:.4f}"
            f"Val Loss: {val_loss:.4f}, Val mAP: {val_metrics['mAP']:.4f}"
        )
        if not self.config.use_wandb:
            return
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_mAP": train_metrics["mAP"],
                "val_loss": val_loss,
                "val_mAP": val_metrics["mAP"],
            }
        )

    def _initialize_wandb(self):
        if self.config.use_wandb:
            run = wandb.init(
                project="audio-tokens",
                # track hyperparameters and run metadata. send it all!
                config=self.config,
            )
            return run.name
        else:
            return "no-wandb"

    def _should_stop_early(self):
        """early stopping logic"""

    def _save_if_best_model(self, best_metric, val_metrics):
        if val_metrics["mAP"] > best_metric:
            self.logger.info(
                f"val mAP of {val_metrics['mAP']:.4f} > {best_metric:.4f}. Saving model."
            )
            best_metric = val_metrics["mAP"]
            torch.save(
                self.model.state_dict(), f"output/{self.run_name}-best_model.pth"
            )
        return best_metric


if __name__ == "__main__":
    config = AudioTokensConfig()
    trainer = ModelTrainer(config)
    val_loss, val_metrics = trainer.train()
    logging.getLogger().info(
        f"Final Validation Loss: {val_loss:.4f}, Final Validation mAP: {val_metrics['mAP']:.4f}"
    )
